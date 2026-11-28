"""C4.5 tự cài: số/danh nghĩa, ô thiếu có trọng số và cắt tỉa bi quan.

Mỗi nút tự tìm Information Gain, Gain Ratio; thuộc tính số có ngưỡng nhị
phân và hiệu chỉnh MDL. Mẫu thiếu được chia trọng số sang các nhánh.
Sau khi dựng cây, thực hiện collapse, thay cây bằng lá và subtree raising.
Đối chiếu cấu hình với J48; không gọi thư viện học máy hay hàm thống kê
phân phối có sẵn. Công thức và phạm vi được ghi trong DOCS/C45.md.
"""

from copy import deepcopy
from math import isfinite, log, log2, sqrt
from numbers import Integral, Real

from ..data_processing.preprocessing import _midpoint
from .common import entropy

EPSILON = 1e-6  # So sánh điểm số cùng dung sai với J48, không bỏ trọng số mẫu.


def normal_quantile(probability):
    """Tự xấp xỉ nghịch đảo chuẩn bằng phân thức Acklam, sai số khoảng 1e-9.

    Chỉ phục vụ cận lỗi khi cắt tỉa, không phải mô hình Naive Bayes Gaussian.
    Nhận p trong (0,1); không dùng scipy/statistics hoặc hàm phân phối sẵn.
    """
    if not 0 < probability < 1:
        raise ValueError("Xác suất phân vị phải nằm trong (0,1).")
    a = (-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924)
    b = (-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857)
    c = (-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878)
    d = (0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742)

    def polynomial(coefficients, value):
        result = coefficients[0]
        for coefficient in coefficients[1:]:
            result = result * value + coefficient
        return result

    if probability < 0.02425 or probability > 0.97575:
        lower = probability < 0.5
        tail = probability if lower else 1 - probability
        q = sqrt(-2 * log(tail))
        value = polynomial(c, q) / (polynomial(d, q) * q + 1)
        return value if lower else -value
    q = probability - 0.5
    r = q * q
    return q * polynomial(a, r) / (polynomial(b, r) * r + 1)


def extra_errors(total, errors, confidence=0.25):
    """Phần lỗi cộng thêm của cận nhị thức một phía có hiệu chỉnh liên tục.

    total và errors là trọng số, có thể không nguyên. Xử lý riêng E=0,
    0<E<1 và E+0.5>=N; phần giữa dùng z của phân vị 1-CF. Công thức tương
    ứng Stats.addErrs của J48; mọi phép tính được viết tại đây.
    """
    if (not isfinite(total) or not isfinite(errors) or total < 0 or
            errors < 0 or errors > total or not 0 < confidence <= 0.5):
        raise ValueError("Tham số ước lượng lỗi C4.5 không hợp lệ.")
    if total == 0:
        return 0.0
    if errors < 1:
        base = total * (1 - confidence ** (1 / total))
        if errors == 0:
            return base
        # E=1 dùng nhánh giới hạn trên khi N<1, không gọi đệ quy sai miền.
        at_one = 0.0 if total < 1 else extra_errors(total, 1.0, confidence)
        return base + errors * (at_one - base)
    if errors + 0.5 >= total:
        return max(total - errors, 0.0)
    z = -normal_quantile(confidence)
    frequency = (errors + 0.5) / total
    z_squared = z * z
    upper_rate = (frequency + z_squared / (2 * total) + z * sqrt(
        frequency * (1 - frequency) / total + z_squared / (4 * total * total)
    )) / (1 + z_squared / total)
    return max(0.0, upper_rate * total - errors)


class C45:
    """C4.5 nhị phân cho nhãn 0/1; có thể trộn cột số và danh nghĩa.

    min_samples_leaf=2 là trọng số tối thiểu ở ít nhất hai nhánh hợp lệ.
    confidence=0.25 điều khiển cận lỗi; CF nhỏ cắt tỉa mạnh hơn. pruned,
    subtree_raising, collapse và mdl_correction mặc định bật theo J48.
    actual_split_points đưa ngưỡng về giá trị train gần nhất không vượt
    trung điểm, để thống nhất cách đặt ngưỡng với J48 mặc định.
    """

    def __init__(self, min_samples_leaf=2, confidence=0.25, pruned=True,
                 subtree_raising=True, collapse=True, mdl_correction=True,
                 actual_split_points=True):
        if (type(min_samples_leaf) is not int or min_samples_leaf < 1 or
                isinstance(confidence, bool) or not isinstance(confidence, Real) or
                not isfinite(confidence) or not 0 < confidence <= 0.5):
            raise ValueError("min_samples_leaf>=1, 0<confidence<=0.5.")
        flags = (pruned, subtree_raising, collapse, mdl_correction, actual_split_points)
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("Các cờ C4.5 phải có kiểu bool.")
        self.config = dict(min_samples_leaf=min_samples_leaf, confidence=float(confidence),
                           pruned=pruned, subtree_raising=subtree_raising, collapse=collapse,
                           mdl_correction=mdl_correction, actual_split_points=actual_split_points)

    def _matrix(self, X, feature_types):
        """Sao chép ma trận n×p: None/NaN là thiếu; nominal nhận số hoặc chuỗi."""
        if isinstance(X, (str, bytes, dict)):
            raise ValueError("X phải là ma trận hai chiều.")
        rows = []
        for row in X:
            if isinstance(row, (str, bytes, dict)):
                raise ValueError("Mỗi hàng X phải là một dãy.")
            values = list(row)
            if len(values) != len(feature_types):
                raise ValueError("X sai số thuộc tính.")
            clean = []
            for value, kind in zip(values, feature_types):
                if value is None or (isinstance(value, Real) and value != value):
                    clean.append(None)
                elif isinstance(value, bool):
                    raise ValueError("Không nhận bool làm số đo hoặc mã danh nghĩa.")
                elif kind == 'nominal' and isinstance(value, str):
                    clean.append(value)
                elif isinstance(value, Real) and isfinite(value):
                    clean.append(float(value))
                else:
                    raise ValueError("Thuộc tính số phải hữu hạn; danh nghĩa là số/chuỗi.")
            rows.append(clean)
        if not rows or not feature_types:
            raise ValueError("X không được rỗng.")
        return rows

    def _counts(self, samples):
        """Đếm tổng trọng số hai lớp, không làm tròn trọng số phân số."""
        counts = [0.0, 0.0]
        for index, weight in samples:
            counts[self._labels[index]] += weight
        return counts

    def fit(self, X, y, feature_types=None, sample_weight=None):
        """Học chỉ từ train; loại trọng số 0, giữ ô thiếu để chia phân số.

        feature_types mặc định toàn numeric; có thể chỉ định nominal cho
        từng cột. sample_weight mặc định 1; phải hữu hạn, không âm và tổng
        dương. Nhãn bắt buộc 0/1, không tự điền nhãn thiếu.
        """
        raw = list(X)
        if not raw:
            raise ValueError("X không được rỗng.")
        feature_types = list(feature_types) if feature_types is not None else ['numeric'] * len(raw[0])
        if any(kind not in ('numeric', 'nominal') for kind in feature_types):
            raise ValueError("feature_types chỉ nhận numeric hoặc nominal.")
        self.feature_types = feature_types
        self._rows = self._matrix(raw, feature_types)
        self._labels = list(y)
        if (len(self._rows) != len(self._labels) or
                any(isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1)
                    for label in self._labels)):
            raise ValueError("y phải có nhãn nguyên 0/1 cho từng hàng.")
        weights = list(sample_weight) if sample_weight is not None else [1.0] * len(raw)
        if (len(weights) != len(raw) or any(isinstance(w, bool) or not isinstance(w, Real) or
                not isfinite(w) or w < 0 for w in weights) or not 0 < sum(weights) < float('inf')):
            raise ValueError("Trọng số phải không âm, hữu hạn và tổng dương.")
        samples = [(i, float(w)) for i, w in enumerate(weights) if w > 0]
        self._train_size = len(samples)
        self.values = [sorted({self._rows[i][j] for i, _ in samples if self._rows[i][j] is not None},
                              key=lambda v: (isinstance(v, str), v)) for j in range(len(feature_types))]
        self.pruning = {'collapsed': 0, 'replaced_by_leaf': 0, 'raised': 0}
        self.root = self._grow(samples)
        self.pruning['before'] = self.tree_size()
        if self.config['collapse']:
            self._collapse(self.root)
        if self.config['pruned']:
            self._prune(self.root)
        self.pruning['after'] = self.tree_size()
        self._discard_training_samples(self.root)
        del self._rows, self._labels, self._train_size
        return self

    def _criteria(self, branches, total):
        """Gain = K/N × gain_known; SplitInfo có thêm một nhóm unknown/N."""
        sizes = [sum(counts) for counts in branches]
        known = sum(sizes)
        if known <= 0:
            return 0.0, 0.0
        parent = [sum(counts[c] for counts in branches) for c in (0, 1)]
        gain = known / total * (entropy(parent) - sum(
            size / known * entropy(counts) for size, counts in zip(sizes, branches)))
        split_info = entropy(sizes + [max(0.0, total - known)])
        return max(0.0, gain), split_info

    def _numeric_split(self, samples, column, total):
        """Duyệt ngưỡng tăng dần bằng bảng đếm tiền tố; chọn gain rồi trừ MDL."""
        ordered = sorted((self._rows[i][column], i, w) for i, w in samples
                         if self._rows[i][column] is not None)
        right = [0.0, 0.0]
        for _, i, w in ordered:
            right[self._labels[i]] += w
        known = sum(right)
        minimum = max(self.config['min_samples_leaf'], min(25.0, 0.1 * known / 2))
        if known < 2 * minimum - EPSILON:
            return None
        left = [0.0, 0.0]
        best, number_of_cuts = None, 0
        for position in range(len(ordered) - 1):
            value, index, weight = ordered[position]
            label = self._labels[index]
            left[label] += weight
            right[label] = max(0.0, right[label] - weight)
            following = ordered[position + 1][0]
            if value + 1e-5 >= following:
                continue
            if min(sum(left), sum(right)) + EPSILON < minimum:
                continue
            number_of_cuts += 1
            gain, split_info = self._criteria([left, right], total)
            if best is None or gain > best['gain'] + EPSILON:
                threshold = _midpoint(value, following)
                if threshold >= following:
                    threshold = value
                best = dict(column=column, kind='numeric', threshold=threshold,
                            gain=gain, split_info=split_info)
        if best is None:
            return None
        if self.config['mdl_correction']:
            best['gain'] -= log2(number_of_cuts) / total
        if best['gain'] <= EPSILON or best['split_info'] <= EPSILON:
            return None
        best['gain_ratio'] = best['gain'] / best['split_info']
        return best

    def _nominal_split(self, samples, column, total):
        """Một nhánh mỗi giá trị danh nghĩa; ít nhất hai nhánh đủ trọng số."""
        values = self.values[column]
        positions = {value: position for position, value in enumerate(values)}
        branches = [[0.0, 0.0] for _ in values]
        for index, weight in samples:
            value = self._rows[index][column]
            if value is not None:
                branches[positions[value]][self._labels[index]] += weight
        if sum(sum(counts) + EPSILON >= self.config['min_samples_leaf'] for counts in branches) < 2:
            return None
        gain, split_info = self._criteria(branches, total)
        return dict(column=column, kind='nominal', values=values, gain=gain,
                    split_info=split_info, gain_ratio=gain / split_info if split_info else 0.0)

    def _select_split(self, samples, total):
        """Lọc gain quanh trung bình rồi chọn gain ratio lớn nhất, hòa chọn cột trước."""
        candidates = []
        for column, kind in enumerate(self.feature_types):
            method = self._numeric_split if kind == 'numeric' else self._nominal_split
            split = method(samples, column, total)
            if split is not None:
                candidates.append(split)
        # Quy tắc C4.5/J48 hạn chế thuộc tính danh nghĩa có quá nhiều giá trị
        # tham gia trung bình khi vẫn có thuộc tính số hoặc ít giá trị.
        all_multivalued = all(kind == 'nominal' and len(self.values[j]) >= 0.3 * self._train_size
                             for j, kind in enumerate(self.feature_types))
        eligible = [s for s in candidates if all_multivalued or s['kind'] == 'numeric' or
                    len(self.values[s['column']]) < 0.3 * self._train_size]
        if not eligible:
            return None
        average_gain = sum(s['gain'] for s in eligible) / len(eligible)
        best = None
        for split in candidates:
            if (split['gain'] >= average_gain - 0.001 and split['gain_ratio'] > EPSILON and
                    (best is None or split['gain_ratio'] > best['gain_ratio'] + EPSILON)):
                best = split
        if best and best['kind'] == 'numeric' and self.config['actual_split_points']:
            # Tìm nhị phân trên giá trị train, không đọc validation/test.
            values, limit = self.values[best['column']], best['threshold']
            low, high = 0, len(values)
            while low < high:
                middle = (low + high) // 2
                if values[middle] <= limit:
                    low = middle + 1
                else:
                    high = middle
            best['threshold'] = values[low - 1]
        return best

    @staticmethod
    def _branch(split, row):
        """Chỉ số nhánh; None khi thiếu hoặc gặp giá trị danh nghĩa mới."""
        value = row[split['column']]
        if value is None:
            return None
        if split['kind'] == 'numeric':
            return int(value > split['threshold'])
        try:
            return split['values'].index(value)
        except ValueError:
            return None

    def _partition(self, split, samples):
        """Chia mẫu thiếu theo w_b = tổng_known_b / tổng_known, bảo toàn trọng số."""
        size = 2 if split['kind'] == 'numeric' else len(split['values'])
        groups, missing = [[] for _ in range(size)], []
        known_weights = [0.0] * size
        for index, weight in samples:
            branch = self._branch(split, self._rows[index])
            if branch is None:
                missing.append((index, weight))
            else:
                groups[branch].append((index, weight))
                known_weights[branch] += weight
        known = sum(known_weights)
        proportions = [w / known for w in known_weights] if known > 0 else [1 / size] * size
        for index, weight in missing:
            for branch, fraction in enumerate(proportions):
                if fraction > 0:
                    groups[branch].append((index, weight * fraction))
        return groups, proportions

    def _grow(self, samples):
        """Dựng cây đầy đủ trước cắt tỉa; cột số có thể được dùng lại ở nút sâu."""
        counts = self._counts(samples)
        node = {'counts': counts, '_samples': samples}
        total = sum(counts)
        if total < 2 * self.config['min_samples_leaf'] - EPSILON or min(counts) < EPSILON:
            return node
        split = self._select_split(samples, total)
        if split is None:
            return node
        groups, proportions = self._partition(split, samples)
        node.update(split=split, proportions=proportions,
                    children=[self._grow(group) for group in groups])
        return node

    def _leaf_error(self, counts):
        """Lỗi quan sát + phần cộng cận trên, tính theo nhãn đa số của lá."""
        total = sum(counts)
        errors = min(counts)
        return errors + extra_errors(total, errors, self.config['confidence'])

    def _subtree_error(self, node, estimated):
        if 'children' not in node:
            return self._leaf_error(node['counts']) if estimated else min(node['counts'])
        return sum(self._subtree_error(child, estimated) for child in node['children'])

    @staticmethod
    def _make_leaf(node):
        """Bỏ phép chia, giữ phân bố và mẫu train của chính nút."""
        for key in ('split', 'proportions', 'children'):
            node.pop(key, None)

    def _collapse(self, node):
        """Thu gọn từ trên xuống nếu chia cây không giảm lỗi huấn luyện."""
        if 'children' not in node:
            return
        if self._subtree_error(node, False) >= min(node['counts']) - 0.001:
            self._make_leaf(node)
            self.pruning['collapsed'] += 1
        else:
            for child in node['children']:
                self._collapse(child)

    def _branch_error(self, node, samples):
        """Ước lượng cây con trên toàn bộ mẫu nút cha, không thay phép chia."""
        if 'children' not in node:
            return self._leaf_error(self._counts(samples))
        groups, _ = self._partition(node['split'], samples)
        return sum(self._branch_error(child, group) for child, group in zip(node['children'], groups))

    def _redistribute(self, node, samples):
        """Sau nâng cây con, học lại phân bố theo cấu trúc đã nâng, không chọn ngưỡng mới."""
        node['counts'], node['_samples'] = self._counts(samples), samples
        if 'children' in node:
            groups, node['proportions'] = self._partition(node['split'], samples)
            for child, group in zip(node['children'], groups):
                self._redistribute(child, group)

    def _prune(self, node):
        """Duyệt từ dưới lên; so sánh giữ cây, thay bằng lá và nâng nhánh lớn nhất."""
        if 'children' not in node:
            return
        for child in node['children']:
            self._prune(child)
        largest = max(range(len(node['children'])), key=lambda i: node['proportions'][i])
        largest_child = node['children'][largest]
        leaf_error = self._leaf_error(node['counts'])
        tree_error = self._subtree_error(node, True)
        raised_error = (self._branch_error(largest_child, node['_samples'])
                        if self.config['subtree_raising'] else float('inf'))
        if leaf_error <= min(tree_error, raised_error) + 0.1 + EPSILON:
            self._make_leaf(node)
            self.pruning['replaced_by_leaf'] += 1
        elif raised_error <= tree_error + 0.1 + EPSILON:
            samples = node['_samples']
            replacement = dict(largest_child)
            node.clear()
            node.update(replacement)
            self._redistribute(node, samples)
            self.pruning['raised'] += 1
            self._prune(node)

    def _discard_training_samples(self, node):
        node.pop('_samples', None)
        for child in node.get('children', []):
            self._discard_training_samples(child)

    def tree_size(self):
        """Số nút, số lá và độ sâu (lá gốc có độ sâu 0)."""
        def measure(node):
            if 'children' not in node:
                return {'nodes': 1, 'leaves': 1, 'depth': 0}
            children = [measure(child) for child in node['children']]
            return {'nodes': 1 + sum(c['nodes'] for c in children),
                    'leaves': sum(c['leaves'] for c in children),
                    'depth': 1 + max(c['depth'] for c in children)}
        return measure(self.root)

    def _probability(self, node, row):
        """Trộn xác suất các nhánh khi thiếu thuộc tính; nhánh rỗng dùng nút cha."""
        if 'children' not in node:
            total = sum(node['counts'])
            return node['counts'][1] / total if total else 0.5
        branch = self._branch(node['split'], row)
        if branch is not None:
            child = node['children'][branch]
            if sum(child['counts']) <= EPSILON:
                return node['counts'][1] / sum(node['counts'])
            return self._probability(child, row)
        return sum(fraction * self._probability(child, row)
                   for fraction, child in zip(node['proportions'], node['children']) if fraction > 0)

    def predict_with_scores(self, X):
        """Trả nhãn và xác suất lớp 1; hòa chọn 0; không fit lại dữ liệu đầu vào."""
        if not hasattr(self, 'root'):
            raise ValueError("Cần fit C4.5 trước.")
        scores = [min(1.0, max(0.0, self._probability(self.root, row)))
                  for row in self._matrix(X, self.feature_types)]
        return [int(score > 0.5) for score in scores], scores

    def predict(self, X):
        return self.predict_with_scores(X)[0]

    def predict_proba(self, X):
        return [[1 - score, score] for score in self.predict_with_scores(X)[1]]

    def to_state(self):
        """JSON thuần, không lưu dữ liệu huấn luyện hay đối tượng thư viện."""
        if not hasattr(self, 'root'):
            raise ValueError("Cần fit trước khi lưu C4.5.")
        return deepcopy({'schema': 'c45-v1', 'config': self.config,
                         'feature_types': self.feature_types, 'values': self.values,
                         'root': self.root, 'pruning': self.pruning})

    @classmethod
    def from_state(cls, state):
        """Nạp cây đã lưu; kiểm cấu trúc, số đếm, ngưỡng và trọng số nhánh."""
        if not isinstance(state, dict) or state.get('schema') != 'c45-v1':
            raise ValueError("Trạng thái không phải cây C4.5 đầy đủ schema c45-v1.")
        model = cls(**state['config'])
        model.feature_types = list(state['feature_types'])
        model.values = deepcopy(state['values'])
        if not model.feature_types or any(k not in ('numeric', 'nominal') for k in model.feature_types):
            raise ValueError("Sai kiểu thuộc tính C4.5.")

        def validate(node):
            counts = node.get('counts', [])
            if (len(counts) != 2 or any(isinstance(v, bool) or not isinstance(v, Real) or
                                      not isfinite(v) or v < 0 for v in counts)):
                raise ValueError("Số đếm trọng số C4.5 không hợp lệ.")
            if 'children' not in node:
                return
            split = node['split']
            column = split['column']
            if type(column) is not int or not 0 <= column < len(model.feature_types):
                raise ValueError("Cột chia C4.5 không hợp lệ.")
            if split['kind'] != model.feature_types[column]:
                raise ValueError("Phép chia sai kiểu thuộc tính.")
            if split['kind'] == 'numeric' and not isfinite(split['threshold']):
                raise ValueError("Ngưỡng C4.5 phải hữu hạn.")
            size = 2 if split['kind'] == 'numeric' else len(split['values'])
            weights = node['proportions']
            if (size < 2 or len(node['children']) != size or len(weights) != size or
                    any(not isfinite(v) or v < 0 for v in weights) or abs(sum(weights) - 1) > EPSILON):
                raise ValueError("Trọng số nhánh C4.5 không hợp lệ.")
            for child in node['children']:
                validate(child)
            for label in (0, 1):
                child_total = sum(child['counts'][label] for child in node['children'])
                if abs(child_total - counts[label]) > EPSILON * max(1, counts[label]):
                    raise ValueError("Cây không bảo toàn trọng số lớp.")

        validate(state['root'])
        if sum(state['root']['counts']) <= 0:
            raise ValueError("Cây C4.5 rỗng.")
        model.root, model.pruning = deepcopy(state['root']), deepcopy(state['pruning'])
        return model
