"""Thống kê EDA tự cài, không đọc dữ liệu hoặc quyết định tập phân tích.

Caller phải truyền đúng tập phát triển nếu thống kê dùng để chọn phương pháp.
None và NaN số là missing; chuỗi, bool và vô hạn không phải missing hợp lệ.
"""

from math import isfinite, sqrt
from numbers import Real


def _observed_number(value):
    """Chuẩn hóa một ô số thành float/None; giá trị sai gây ValueError."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Ô EDA phải là số, None hoặc NaN.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("Ô EDA vượt giới hạn số thực hữu hạn.") from error
    if number != number:
        return None
    if not isfinite(number):
        raise ValueError("Ô EDA không được là số vô hạn.")
    return number


def _linear_quantile(ordered, fraction):
    """Nội suy tuyến tính tại (n-1)*q; dãy đã sắp rỗng trả None."""
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return (1 - weight) * ordered[lower_index] + weight * ordered[upper_index]


def summarize_features(X, feature_names):
    """Nhận ma trận (n,p), p tên duy nhất; trả dict thống kê theo tên cột.

    count là số ô quan sát, missing là số ô None/NaN; cột rỗng có thống kê
    None và outliers=[]. quartiles={q1,q2,q3}; boxplot chứa lower_fence,
    upper_fence, whisker_low, whisker_high, outliers. Râu là quan sát xa
    nhất trong hàng rào Q1-1.5*IQR, Q3+1.5*IQR, không dùng hàng rào làm râu.
    Cho phép n=0, không cho p=0; sai shape/tên/ô hoặc tràn số: ValueError.
    """
    names = list(feature_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Tên đặc trưng phải là chuỗi không rỗng.")
    if len(set(names)) != len(names):
        raise ValueError("Tên đặc trưng không được trùng.")
    columns = [[] for _ in names]
    missing_counts = [0 for _ in names]
    for row in X:
        if len(row) != len(names):
            raise ValueError("Số cột không khớp tên đặc trưng.")
        for index, value in enumerate(row):
            number = _observed_number(value)
            if number is None:
                missing_counts[index] += 1
            else:
                columns[index].append(number)
    result = {}
    for index, name in enumerate(names):
        ordered = sorted(columns[index])
        count = len(ordered)
        mean = sum(value / count for value in ordered) if count else None
        q1 = _linear_quantile(ordered, 0.25)
        median = _linear_quantile(ordered, 0.5)
        q3 = _linear_quantile(ordered, 0.75)
        boxplot = {"lower_fence": None, "upper_fence": None,
                   "whisker_low": None, "whisker_high": None, "outliers": []}
        if count:
            interquartile_range = q3 - q1
            lower_fence = q1 - 1.5 * interquartile_range
            upper_fence = q3 + 1.5 * interquartile_range
            if not all(isfinite(value) for value in (mean, q1, median, q3,
                                                      lower_fence, upper_fence)):
                raise ValueError("Thống kê EDA vượt giới hạn số học hữu hạn.")
            inside = [value for value in ordered if lower_fence <= value <= upper_fence]
            boxplot = {"lower_fence": lower_fence, "upper_fence": upper_fence,
                       "whisker_low": inside[0], "whisker_high": inside[-1],
                       "outliers": [value for value in ordered
                                    if value < lower_fence or value > upper_fence]}
        result[name] = {"count": count, "missing": missing_counts[index],
                        "min": ordered[0] if count else None,
                        "max": ordered[-1] if count else None, "mean": mean,
                        "median": median, "quartiles": {"q1": q1, "q2": median, "q3": q3},
                        "boxplot": boxplot}
    return result


def histogram_counts(values, bins=10):
    """Trả {edges, counts}, bins nguyên dương hoặc dãy biên tăng hữu hạn.

    Bỏ None/NaN; khoảng [trái, phải), bin cuối đóng cả hai đầu. Biên tự tạo
    chia đều min/max; dãy hằng mở rộng +/-0.5; rỗng dùng [0,1]. Biên truyền
    vào cho phép tái dùng trên tập khác; giá trị ngoài biên không được đếm.
    Sai bins/ô hoặc biên không thể biểu diễn phân biệt gây ValueError.
    """
    observed = []
    for value in values:
        number = _observed_number(value)
        if number is not None:
            observed.append(number)
    if isinstance(bins, bool):
        raise ValueError("bins phải là số nguyên dương hoặc dãy biên.")
    if isinstance(bins, int):
        if bins <= 0:
            raise ValueError("Số bin phải dương.")
        lower = min(observed) if observed else 0.0
        upper = max(observed) if observed else 1.0
        if lower == upper:
            lower -= 0.5
            upper += 0.5
        edges = [(1 - index / bins) * lower + (index / bins) * upper
                 for index in range(bins + 1)]
    else:
        try:
            edges = [_observed_number(value) for value in bins]
        except TypeError as error:
            raise ValueError("bins phải là số nguyên dương hoặc dãy biên.") from error
    if len(edges) < 2 or any(value is None or not isfinite(value) for value in edges):
        raise ValueError("Cần ít nhất hai biên hữu hạn.")
    if any(edges[index] >= edges[index + 1] for index in range(len(edges) - 1)):
        raise ValueError("Biên bin phải tăng nghiêm ngặt.")
    counts = [0] * (len(edges) - 1)
    for value in observed:
        for index in range(len(counts)):
            if edges[index] <= value < edges[index + 1] or (
                index == len(counts) - 1 and value == edges[-1]
            ):
                counts[index] += 1
                break
    return {"edges": edges, "counts": counts}


def pearson_correlation(x, y):
    """Trả (r hoặc None, số cặp) của hai dãy cùng độ dài; bỏ cặp thiếu.

    r=sum(dx*dy)/sqrt(sum(dx²)*sum(dy²)); dưới hai cặp hoặc cột hằng: None.
    Chỉ dùng trung bình các cặp hợp lệ. Sai độ dài/ô, tràn số: ValueError.
    """
    left = list(x)
    right = list(y)
    if len(left) != len(right):
        raise ValueError("Hai dãy Pearson phải cùng độ dài.")
    pairs = []
    for left_value, right_value in zip(left, right):
        left_number = _observed_number(left_value)
        right_number = _observed_number(right_value)
        if left_number is not None and right_number is not None:
            pairs.append((left_number, right_number))
    count = len(pairs)
    if count < 2:
        return None, count
    # Kiểm tra trực tiếp để sai số cộng mean không tạo phương sai giả.
    if any(all(pair[column] == pairs[0][column] for pair in pairs)
           for column in (0, 1)):
        return None, count
    mean_left = sum(pair[0] / count for pair in pairs)
    mean_right = sum(pair[1] / count for pair in pairs)
    cross_sum = left_square_sum = right_square_sum = 0.0
    for left_number, right_number in pairs:
        left_deviation = left_number - mean_left
        right_deviation = right_number - mean_right
        cross_sum += left_deviation * right_deviation
        left_square_sum += left_deviation * left_deviation
        right_square_sum += right_deviation * right_deviation
    if not all(isfinite(value) for value in (cross_sum, left_square_sum, right_square_sum)):
        raise ValueError("Tương quan vượt giới hạn số học hữu hạn.")
    if not left_square_sum or not right_square_sum:
        return None, count
    correlation = cross_sum / sqrt(left_square_sum) / sqrt(right_square_sum)
    return max(-1.0, min(1.0, correlation)), count
