"""Ví dụ C4.5 có đáp án tính tay, bảo toàn trọng số và kiểm tra cắt tỉa."""

from copy import deepcopy
import json
import math
import pytest

from src.algorithms.c45 import C45, extra_errors, normal_quantile


def test_continuous_threshold_gain_and_mdl_are_hand_calculated():
    X, y = [[i] for i in range(1, 9)], [0] * 4 + [1] * 4
    model = C45(min_samples_leaf=1).fit(X, y)
    split = model.root['split']
    assert split['threshold'] == 4
    assert split['gain'] == pytest.approx(1 - math.log2(7) / 8)
    assert split['gain_ratio'] == pytest.approx(split['gain'])
    assert model.predict(X) == y
    assert model.tree_size() == {'nodes': 3, 'leaves': 2, 'depth': 1}


def test_missing_gain_fractional_counts_and_mixture_have_exact_answers():
    X, y = [[0], [0], [10], [10], [None]], [0, 0, 1, 1, 1]
    model = C45(min_samples_leaf=1, mdl_correction=False, pruned=False, collapse=False).fit(X, y)
    node = model.root
    assert node['split']['gain'] == pytest.approx(4 / 5)
    split_info = -2 * (2 / 5) * math.log2(2 / 5) - (1 / 5) * math.log2(1 / 5)
    assert node['split']['gain_ratio'] == pytest.approx((4 / 5) / split_info)
    assert node['proportions'] == [0.5, 0.5]
    assert node['children'][0]['counts'] == [2, 0.5]
    assert node['children'][1]['counts'] == [0, 2.5]
    assert model.predict_proba([[None]])[0] == pytest.approx([0.4, 0.6])


def test_sample_weights_are_preserved_through_unknown_value():
    model = C45(min_samples_leaf=1, mdl_correction=False, pruned=False, collapse=False).fit(
        [[0], [0], [10], [10], [None]], [0, 0, 1, 1, 1], sample_weight=[1, 1, 2, 2, 2])
    assert model.root['counts'] == [2, 6]
    assert model.root['proportions'] == pytest.approx([1 / 3, 2 / 3])
    assert model.root['children'][0]['counts'] == pytest.approx([2, 2 / 3])
    assert model.root['children'][1]['counts'] == pytest.approx([0, 16 / 3])
    assert model.predict_proba([[None]])[0] == pytest.approx([0.25, 0.75])


def test_nominal_strings_unknown_categories_and_missing_values():
    model = C45(min_samples_leaf=1, pruned=False, collapse=False).fit(
        [['red'], ['red'], ['blue'], ['blue'], [None]], [0, 0, 1, 1, 1], ['nominal'])
    assert model.root['split']['kind'] == 'nominal'
    for probabilities in model.predict_proba([['green'], [None]]):
        assert probabilities == pytest.approx([0.4, 0.6])


def test_numeric_attribute_can_be_reused_deeper_in_the_tree():
    model = C45(min_samples_leaf=1, mdl_correction=False, pruned=False, collapse=False).fit(
        [[i] for i in range(12)], [0] * 4 + [1] * 4 + [0] * 4)
    assert model.tree_size()['depth'] == 2
    assert model.root['split']['column'] == 0
    assert model.root['children'][1]['split']['column'] == 0
    assert model.predict([[1], [5], [10]]) == [0, 1, 0]


def test_pruning_quantiles_and_zero_error_bound_match_known_values():
    assert normal_quantile(0.5) == 0
    assert normal_quantile(0.75) == pytest.approx(0.674489750196, abs=1e-8)
    assert normal_quantile(0.975) == pytest.approx(1.95996398454, abs=1e-8)
    assert normal_quantile(0.001) == pytest.approx(-3.09023230617, abs=1e-8)
    assert extra_errors(10, 0) == pytest.approx(10 * (1 - 0.25 ** 0.1))
    assert extra_errors(10, 10) == 0
    assert extra_errors(0, 0) == 0
    assert extra_errors(10, 0.5) == pytest.approx((extra_errors(10, 0) + extra_errors(10, 1)) / 2)


def test_all_missing_constant_and_pure_training_sets_stop_at_leaf():
    for X in ([[None], [None], [None]], [[1], [1], [1]]):
        model = C45().fit(X, [0, 1, 1])
        assert model.tree_size()['nodes'] == 1
        assert model.predict_proba([[None]])[0] == pytest.approx([1 / 3, 2 / 3])
    assert C45().fit([[1], [2]], [0, 0]).predict([[5]]) == [0]


def test_serialization_is_json_only_preserves_mixture_and_cannot_change_fit():
    X = [[i, None if i % 3 else 2] for i in range(20)]
    model = C45().fit(X, [0] * 10 + [1] * 10)
    original = deepcopy(model.to_state())
    restored = C45.from_state(json.loads(json.dumps(original, allow_nan=False)))
    assert restored.predict_with_scores(X + [[None, None]]) == model.predict_with_scores(X + [[None, None]])
    restored.predict([[999, None]])
    assert restored.to_state() == original
    assert '"_samples":' not in json.dumps(original)
    broken = deepcopy(original)
    broken['root']['counts'][0] = -1
    with pytest.raises(ValueError):
        C45.from_state(broken)


def test_collapse_leaf_pruning_and_subtree_raising_preserve_counts():
    """Dữ liệu tổng hợp seed 91 để đi qua cả ba nhánh cắt tỉa, không phải dữ liệu thực nghiệm."""
    import random
    generator = random.Random(91)
    X = [[generator.randrange(4), generator.randrange(4), generator.random()] for _ in range(30)]
    y = [generator.randrange(2) for _ in X]
    model = C45(mdl_correction=False).fit(X, y)
    assert model.pruning['collapsed'] > 0
    assert model.pruning['replaced_by_leaf'] > 0
    assert model.pruning['raised'] > 0
    assert model.pruning['after']['nodes'] < model.pruning['before']['nodes']
    assert model.root['counts'] == [len(y) - sum(y), sum(y)]
    # Bộ kiểm trạng thái kiểm cả tổng hai lớp của mọi nút sau khi nâng cây.
    assert C45.from_state(model.to_state()).predict(X) == model.predict(X)


@pytest.mark.parametrize('config', [dict(confidence=0), dict(confidence=0.6), dict(min_samples_leaf=0),
                                   dict(pruned=1), dict(confidence=float('nan'))])
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        C45(**config)


@pytest.mark.parametrize('X,y,weights', [([[1]], [2], None), ([[float('inf')]], [0], None),
                                       ([[True]], [0], None), ([[1]], [0], [-1]),
                                       ([[1]], [0], [0]), ([[1]], [], None)])
def test_invalid_training_data_are_rejected(X, y, weights):
    with pytest.raises(ValueError):
        C45().fit(X, y, sample_weight=weights)
