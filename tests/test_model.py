"""src/cyclone_intensity 核心逻辑的无模型单元测试（AGENTS 7.2）。

不加载真实权重：Predictor 仅测试可预期失败的守卫分支；
类别映射、概率合法性与 T 值后处理用合成数据覆盖（含均匀分布、
one-hot、边界类别与异常输入）。
"""

from __future__ import annotations

import hashlib

import pytest

from cyclone_intensity.model import (
    EXPECTED_CLASS_NAMES,
    InferenceError,
    Predictor,
    check_probability_legality,
    compute_t_mean,
    compute_t_mode,
    sha256_of,
    verify_class_names,
)

# 实际权重的类别表（docs/model_provenance.md 第 4 节：13 类，缺 T2.0）
REAL_T_VALUES_13 = [1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
REAL_NAMES_13 = {i: str(int(t * 10)) for i, t in enumerate(REAL_T_VALUES_13)}


class TestVerifyClassNames:
    def test_real_13class_table_maps_and_reports_missing_t20(self):
        t_values, issues = verify_class_names(REAL_NAMES_13)
        assert t_values == REAL_T_VALUES_13
        assert len(issues) == 2  # 缺类明细 + 总数说明
        assert any("20" in msg for msg in issues)

    def test_expected_14class_table_has_no_issues(self):
        names = {i: n for i, n in enumerate(EXPECTED_CLASS_NAMES)}
        t_values, issues = verify_class_names(names)
        assert not issues
        assert t_values == [x / 10 for x in range(15, 85, 5)]

    def test_dict_input_ordered_by_index(self):
        names = {2: "30", 0: "15", 1: "25"}
        t_values, _ = verify_class_names(names)
        assert t_values == [1.5, 2.5, 3.0]

    @pytest.mark.parametrize("bad_name", ["t15", "1.5", "15.0", None])
    def test_non_integer_name_raises(self, bad_name):
        with pytest.raises(ValueError, match="不是整数编码"):
            verify_class_names({0: bad_name})

    @pytest.mark.parametrize("bad_name", ["10", "85", "22", "0", "-15"])
    def test_out_of_range_or_bad_step_raises(self, bad_name):
        with pytest.raises(ValueError, match="不在期望范围"):
            verify_class_names({0: bad_name})


class TestCheckProbabilityLegality:
    def test_uniform_distribution_passes_all(self):
        checks = check_probability_legality([1 / 13] * 13, 13)
        assert all(checks.values())

    def test_one_hot_passes_all(self):
        checks = check_probability_legality([0.0] * 12 + [1.0], 13)
        assert all(checks.values())

    def test_count_mismatch_detected(self):
        checks = check_probability_legality([0.5, 0.5], 3)
        assert checks["prob_count_matches_classes"] is False

    def test_negative_detected(self):
        checks = check_probability_legality([1.2, -0.2], 2)
        assert checks["probabilities_nonnegative"] is False

    @pytest.mark.parametrize("bad", [float("inf"), float("nan")])
    def test_non_finite_detected(self, bad):
        checks = check_probability_legality([bad, 0.5], 2)
        assert checks["probabilities_finite"] is False

    def test_sum_not_near_one_detected(self):
        checks = check_probability_legality([0.5, 0.4], 2)
        assert checks["probabilities_sum_near_1"] is False


class TestTPostprocessing:
    def test_one_hot_boundary_classes(self):
        first = [1.0] + [0.0] * 12
        last = [0.0] * 12 + [1.0]
        assert compute_t_mode(first, REAL_T_VALUES_13) == 1.5
        assert compute_t_mean(first, REAL_T_VALUES_13) == pytest.approx(1.5)
        assert compute_t_mode(last, REAL_T_VALUES_13) == 8.0
        assert compute_t_mean(last, REAL_T_VALUES_13) == pytest.approx(8.0)

    def test_uniform_distribution_mean_and_tie_breaking(self):
        probs = [1 / 13] * 13
        assert compute_t_mean(probs, REAL_T_VALUES_13) == pytest.approx(sum(REAL_T_VALUES_13) / 13)
        assert compute_t_mode(probs, REAL_T_VALUES_13) == 1.5  # 并列取索引最小

    def test_known_weighted_mean_and_mode(self):
        probs = [0.0, 0.7, 0.0, 0.3] + [0.0] * 9
        assert compute_t_mean(probs, REAL_T_VALUES_13) == pytest.approx(0.7 * 2.5 + 0.3 * 3.5)
        assert compute_t_mode(probs, REAL_T_VALUES_13) == 2.5


class TestPredictorGuards:
    def test_missing_weights_raises_inference_error(self, tmp_path):
        with pytest.raises(InferenceError, match="权重不存在"):
            Predictor(tmp_path / "nope.pt")

    def test_wrong_sha256_raises_before_model_load(self, tmp_path):
        fake = tmp_path / "fake.pt"
        fake.write_bytes(b"not-a-real-checkpoint")
        with pytest.raises(InferenceError, match="SHA-256 不符"):
            Predictor(fake)


def test_sha256_of_known_content(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello cyclone\n")
    assert sha256_of(p) == hashlib.sha256(b"hello cyclone\n").hexdigest()
