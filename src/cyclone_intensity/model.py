"""推理核心：可信权重加载核验、类别表映射、单图推理与 T 值后处理。

从 scripts/demo_i1_infer.py（I1 阶段验收产物）提取，是仓库内唯一的
推理事实来源；输出契约见 README 第 5.3 节与 AGENTS 第 5 节：
- 只加载已登记 SHA-256 的可信权重（AGENTS 5.2）；
- 类别顺序与 T 值映射以权重内实际类别表为准，核验后方可使用；
- 概率为 Softmax 原始输出，未校准；预处理管线 ID 如实标注未核验。
"""

from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

# docs/model_provenance.md 第 2.1 节登记的可信权重 SHA-256
EXPECTED_SHA256 = "55fd4f72af0e3825dccdd4dcccb82a8d8b46739a30b3681304c35528f8e078a1"

# 上游自述的期望类别：T1.5–T8.0、步长 0.5、共 14 类（类别名 = T×10）
EXPECTED_CLASS_NAMES = [str(int(t * 10)) for t in [x / 2 for x in range(3, 17)]]

PREPROCESSING_ID = "ultralytics-default-classify-unverified"

MODEL_ID = "upstream-EnceladusCat-Cyclone-Intensity-ML@55328c4/model.pt"


class InferenceError(RuntimeError):
    """推理流程中的可预期失败（权重/图像/输出合法性），消息可直接面向用户。"""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_class_names(names: dict[int, str]) -> tuple[list[float], list[str]]:
    """核验类别表并返回按索引排序的 T 值列表与差异报告。

    类别名必须能映射为 T 值（T×10 整数编码，范围 [15, 80]、步长 5），
    否则抛出 ValueError；与上游自述 14 类的差异仅记录为 issues，不阻断。
    """
    issues: list[str] = []
    actual = [names[i] for i in sorted(names)]

    t_values: list[float] = []
    for name in actual:
        try:
            code = int(name)
        except (TypeError, ValueError):
            raise ValueError(f"类别名 {name!r} 不是整数编码（期望 T×10），无法映射为 T 值")
        if code < 15 or code > 80 or code % 5 != 0:
            raise ValueError(f"类别名 {name!r} 不在期望范围 [15, 80]（T1.5–T8.0，步长 0.5）")
        t_values.append(code / 10)

    if actual != EXPECTED_CLASS_NAMES:
        missing = sorted(set(EXPECTED_CLASS_NAMES) - set(actual))
        extra = sorted(set(actual) - set(EXPECTED_CLASS_NAMES))
        if missing:
            issues.append(f"相对上游自述 14 类缺少 {len(missing)} 类: {missing}（对应 T 值 {[int(m)/10 for m in missing]}）")
        if extra:
            issues.append(f"多出 {len(extra)} 类: {extra}")
        issues.append(f"实际 {len(actual)} 类，上游自述 14 类；以权重内实际类别表为准")
    return t_values, issues


def check_probability_legality(probs: list[float], n_classes: int) -> dict[str, bool]:
    """概率输出合法性检查（非负、有限、和近似 1、数量与类别数一致）。"""
    return {
        "prob_count_matches_classes": len(probs) == n_classes,
        "probabilities_nonnegative": all(p >= 0.0 for p in probs),
        "probabilities_finite": all(p == p and abs(p) != float("inf") for p in probs),
        "probabilities_sum_near_1": abs(sum(probs) - 1.0) < 1e-4,
    }


def compute_t_mode(probs: list[float], t_values: list[float]) -> float:
    """最大概率类别对应的 T 值（并列时取索引最小者）。"""
    return t_values[probs.index(max(probs))]


def compute_t_mean(probs: list[float], t_values: list[float]) -> float:
    """概率加权平均 T 值 E[T] = Σ p_i × T_i（概率未校准）。"""
    return sum(p * t for p, t in zip(probs, t_values))


class Predictor:
    """加载已登记的可信权重并提供单图推理（I1 输出契约）。

    构造时完成权重存在性、SHA-256、task=classify 与类别表核验；
    可预期失败抛 InferenceError（消息为面向用户的具体原因），
    类别表无法映射为 T 值时抛 verify_class_names 的 ValueError。
    """

    def __init__(self, weights: Path):
        weights = Path(weights)
        if not weights.is_file():
            raise InferenceError(f"权重不存在: {weights}")
        self.weights = weights
        self.sha256 = sha256_of(weights)
        if self.sha256 != EXPECTED_SHA256:
            raise InferenceError(f"权重 SHA-256 不符: 实测 {self.sha256}，登记 {EXPECTED_SHA256}")

        self.model = YOLO(str(weights))
        self.names = self.model.names
        task = getattr(self.model.model, "args", {}).get("task")
        if task != "classify":
            raise InferenceError(f"权重 task 不是 classify: {task}")

        self.t_values, self.issues = verify_class_names(self.names)
        self.last_checks: dict[str, bool] | None = None

    def predict(self, image: Path) -> dict:
        """单图推理两次（确定性核验），通过合法性检查后返回 I1 契约 payload。"""
        image = Path(image)
        if not image.is_file():
            raise InferenceError(f"输入图像不存在: {image}")

        probs_runs = []
        for _ in range(2):
            result = self.model(str(image), verbose=False)[0]
            probs_runs.append(result.probs.data.tolist())
        probs, probs_repeat = probs_runs

        checks = check_probability_legality(probs, len(self.t_values))
        self.last_checks = checks
        if not all(checks.values()):
            raise InferenceError("输出数值合法性检查未通过")

        return {
            "model_id": MODEL_ID,
            "model_sha256": self.sha256,
            "input_id": image.name,
            "image_time_utc": None,
            "class_labels": {
                "names": [self.names[i] for i in sorted(self.names)],
                "t_values": self.t_values,
            },
            "probabilities": probs,
            "t_mode": compute_t_mode(probs, self.t_values),
            "t_mean": compute_t_mean(probs, self.t_values),
            "unit": "Dvorak T-number",
            "calibrated": False,
            "preprocessing_id": PREPROCESSING_ID,
            "class_mapping_issues": self.issues,
            "deterministic_repeat": probs == probs_repeat,
            "run": {
                "utc_time": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "device": "cpu" if not torch.cuda.is_available() else "cuda",
            },
        }
