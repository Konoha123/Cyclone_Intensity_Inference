#!/usr/bin/env python
"""I1 阶段 demo：加载上游权重并输出类别概率。

验收目标（README 阶段计划 I1）：
1. 类别映射核验（model.names 与 T 值对应关系）；
2. 单图推理成功；
3. 输出数值合法（概率非负、有限、和近似 1）。

用法（在 cyclone-intensity-inference-env 环境中，仓库根目录下执行）：
    python scripts/demo_i1_infer.py [--weights weights/model.pt]
                                    [--image data/samples/example_image.jpg]
                                    [--output outputs/demo_i1_result.json]

说明：本脚本使用 Ultralytics 8.4.175 分类默认预处理管线，其具体变换
（CenterCrop/缩放/归一化）尚未核验，属 I2 阶段工作；输出中
preprocessing_id 如实标注为未核验的默认管线。概率未校准。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]

# docs/model_provenance.md 第 2.1 节登记的可信权重 SHA-256
EXPECTED_SHA256 = "55fd4f72af0e3825dccdd4dcccb82a8d8b46739a30b3681304c35528f8e078a1"

# 上游自述的期望类别：T1.5–T8.0、步长 0.5、共 14 类（类别名 = T×10）
EXPECTED_CLASS_NAMES = [str(int(t * 10)) for t in [x / 2 for x in range(3, 17)]]

PREPROCESSING_ID = "ultralytics-default-classify-unverified"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_class_names(names: dict[int, str]) -> tuple[list[float], list[str]]:
    """核验类别表并返回按索引排序的 T 值列表与差异报告。"""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="I1 demo: 加载权重并输出类别概率")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "weights" / "model.pt")
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "data" / "samples" / "example_image.jpg")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "demo_i1_result.json")
    args = parser.parse_args()

    # 1. 权重完整性（AGENTS 5.2：只加载已记录 SHA-256 的可信权重）
    if not args.weights.is_file():
        print(f"[FAIL] 权重不存在: {args.weights}", file=sys.stderr)
        return 1
    digest = sha256_of(args.weights)
    if digest != EXPECTED_SHA256:
        print(f"[FAIL] 权重 SHA-256 不符: 实测 {digest}，登记 {EXPECTED_SHA256}", file=sys.stderr)
        return 1
    print(f"[OK] 权重 SHA-256 与 docs/model_provenance.md 登记一致: {digest}")

    # 2. 加载与类别映射核验
    model = YOLO(str(args.weights))
    names = model.names
    if getattr(model.model, "args", {}).get("task") != "classify":
        print(f"[FAIL] 权重 task 不是 classify: {model.model.args.get('task')}", file=sys.stderr)
        return 1
    t_values, issues = verify_class_names(names)
    print(f"[{'WARN' if issues else 'OK'}] 类别映射: {[{'index': i, 'name': names[i], 'T': t} for i, t in enumerate(t_values)]}")
    for msg in issues:
        print(f"[WARN] {msg}")

    # 3. 单图推理（两次，核验确定性）
    if not args.image.is_file():
        print(f"[FAIL] 输入图像不存在: {args.image}", file=sys.stderr)
        return 1
    probs_runs = []
    for _ in range(2):
        result = model(str(args.image), verbose=False)[0]
        probs_runs.append(result.probs.data.tolist())
    probs, probs_repeat = probs_runs
    if probs != probs_repeat:
        print("[WARN] 两次推理结果不一致（浮点非确定性）", file=sys.stderr)

    # 4. 输出数值合法性
    n = len(probs)
    checks = {
        "prob_count_matches_classes": n == len(t_values),
        "probabilities_nonnegative": all(p >= 0.0 for p in probs),
        "probabilities_finite": all(p == p and abs(p) != float("inf") for p in probs),
        "probabilities_sum_near_1": abs(sum(probs) - 1.0) < 1e-4,
    }
    for name, ok in checks.items():
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        print("[FAIL] 输出数值合法性检查未通过", file=sys.stderr)
        return 1

    # 5. T 值后处理（README 5.3 输出契约；概率未校准）
    t_mode = t_values[probs.index(max(probs))]
    t_mean = sum(p * t for p, t in zip(probs, t_values))

    payload = {
        "model_id": "upstream-EnceladusCat-Cyclone-Intensity-ML@55328c4/model.pt",
        "model_sha256": digest,
        "input_id": args.image.name,
        "image_time_utc": None,
        "class_labels": {"names": [names[i] for i in sorted(names)], "t_values": t_values},
        "probabilities": probs,
        "t_mode": t_mode,
        "t_mean": t_mean,
        "unit": "Dvorak T-number",
        "calibrated": False,
        "preprocessing_id": PREPROCESSING_ID,
        "class_mapping_issues": issues,
        "deterministic_repeat": probs == probs_repeat,
        "run": {
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "device": "cpu" if not torch.cuda.is_available() else "cuda",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    top3 = sorted(zip(t_values, probs), key=lambda x: -x[1])[:3]
    print(f"[OK] 单图推理成功: t_mode={t_mode}, t_mean={t_mean:.4f}（未校准）")
    print(f"[OK] Top-3: " + ", ".join(f"T{t:.1f}={p:.4f}" for t, p in top3))
    print(f"[OK] 结果已写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
