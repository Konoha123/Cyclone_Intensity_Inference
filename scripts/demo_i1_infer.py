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

2026-10-12（M1）起，核心推理逻辑提取至 src/cyclone_intensity/model.py，
本脚本为其薄封装；控制台输出格式、字段与退出码保持不变。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyclone_intensity.model import (  # noqa: E402
    InferenceError,
    Predictor,
    check_probability_legality,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="I1 demo: 加载权重并输出类别概率")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "weights" / "model.pt")
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "data" / "samples" / "example_image.jpg")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "demo_i1_result.json")
    args = parser.parse_args()

    # 1. 权重完整性（AGENTS 5.2：只加载已记录 SHA-256 的可信权重）+ 加载与类别映射核验
    try:
        predictor = Predictor(args.weights)
    except InferenceError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(f"[OK] 权重 SHA-256 与 docs/model_provenance.md 登记一致: {predictor.sha256}")
    print(f"[{'WARN' if predictor.issues else 'OK'}] 类别映射: {[{'index': i, 'name': predictor.names[i], 'T': t} for i, t in enumerate(predictor.t_values)]}")
    for msg in predictor.issues:
        print(f"[WARN] {msg}")

    # 2. 单图推理（两次，核验确定性）与输出数值合法性
    try:
        payload = predictor.predict(args.image)
    except InferenceError as exc:
        for name, ok in (predictor.last_checks or {}).items():
            print(f"[{'OK' if ok else 'FAIL'}] {name}")
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    if not payload["deterministic_repeat"]:
        print("[WARN] 两次推理结果不一致（浮点非确定性）", file=sys.stderr)

    checks = check_probability_legality(payload["probabilities"], len(predictor.t_values))
    for name, ok in checks.items():
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        print("[FAIL] 输出数值合法性检查未通过", file=sys.stderr)
        return 1

    # 3. T 值后处理结果写出（README 5.3 输出契约；概率未校准）
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    top3 = sorted(zip(payload["class_labels"]["t_values"], payload["probabilities"]), key=lambda x: -x[1])[:3]
    print(f"[OK] 单图推理成功: t_mode={payload['t_mode']}, t_mean={payload['t_mean']:.4f}（未校准）")
    print(f"[OK] Top-3: " + ", ".join(f"T{t:.1f}={p:.4f}" for t, p in top3))
    print(f"[OK] 结果已写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
