"""webui 接口层数据结构（Pydantic v2）。

字段与 I1 输出契约（README 5.3 / scripts/demo_i1_infer.py payload）一一对应，
作为 API 的唯一响应规范；构造时做最后一道契约校验
（概率数量与类别数一致、非负且总和近似 1），不满足即拒绝出站。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ClassLabels(BaseModel):
    """权重内按索引排序的类别表。"""

    names: list[str] = Field(..., description="类别名（T×10 整数编码）")
    t_values: list[float] = Field(..., description="与 names 一一对应的 Dvorak T 值")


class RunInfo(BaseModel):
    """单次推理的运行环境登记。"""

    utc_time: str = Field(..., description="推理 UTC 时间（ISO-8601）")
    python: str
    torch: str
    ultralytics: str
    device: str


class PredictResponse(BaseModel):
    """POST /api/predict 响应（概率未校准）。"""

    model_id: str
    model_sha256: str
    input_id: str
    image_time_utc: str | None = Field(default=None, description="图像时刻（当前来源未提供）")
    class_labels: ClassLabels
    probabilities: list[float] = Field(..., description="按类别索引排序的 Softmax 概率")
    t_mode: float = Field(..., description="最大概率类别对应的 T 值")
    t_mean: float = Field(..., description="概率加权平均 T 值 E[T]=Σp_i·T_i（未校准）")
    unit: str = Field(default="Dvorak T-number")
    calibrated: bool = Field(default=False, description="概率未经过独立校准评估")
    preprocessing_id: str
    class_mapping_issues: list[str] = Field(default_factory=list)
    run: RunInfo

    @model_validator(mode="after")
    def _check_contract(self) -> "PredictResponse":
        if len(self.probabilities) != len(self.class_labels.t_values):
            raise ValueError("probabilities 数量与 class_labels.t_values 不一致")
        if any(p < 0.0 for p in self.probabilities):
            raise ValueError("probabilities 存在负值")
        if abs(sum(self.probabilities) - 1.0) > 1e-4:
            raise ValueError("probabilities 总和偏离 1")
        return self

    @classmethod
    def from_payload(cls, payload: dict) -> "PredictResponse":
        """由核心库 Predictor.predict 的 payload dict 构造（字段名一致）。"""
        return cls(
            model_id=payload["model_id"],
            model_sha256=payload["model_sha256"],
            input_id=payload["input_id"],
            image_time_utc=payload.get("image_time_utc"),
            class_labels=ClassLabels(**payload["class_labels"]),
            probabilities=payload["probabilities"],
            t_mode=payload["t_mode"],
            t_mean=payload["t_mean"],
            unit=payload.get("unit", "Dvorak T-number"),
            calibrated=payload.get("calibrated", False),
            preprocessing_id=payload["preprocessing_id"],
            class_mapping_issues=payload.get("class_mapping_issues", []),
            run=RunInfo(**payload["run"]),
        )


class ErrorResponse(BaseModel):
    """错误响应统一结构（FastAPI HTTPException 的 detail 包装）。"""

    detail: str
