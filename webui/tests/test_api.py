"""webui 后端 API 测试。

- 主线用假 Predictor（StubPredictor）走完整 HTTP 流程，不依赖权重；
- 真权重冒烟仅在 weights/model.pt 与示例图像同时存在时执行，
  结果断言与 outputs/demo_i1_result.json 基线（I1）一致。
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import MAX_UPLOAD_BYTES, create_app
from cyclone_intensity.model import compute_t_mean, compute_t_mode

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_WEIGHTS = REPO_ROOT / "weights" / "model.pt"
EXAMPLE_IMAGE = REPO_ROOT / "data" / "samples" / "example_image.jpg"

T13 = [1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
STUB_PROBS = [0.0] * 10 + [0.2, 0.7, 0.1]  # t_mode=7.5, t_mean=7.45


class StubPredictor:
    """与核心库 Predictor.predict 输出契约一致的替身（T 值后处理复用核心函数）。"""

    def __init__(self) -> None:
        self.t_values = T13
        self.probs = STUB_PROBS

    def predict(self, image: Path) -> dict:
        return {
            "model_id": "stub/model.pt",
            "model_sha256": "0" * 64,
            "input_id": Path(image).name,
            "image_time_utc": None,
            "class_labels": {
                "names": [str(int(t * 10)) for t in self.t_values],
                "t_values": self.t_values,
            },
            "probabilities": self.probs,
            "t_mode": compute_t_mode(self.probs, self.t_values),
            "t_mean": compute_t_mean(self.probs, self.t_values),
            "unit": "Dvorak T-number",
            "calibrated": False,
            "preprocessing_id": "ultralytics-default-classify-unverified",
            "class_mapping_issues": ["stub: 相对上游自述缺 T2.0"],
            "deterministic_repeat": True,
            "run": {
                "utc_time": "2026-10-12T00:00:00+00:00",
                "python": "3.13.16",
                "torch": "stub",
                "ultralytics": "stub",
                "device": "cpu",
            },
        }


def png_bytes(size: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client():
    with TestClient(create_app(predictor=StubPredictor())) as c:
        yield c


class TestPredictEndpoint:
    def test_valid_image_returns_contract_fields(self, client):
        resp = client.post("/api/predict", files={"image": ("sample.png", png_bytes(), "image/png")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_id"] == "stub/model.pt"
        assert body["class_labels"]["t_values"] == T13
        assert len(body["probabilities"]) == 13
        assert body["t_mode"] == 7.5
        assert body["t_mean"] == pytest.approx(7.45)
        assert body["calibrated"] is False
        assert body["unit"] == "Dvorak T-number"
        assert body["preprocessing_id"] == "ultralytics-default-classify-unverified"
        assert body["class_mapping_issues"] == ["stub: 相对上游自述缺 T2.0"]

    def test_empty_upload_rejected(self, client):
        resp = client.post("/api/predict", files={"image": ("empty.png", b"", "image/png")})
        assert resp.status_code == 400

    def test_non_image_rejected(self, client):
        resp = client.post("/api/predict", files={"image": ("note.txt", b"not an image", "text/plain")})
        assert resp.status_code == 400

    def test_truncated_image_rejected(self, client):
        resp = client.post(
            "/api/predict",
            files={"image": ("trunc.png", png_bytes()[: len(png_bytes()) // 2], "image/png")},
        )
        assert resp.status_code == 400

    def test_unsupported_format_rejected(self, client):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="GIF")
        resp = client.post("/api/predict", files={"image": ("x.gif", buf.getvalue(), "image/gif")})
        assert resp.status_code == 400

    def test_oversized_upload_rejected(self, client):
        big = b"\x89PNG" + b"\x00" * MAX_UPLOAD_BYTES
        resp = client.post("/api/predict", files={"image": ("big.png", big, "image/png")})
        assert resp.status_code == 413


def test_frontend_root_returns_404_until_m3(client):
    resp = client.get("/")
    assert resp.status_code == 404  # M2 预期：frontend/ 未落盘，M3 后此断言随前端验收更新


BASELINE = REPO_ROOT / "outputs" / "demo_i1_result.json"


@pytest.mark.skipif(not REAL_WEIGHTS.is_file(), reason="weights/model.pt 不存在，跳过真权重冒烟")
@pytest.mark.skipif(not EXAMPLE_IMAGE.is_file(), reason="data/samples/example_image.jpg 不存在")
@pytest.mark.skipif(not BASELINE.is_file(), reason="outputs/demo_i1_result.json 基线不存在")
class TestRealWeightsSmoke:
    def test_real_weights_example_image_matches_i1_baseline(self):
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        with TestClient(create_app()) as c:
            with open(EXAMPLE_IMAGE, "rb") as f:
                resp = c.post("/api/predict", files={"image": ("example_image.jpg", f, "image/jpeg")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["probabilities"] == baseline["probabilities"]  # 与 I1 基线逐位一致
        assert body["t_mode"] == baseline["t_mode"]
        assert body["t_mean"] == pytest.approx(baseline["t_mean"])
        assert body["model_sha256"] == baseline["model_sha256"]
