"""webui 后端：FastAPI 应用（上传卫星图像 → Dvorak T 值概率 API）。

- lifespan 启动时经核心库（src/cyclone_intensity）加载并核验可信权重，
  失败即拒绝启动；测试可通过 create_app(predictor=...) 注入替身；
- POST /api/predict：multipart 图像字节 → PIL 解码核验 → 临时文件 →
  Predictor.predict（与 demo 脚本完全一致的文件路径输入方式，用后即删）；
- 根路径挂载 frontend/ 静态目录（check_dir=False：M2 阶段该目录尚未建立，
  GET / 返回 404 属预期；M3 落盘前端文件后自动生效，本文件不再改动）。
"""

from __future__ import annotations

import io
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

from backend.schemas import ErrorResponse, PredictResponse
from cyclone_intensity.model import InferenceError, Predictor

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "model.pt"
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# 与 demo 输入（.jpg 文件路径）兼容的可解码格式 → 临时文件后缀
ALLOWED_IMAGE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp", "WEBP": ".webp"}


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if app.state.predictor is None:
        try:
            app.state.predictor = Predictor(Path(app.state.weights_path))
        except InferenceError as exc:
            raise RuntimeError(f"模型加载失败，服务拒绝启动: {exc}") from exc
    yield


class _FrontendStaticFiles(StaticFiles):
    """frontend/ 未落盘（M2 阶段）时按普通 404 处理而非抛错。

    starlette 1.7.0 在首次请求时经 check_config 对缺失目录抛 RuntimeError；
    此处仅在目录确实不存在时跳过该强校验——lookup_path 对缺失文件本就
    返回 404，M3 落盘前端文件后走原生逻辑，本类无需任何改动。
    """

    async def check_config(self) -> None:
        if self.directory is not None and not Path(self.directory).exists():
            return
        await super().check_config()


def create_app(
    weights: Path = DEFAULT_WEIGHTS,
    predictor: Predictor | None = None,
) -> FastAPI:
    """构建 webui 应用；predictor 供测试注入替身，缺省在 lifespan 加载真权重。"""
    app = FastAPI(
        title="Cyclone Intensity Inference WebUI",
        description="上传热带气旋卫星图像，返回 Dvorak T 值类别概率分布（未校准）",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.weights_path = Path(weights)
    app.state.predictor = predictor

    @app.post(
        "/api/predict",
        response_model=PredictResponse,
        responses={
            400: {"model": ErrorResponse, "description": "空内容/无法解码/不支持格式"},
            413: {"model": ErrorResponse, "description": "超过大小限制"},
            500: {"model": ErrorResponse, "description": "推理失败（模型输出契约异常）"},
        },
    )
    async def predict(image: UploadFile) -> PredictResponse:
        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="上传内容为空")

        declared_size = getattr(image, "size", None)
        if (declared_size is not None and declared_size > MAX_UPLOAD_BYTES) or len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"图像超过大小限制 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
            )

        try:
            with Image.open(io.BytesIO(data)) as im:
                suffix = ALLOWED_IMAGE_FORMATS.get((im.format or "").upper())
                im.load()  # 强制完整解码，拦截截断/损坏文件
        except Exception as exc:
            raise HTTPException(status_code=400, detail="无法解码为有效图片") from exc
        if suffix is None:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的图像格式，允许: {sorted(ALLOWED_IMAGE_FORMATS)}",
            )

        def _run_predict() -> dict:
            with tempfile.NamedTemporaryFile(prefix="webui_upload_", suffix=suffix) as tmp:
                tmp.write(data)
                tmp.flush()
                return app.state.predictor.predict(Path(tmp.name))

        try:
            payload = await run_in_threadpool(_run_predict)
        except InferenceError as exc:
            raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

        return PredictResponse.from_payload(payload)

    app.mount("/", _FrontendStaticFiles(directory=str(FRONTEND_DIR), html=True, check_dir=False), name="frontend")
    return app


app = create_app()
