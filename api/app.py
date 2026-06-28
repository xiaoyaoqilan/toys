from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .routes import router
from .container import get_container


WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Finance Multi-Source RAG Platform",
        version="0.1.0",
        description="金融多信源校验分布式 RAG 平台",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")

    @app.on_event("startup")
    def _startup() -> None:
        get_container()

    @app.get("/")
    def _root() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/healthz")
    def _healthz() -> dict:
        return {"status": "ok"}

    return app
