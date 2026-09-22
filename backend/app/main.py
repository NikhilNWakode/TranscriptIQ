"""FastAPI entry point. Run with:  uvicorn app.main:app --reload --port 8000  (from /backend)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router
from .config import get_settings
from .services.state import AppState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("transcriptiq")


def create_app(app_state: AppState | None = None, sync_on_startup: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = app_state or AppState(get_settings())
        app.state.app_state = s
        if sync_on_startup:
            try:
                report = s.refresh()
                log.info("Startup scan: %d files (%d added, %d updated, %d removed, %d unchanged, %d failed)",
                         report.total, len(report.added), len(report.updated), len(report.removed),
                         len(report.unchanged), len(report.failed))
            except Exception:
                log.exception("Startup transcript scan failed")
        yield
        s.db.close()

    app = FastAPI(title="TranscriptIQ", version="1.0.0", lifespan=lifespan)
    settings = get_settings()
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST", "PATCH"],
                       allow_headers=["Content-Type"])

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        return JSONResponse(status_code=422, content={"detail": f"Invalid request: {first.get('msg', 'bad input')}"})

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("Unhandled error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal error — see server logs."})

    app.include_router(router)
    return app


app = create_app()
