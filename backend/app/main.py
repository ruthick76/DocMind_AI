import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.api.routes import router
from app.core.config import Settings
from app.core.errors import AppError
from app.services.ingestion import Ingestion
from app.services.providers import Providers
from app.services.rag import RagService
from app.services.storage import Store

logger = logging.getLogger("docmind")


def create_app(settings=None, providers=None):
    settings = settings or Settings()
    providers = providers or Providers(settings)

    @asynccontextmanager
    async def lifespan(app):
        app.state.settings = settings
        app.state.providers = providers
        app.state.store = Store(settings, providers)
        app.state.ingestion = Ingestion(settings)
        app.state.rag = RagService(app.state.store, providers)
        yield

    app = FastAPI(title="DocMind AI", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_credentials=False, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["Content-Type"])

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Do not return original inputs: requests may contain private document text.
        return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "message": "Invalid request. Check required fields, question length, and UUID identifiers."}})

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": "http_error", "message": "Request could not be processed."}})

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        # Provider exception text may contain request data. Log class only.
        logger.error("Request failed (%s)", type(exc).__name__)
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "An internal operation failed. Check storage permissions and server diagnostics."}})

    app.include_router(router)
    return app


app = create_app()
