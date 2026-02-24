
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.utility.exceptions.exceptions import AppError
from app.utility.log.log_root import log_ctx


def register_exception_handlers(app: FastAPI):
    @app.exception_handler(AppError)
    async def app_error_handler(request, exc: AppError):
        log_ctx(
            endpoint=request.url.path,
            method=request.method,
            status_code=exc.status_code,
            error_type=exc.__class__.__name__,
        ).warning("app_error")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc)},
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request, exc: Exception):
        log_ctx(
            endpoint=request.url.path,
            method=request.method,
            error_type=exc.__class__.__name__,
        ).exception("unhandled_exception")
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )
