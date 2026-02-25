from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from app.utility.exceptions.exceptions import AppError
from app.utility.log.log_root import log_ctx

def _error_payload(status_code: int, error_code: str, detail: str):
    return {"status_code": status_code, "error_code": error_code, "detail": detail}

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
            content=_error_payload(
                status_code=exc.status_code,
                error_code=exc.error_code or exc.__class__.__name__.upper(),
                detail=str(exc),
            ),
            headers=exc.headers,
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        log_ctx(
            endpoint=request.url.path,
            method=request.method,
            status_code=exc.status_code,
            error_type=exc.__class__.__name__,
        ).warning("http_exception")
        detail = exc.detail if isinstance(exc.detail, str) else "HTTP error"
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                status_code=exc.status_code,
                error_code="HTTP_ERROR",
                detail=detail,
            ),
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
            status_code=500,
            content=_error_payload(
                status_code=500,
                error_code="INTERNAL_SERVER_ERROR",
                detail="Internal server error",
            ),
        )
