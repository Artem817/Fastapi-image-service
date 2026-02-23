class AppError(Exception):
    """Base domain error with an HTTP status code."""

    def __init__(
        self,
        message: str | None = None,
        *,
        status_code: int = 400,
        headers: dict[str, str] | None = None,
    ):
        self.message = message or "Application error"
        self.status_code = status_code
        self.headers = headers
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
    
class ImageNotFoundError(AppError):
    def __init__(
        self,
        user_id: int | None = None,
        file_id: str | None = None,
        detail: str | None = None,
    ):
        self.user_id = user_id
        self.file_id = file_id
        message = detail or self._build_message()
        super().__init__(message, status_code=404)

    def _build_message(self) -> str:
        if self.file_id:
            return f"Image '{self.file_id}' not found"
        if self.user_id:
            return f"No active image for user '{self.user_id}'"
        return "Image not found"
    

class InvalidFileError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status_code=400)
        
class ModelNotAvailableError(AppError):
    def __init__(self):
        super().__init__("Model weights not available", status_code=503)


class ImageProcessingError(AppError):
    def __init__(self, message: str, operation: str | None = None):
        detail = f"Image processing failed: {message}"
        if operation:
            detail = f"Image {operation} failed: {message}"
        super().__init__(detail, status_code=400)


class RateLimitExceededError(AppError):
    def __init__(self, max_uploads: int, retry_after: int | None = None):
        safe_retry_after = max(retry_after or 0, 0)
        super().__init__(
            (
                f"Too many requests. Maximum {max_uploads} images per hour. "
                f"Please wait {safe_retry_after} seconds."
            ),
            status_code=429,
            headers={"Retry-After": str(safe_retry_after)},
        )
