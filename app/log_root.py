import logging
import inspect
from datetime import datetime, timezone
from pythonjsonlogger import jsonlogger


class ContextLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra") or {}
        base = self.extra or {}
        if base or extra:
            kwargs["extra"] = {**base, **extra}
        else:
            kwargs["extra"] = {}
        return msg, kwargs

class UtcTimestampFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.timestamp = datetime.now(timezone.utc).isoformat()
        return True

def setup_logging():
    logger = logging.getLogger()
    if logger.handlers:
        return
    log_handler = logging.StreamHandler()
    log_handler.addFilter(UtcTimestampFilter())

    formatter = jsonlogger.JsonFormatter(
        "%(timestamp)s %(levelname)s %(name)s %(message)s"
    )

    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)

def log_ctx(**fields) -> logging.LoggerAdapter:
    """
    Get a LoggerAdapter with context fields from the caller's module.
    
    Usage in any file:
        from app.log_root import log_ctx
        
        log = log_ctx(endpoint="upload", user_id=123)
        log.info("request_received")
    
    No need to create logger = logging.getLogger(__name__) manually!
    """
    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    module_name = caller.f_globals.get("__name__", "app") if caller else "app"
    
    logger = logging.getLogger(module_name)
    return ContextLoggerAdapter(logger, fields)
