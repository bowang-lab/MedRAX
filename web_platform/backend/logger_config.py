"""
Centralized logging configuration for MedRAX Backend
Uses structlog for structured logging with proper formatting
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import structlog

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure standard logging
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

# Suppress noisy third-party loggers
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("multipart").setLevel(logging.WARNING)

# Timestamper for consistent timestamps
timestamper = structlog.processors.TimeStamper(fmt="iso")

# Configure structlog processors
shared_processors = [
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    timestamper,
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]

# Console renderer with colors for development
structlog.configure(
    processors=shared_processors + [
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Setup formatters for different outputs
console_formatter = structlog.stdlib.ProcessorFormatter(
    processor=structlog.dev.ConsoleRenderer(colors=True),
    foreign_pre_chain=shared_processors,
)

file_formatter = structlog.stdlib.ProcessorFormatter(
    processor=structlog.processors.JSONRenderer(),
    foreign_pre_chain=shared_processors,
)

# Setup handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)

# File handler for JSON logs
log_file = LOGS_DIR / f"medrax_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)

# Error-only file handler
error_log_file = LOGS_DIR / f"medrax_errors_{datetime.now().strftime('%Y%m%d')}.log"
error_handler = logging.FileHandler(error_log_file)
error_handler.setFormatter(file_formatter)
error_handler.setLevel(logging.ERROR)

# Configure root logger
root_logger = logging.getLogger()
root_logger.handlers.clear()  # Clear default handlers to avoid duplicates
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)
root_logger.addHandler(error_handler)
root_logger.setLevel(logging.INFO)  # Set to INFO for cleaner console output


def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger

    Usage:
        logger = get_logger(__name__)
        logger.info("session_created", session_id=sid, user="admin")
        logger.error("tool_failed", tool="classifier", error=str(e), exc_info=True)
    """
    return structlog.get_logger(name)


# Export convenience functions
log = get_logger(__name__)

def log_tool_execution(tool_name: str, duration_ms: float, success: bool, **kwargs):
    """Log tool execution metrics"""
    log.info(
        "tool_executed",
        tool=tool_name,
        duration_ms=round(duration_ms, 2),
        success=success,
        **kwargs
    )

def log_api_request(method: str, path: str, status_code: int, duration_ms: float, **kwargs):
    """Log API request metrics"""
    log.info(
        "api_request",
        method=method,
        path=path,
        status=status_code,
        duration_ms=round(duration_ms, 2),
        **kwargs
    )

def log_error(error_type: str, message: str, **kwargs):
    """Log error with context"""
    log.error(
        "error_occurred",
        error_type=error_type,
        message=message,
        **kwargs,
        exc_info=True
    )


# Example usage:
if __name__ == "__main__":
    logger = get_logger("test")

    logger.info("test_message", key="value", number=42)
    logger.warning("test_warning", reason="demonstration")

    try:
        1 / 0
    except Exception as e:
        logger.error("test_error", error=str(e), exc_info=True)

    log_tool_execution("test_tool", 123.45, True, image_path="temp/test.png")
    log_api_request("POST", "/api/test", 200, 45.2, session_id="test123")

