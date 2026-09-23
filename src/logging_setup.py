"""
Logging configuration for FinRisk AI.

One entry point, :func:`get_logger`, used by every module. Logging is
configured once on first use and is safe to call repeatedly (Streamlit reruns
modules constantly, so idempotency matters).

Privacy stance: log *shapes, counts and decisions*, never row-level financial
values and never credentials. A :class:`SecretRedactingFilter` is installed as a
second line of defence in case a key is ever passed into a log call by mistake.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Final

from src.config import settings

_CONSOLE_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
_FILE_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)-22s | %(funcName)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

_ROOT_LOGGER_NAME: Final[str] = "finrisk"
_configured: bool = False


class SecretRedactingFilter(logging.Filter):
    """Masks anything that looks like a credential before it reaches a handler.

    This is a safety net, not a licence to log secrets. It covers two cases:
    values of known API-key environment variables appearing verbatim in a
    message, and common key-shaped token patterns.
    """

    _KEY_ENV_VARS: Final[tuple[str, ...]] = (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
    )

    _PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
        re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),          # OpenAI-style
        re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),          # Google-style
        re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*\S+"),
    )

    _REDACTED: Final[str] = "***REDACTED***"

    def filter(self, record: logging.LogRecord) -> bool:
        """Always returns True; rewrites ``record.msg`` in place when needed."""
        message = record.getMessage()
        original = message

        for env_var in self._KEY_ENV_VARS:
            secret = os.getenv(env_var, "").strip()
            # Ignore trivially short values to avoid nonsense substitutions.
            if len(secret) >= 8 and secret in message:
                message = message.replace(secret, self._REDACTED)

        for pattern in self._PATTERNS:
            message = pattern.sub(self._REDACTED, message)

        if message != original:
            record.msg = message
            record.args = ()
        return True


def _build_file_handler(log_dir: Path, level: int) -> logging.Handler | None:
    """Create the rotating file handler, or return None if the path is unusable.

    A read-only or missing directory must never stop the application, so any
    failure here degrades to console-only logging.
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler

        handler = RotatingFileHandler(
            log_dir / "finrisk.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
        handler.addFilter(SecretRedactingFilter())
        return handler
    except OSError:
        return None


def configure_logging(force: bool = False) -> logging.Logger:
    """Configure the ``finrisk`` logger tree once and return its root logger.

    Args:
        force: Rebuild handlers even if logging was already configured. Useful
            in tests that need to change the level mid-session.

    Returns:
        The configured ``finrisk`` logger.
    """
    global _configured

    root = logging.getLogger(_ROOT_LOGGER_NAME)

    if _configured and not force:
        return root

    level = getattr(logging, settings.log_level, logging.INFO)
    root.setLevel(level)
    root.handlers.clear()
    # Keep our records out of the interpreter's root logger.
    root.propagate = False

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    console.addFilter(SecretRedactingFilter())
    root.addHandler(console)

    if settings.log_to_file:
        file_handler = _build_file_handler(settings.log_dir, level)
        if file_handler is not None:
            root.addHandler(file_handler)
        else:
            root.warning(
                "Could not open the log directory at %s. Continuing with console logging only.",
                settings.log_dir,
            )

    _configured = True

    # Surface configuration problems exactly once, at startup.
    for warning in settings.warnings:
        root.warning("Configuration: %s", warning)

    root.debug(
        "Logging ready. level=%s file_logging=%s ai_provider=%s",
        settings.log_level,
        settings.log_to_file,
        settings.ai_provider,
    )
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, configuring the tree on first call.

    Args:
        name: Usually ``__name__``. A leading ``src.`` is stripped so logger
            names read as ``finrisk.data_loader`` rather than
            ``finrisk.src.data_loader``.

    Returns:
        A logger under the ``finrisk`` namespace.
    """
    configure_logging()
    short = name.removeprefix("src.").removeprefix("finrisk.")
    if short in {"", "__main__"}:
        short = "app"
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short}")
