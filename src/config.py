"""
Central configuration for FinRisk AI.

Settings are read from environment variables (optionally populated from a local
`.env` file) and exposed through one immutable :class:`Settings` object. Two
rules matter here:

1. **No secret ever appears in this file.** Keys are read from the environment
   at call time and are never logged, echoed or written to disk.
2. **Every setting has a working default.** The platform must run immediately
   after a clone with no `.env` present at all, in analytics-only mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

try:  # python-dotenv is a declared dependency, but never let config be the thing that breaks
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - defensive only
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:  # type: ignore[misc]
        """No-op stand-in used only if python-dotenv is unavailable."""
        return False


# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #

# config.py lives in <root>/src/, so the project root is two levels up.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Load .env from the project root if it exists. `override=False` means a real
# environment variable always beats the file, which is what deployments expect.
load_dotenv(PROJECT_ROOT / ".env", override=False)


# --------------------------------------------------------------------------- #
# Typed environment readers
# --------------------------------------------------------------------------- #

_TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "y", "on"})


def _env_str(name: str, default: str) -> str:
    """Read a string setting, treating blank/whitespace-only values as unset."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    """Read an int setting, falling back to ``default`` on an unparseable value."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float setting, falling back to ``default`` on an unparseable value."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean setting from common truthy spellings."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _resolve(path_value: str) -> Path:
    """Resolve a configured path against the project root when it is relative."""
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

VALID_AI_PROVIDERS: Final[frozenset[str]] = frozenset({"none", "openai", "gemini", "ollama"})


@dataclass(frozen=True)
class Settings:
    """Immutable resolved configuration for one run of the platform."""

    # --- paths ---
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    log_dir: Path
    raw_dataset_filename: str

    # --- analysis ---
    random_state: int
    min_rows_for_analysis: int
    high_utilisation_threshold: float
    outlier_iqr_multiplier: float

    # --- AI (optional) ---
    ai_provider: str
    ai_model: str
    ai_timeout_seconds: int
    ai_max_output_tokens: int
    ai_temperature: float

    # --- logging ---
    log_level: str
    log_to_file: bool

    # --- derived ---
    warnings: tuple[str, ...] = field(default=())

    # ---------------------------------------------------------------- #
    # AI helpers
    # ---------------------------------------------------------------- #

    @property
    def ai_enabled(self) -> bool:
        """True only when a provider is selected *and* usable.

        Ollama runs locally and needs no key; the hosted providers do. This is
        the single check the rest of the codebase should use before attempting
        any AI call.
        """
        if self.ai_provider == "none":
            return False
        if self.ai_provider == "ollama":
            return True
        return bool(self.ai_api_key)

    @property
    def ai_api_key(self) -> str:
        """Fetch the key for the active provider straight from the environment.

        Deliberately a property rather than a stored field so the secret is not
        held on a long-lived object, and never lands in a repr or a log line.
        """
        env_var = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }.get(self.ai_provider)
        if env_var is None:
            return ""
        return os.getenv(env_var, "").strip()

    @property
    def ai_status_message(self) -> str:
        """Human-readable provider status. Never includes any part of a key."""
        if self.ai_provider == "none":
            return "AI narration disabled (AI_PROVIDER=none). Analytics-only mode."
        if self.ai_provider == "ollama":
            return f"AI provider: ollama (local), model '{self.ai_model}'."
        if self.ai_api_key:
            return f"AI provider: {self.ai_provider}, model '{self.ai_model}'. Key detected."
        return (
            f"AI provider '{self.ai_provider}' selected but no API key found. "
            "Running in analytics-only mode."
        )

    # ---------------------------------------------------------------- #
    # Filesystem
    # ---------------------------------------------------------------- #

    def ensure_directories(self) -> None:
        """Create the writable output directories. Never touches ``raw_dir``."""
        for directory in (self.processed_dir, self.models_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)


def _default_ai_model(provider: str) -> str:
    """Per-provider default model, overridable by its own env var."""
    if provider == "openai":
        return _env_str("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "gemini":
        return _env_str("GEMINI_MODEL", "gemini-1.5-flash")
    if provider == "ollama":
        return _env_str("OLLAMA_MODEL", "llama3.1")
    return ""


def load_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment.

    Invalid values are corrected to safe defaults and recorded in
    ``settings.warnings`` rather than raising, so a typo in `.env` degrades the
    configuration instead of preventing the application from starting.
    """
    warnings: list[str] = []

    provider = _env_str("AI_PROVIDER", "none").lower()
    if provider not in VALID_AI_PROVIDERS:
        warnings.append(
            f"AI_PROVIDER='{provider}' is not recognised. "
            f"Expected one of {sorted(VALID_AI_PROVIDERS)}. Falling back to 'none'."
        )
        provider = "none"

    high_util = _env_float("HIGH_UTILISATION_THRESHOLD", 0.80)
    if not 0.0 < high_util <= 5.0:
        warnings.append(
            f"HIGH_UTILISATION_THRESHOLD={high_util} is outside the sensible range "
            "(0, 5]. Falling back to 0.80."
        )
        high_util = 0.80

    iqr_multiplier = _env_float("OUTLIER_IQR_MULTIPLIER", 1.5)
    if iqr_multiplier <= 0:
        warnings.append(
            f"OUTLIER_IQR_MULTIPLIER={iqr_multiplier} must be positive. Falling back to 1.5."
        )
        iqr_multiplier = 1.5

    min_rows = _env_int("MIN_ROWS_FOR_ANALYSIS", 30)
    if min_rows < 1:
        warnings.append(f"MIN_ROWS_FOR_ANALYSIS={min_rows} must be >= 1. Falling back to 30.")
        min_rows = 30

    log_level = _env_str("LOG_LEVEL", "INFO").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        warnings.append(f"LOG_LEVEL='{log_level}' is not valid. Falling back to 'INFO'.")
        log_level = "INFO"

    temperature = _env_float("AI_TEMPERATURE", 0.2)
    if not 0.0 <= temperature <= 2.0:
        warnings.append(f"AI_TEMPERATURE={temperature} outside [0, 2]. Falling back to 0.2.")
        temperature = 0.2

    return Settings(
        project_root=PROJECT_ROOT,
        raw_dir=_resolve(_env_str("DATA_RAW_DIR", "data/raw")),
        processed_dir=_resolve(_env_str("DATA_PROCESSED_DIR", "data/processed")),
        models_dir=_resolve(_env_str("MODELS_DIR", "models")),
        log_dir=_resolve(_env_str("LOG_DIR", "logs")),
        raw_dataset_filename=_env_str("RAW_DATASET_FILENAME", ""),
        random_state=_env_int("RANDOM_STATE", 42),
        min_rows_for_analysis=min_rows,
        high_utilisation_threshold=high_util,
        outlier_iqr_multiplier=iqr_multiplier,
        ai_provider=provider,
        ai_model=_default_ai_model(provider),
        ai_timeout_seconds=_env_int("AI_TIMEOUT_SECONDS", 30),
        ai_max_output_tokens=_env_int("AI_MAX_OUTPUT_TOKENS", 800),
        ai_temperature=temperature,
        log_level=log_level,
        log_to_file=_env_bool("LOG_TO_FILE", True),
        warnings=tuple(warnings),
    )


# Module-level singleton. Import this rather than calling load_settings() again,
# so every layer of the app sees one consistent configuration.
settings: Final[Settings] = load_settings()
