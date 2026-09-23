"""Tests for configuration, logging redaction, the schema registry and the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.config import VALID_AI_PROVIDERS, load_settings, settings
from src.logging_setup import SecretRedactingFilter, get_logger
from src.schema import (
    MONTH_LABELS,
    RAW_TO_CANONICAL,
    UNAVAILABLE_ROLES,
    ColumnRegistry,
    Role,
    build_registry,
    canonical_columns_for,
    describe_panel_alignment,
    month_label,
)


class TestSettings:
    """Configuration loading."""

    def test_defaults_work_without_a_dotenv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in ("AI_PROVIDER", "RANDOM_STATE", "LOG_LEVEL", "HIGH_UTILISATION_THRESHOLD"):
            monkeypatch.delenv(key, raising=False)
        resolved = load_settings()
        assert resolved.random_state == 42
        assert resolved.ai_provider == "none"
        assert resolved.high_utilisation_threshold == 0.80

    def test_analytics_only_mode_when_no_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The platform must be fully usable with no AI configured."""
        monkeypatch.setenv("AI_PROVIDER", "none")
        resolved = load_settings()
        assert resolved.ai_enabled is False
        assert "analytics-only" in resolved.ai_status_message.lower()

    def test_hosted_provider_without_a_key_is_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AI_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        resolved = load_settings()
        assert resolved.ai_enabled is False
        assert "no api key" in resolved.ai_status_message.lower()

    def test_hosted_provider_with_a_key_is_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AI_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value-1234567890")
        resolved = load_settings()
        assert resolved.ai_enabled is True

    def test_ollama_needs_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_PROVIDER", "ollama")
        assert load_settings().ai_enabled is True

    def test_status_message_never_leaks_the_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "sk-super-secret-key-abcdef123456"
        monkeypatch.setenv("AI_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", secret)
        resolved = load_settings()
        assert secret not in resolved.ai_status_message
        assert secret not in repr(resolved)

    def test_invalid_provider_falls_back_with_a_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AI_PROVIDER", "not_a_provider")
        resolved = load_settings()
        assert resolved.ai_provider == "none"
        assert any("not recognised" in w for w in resolved.warnings)

    @pytest.mark.parametrize(
        "variable,value",
        [
            ("HIGH_UTILISATION_THRESHOLD", "99"),
            ("OUTLIER_IQR_MULTIPLIER", "-1"),
            ("MIN_ROWS_FOR_ANALYSIS", "0"),
            ("LOG_LEVEL", "VERBOSE"),
            ("AI_TEMPERATURE", "9"),
        ],
    )
    def test_out_of_range_values_degrade_not_crash(
        self, monkeypatch: pytest.MonkeyPatch, variable: str, value: str
    ) -> None:
        monkeypatch.setenv(variable, value)
        resolved = load_settings()
        assert resolved.warnings, f"{variable}={value} should record a warning"

    def test_unparseable_numbers_fall_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANDOM_STATE", "not_a_number")
        assert load_settings().random_state == 42

    def test_blank_values_are_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_PROVIDER", "   ")
        assert load_settings().ai_provider == "none"

    def test_paths_resolve_against_the_project_root(self) -> None:
        assert settings.raw_dir.is_absolute()
        assert settings.raw_dir.parent.parent == settings.project_root

    def test_settings_are_immutable(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            settings.random_state = 99  # type: ignore[misc]

    def test_provider_list_is_documented(self) -> None:
        assert VALID_AI_PROVIDERS == frozenset({"none", "openai", "gemini", "ollama"})


class TestSecretRedaction:
    """The logging safety net."""

    def _record(self, message: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__, lineno=1,
            msg=message, args=(), exc_info=None,
        )

    def test_redacts_an_openai_style_key(self) -> None:
        record = self._record("key is sk-abcdefghijklmnopqrstuvwxyz123456")
        SecretRedactingFilter().filter(record)
        assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in str(record.msg)
        assert "REDACTED" in str(record.msg)

    def test_redacts_a_google_style_key(self) -> None:
        record = self._record("token AIzaSyD-1234567890abcdefghijklmnopqrstuvw")
        SecretRedactingFilter().filter(record)
        assert "AIzaSy" not in str(record.msg)

    def test_redacts_labelled_secrets(self) -> None:
        for message in ("api_key=abc123", "password: hunter2", "token=xyz789"):
            record = self._record(message)
            SecretRedactingFilter().filter(record)
            assert "REDACTED" in str(record.msg), message

    def test_redacts_a_configured_environment_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "my-unique-secret-value-here")
        record = self._record("connecting with my-unique-secret-value-here now")
        SecretRedactingFilter().filter(record)
        assert "my-unique-secret-value-here" not in str(record.msg)

    def test_leaves_ordinary_messages_alone(self) -> None:
        record = self._record("Loaded 30,000 rows from the dataset.")
        SecretRedactingFilter().filter(record)
        assert record.msg == "Loaded 30,000 rows from the dataset."

    def test_filter_never_blocks_a_record(self) -> None:
        assert SecretRedactingFilter().filter(self._record("anything")) is True


class TestLogger:
    """Logger construction."""

    def test_names_are_namespaced(self) -> None:
        assert get_logger("src.data_loader").name == "finrisk.data_loader"

    def test_main_becomes_app(self) -> None:
        assert get_logger("__main__").name == "finrisk.app"

    def test_redaction_filter_is_installed(self) -> None:
        """Every handler we install must redact.

        Only our own handlers are checked: pytest injects its log-capture
        handlers into the same logger, and those are not ours to configure.
        """
        from logging.handlers import RotatingFileHandler

        root = logging.getLogger("finrisk")
        ours = [
            handler
            for handler in root.handlers
            if type(handler) in (logging.StreamHandler, RotatingFileHandler)
        ]
        assert ours, "no handlers were installed"
        for handler in ours:
            assert any(isinstance(f, SecretRedactingFilter) for f in handler.filters)

    def test_repeated_calls_do_not_duplicate_handlers(self) -> None:
        get_logger("a")
        before = len(logging.getLogger("finrisk").handlers)
        get_logger("b")
        get_logger("c")
        assert len(logging.getLogger("finrisk").handlers) == before


class TestSchemaRegistry:
    """The column role registry."""

    def test_detects_roles_in_a_full_frame(self, minimal_credit_df: pd.DataFrame) -> None:
        registry = build_registry(minimal_credit_df)
        for role in (
            Role.CLIENT_ID, Role.CREDIT_LIMIT, Role.AGE,
            Role.TARGET, Role.BILL_AMOUNT, Role.PAYMENT_AMOUNT, Role.PAYMENT_STATUS,
        ):
            assert registry.has(role), f"{role.value} should be detected"

    def test_absent_roles_are_reported_missing(self) -> None:
        registry = build_registry(pd.DataFrame({"client_id": [1]}))
        assert not registry.has(Role.CREDIT_LIMIT)
        assert Role.CREDIT_LIMIT in registry.missing(Role.CREDIT_LIMIT, Role.AGE)

    def test_panel_columns_are_month_indexed(self, minimal_credit_df: pd.DataFrame) -> None:
        panel = build_registry(minimal_credit_df).panel_columns(Role.BILL_AMOUNT)
        assert list(panel) == [1, 2, 3, 4, 5, 6]
        assert panel[1] == "bill_amt_m1"

    def test_one_returns_the_most_recent_panel_month(
        self, minimal_credit_df: pd.DataFrame
    ) -> None:
        assert build_registry(minimal_credit_df).one(Role.BILL_AMOUNT) == "bill_amt_m1"

    def test_require_raises_for_a_missing_role(self) -> None:
        registry = build_registry(pd.DataFrame({"client_id": [1]}))
        with pytest.raises(KeyError, match="not available"):
            registry.require(Role.CREDIT_LIMIT)

    def test_prefers_label_column_over_code(self, small_featured_df: pd.DataFrame) -> None:
        registry = build_registry(small_featured_df)
        assert registry.one(Role.EDUCATION) == "education"

    def test_engineered_roles_appear_after_feature_engineering(
        self, small_featured_df: pd.DataFrame
    ) -> None:
        registry = build_registry(small_featured_df)
        for role in (Role.UTILISATION, Role.DELINQUENCY, Role.LIMIT_BAND, Role.AGE_BAND):
            assert registry.has(role)

    def test_unavailable_roles_are_explained(self, small_featured_df: pd.DataFrame) -> None:
        explanations = build_registry(small_featured_df).unavailable_explanations()
        assert "income" in explanations
        assert "no income column" in explanations["income"].lower()

    def test_income_explanation_mentions_the_proxy(self) -> None:
        text = UNAVAILABLE_ROLES[Role.INCOME].lower()
        assert "credit_limit" in text and "never as income" in text

    def test_empty_frame_has_no_roles(self) -> None:
        assert build_registry(pd.DataFrame()).available_role_names() == ()

    def test_has_requires_all_supplied_roles(self, minimal_credit_df: pd.DataFrame) -> None:
        registry = build_registry(minimal_credit_df)
        assert registry.has(Role.AGE, Role.CREDIT_LIMIT)
        assert not registry.has(Role.AGE, Role.INCOME)
        assert registry.has_any(Role.AGE, Role.INCOME)


class TestSchemaConstants:
    """Panel indexing and the raw-name mapping."""

    def test_month_one_is_the_most_recent(self) -> None:
        assert MONTH_LABELS[1] == "Sep 2005"
        assert MONTH_LABELS[6] == "Apr 2005"

    def test_month_label_falls_back_gracefully(self) -> None:
        assert month_label(99) == "Month 99"

    def test_pay_zero_maps_to_month_one(self) -> None:
        """The source has no PAY_1; PAY_0 is September."""
        assert RAW_TO_CANONICAL["PAY_0"] == "pay_status_m1"
        assert "PAY_1" in RAW_TO_CANONICAL  # tolerated in redistributions

    def test_alignment_note_explains_the_offset(self) -> None:
        text = describe_panel_alignment()
        assert "m+1" in text or "month m+1" in text
        assert "PAY_0" in text

    def test_canonical_mapping_ignores_unknown_names(self) -> None:
        mapping = canonical_columns_for(["ID", "AGE", "TOTALLY_UNKNOWN"])
        assert mapping == {"ID": "client_id", "AGE": "age"}


class TestPipeline:
    """Pipeline orchestration on synthetic data."""

    def test_runs_end_to_end(self, tmp_csv: Path, tmp_path: Path) -> None:
        from src.pipeline import run_pipeline

        result = run_pipeline(path=tmp_csv, persist=False, raise_on_invalid=False)
        assert len(result.raw) == 3
        assert len(result.features.columns) > len(result.raw.columns)
        assert result.duration_seconds > 0

    def test_produces_every_artefact(self, tmp_csv: Path) -> None:
        from src.pipeline import run_pipeline

        result = run_pipeline(path=tmp_csv, persist=False, raise_on_invalid=False)
        assert result.quality is not None
        assert result.cleaning is not None
        assert result.feature_report is not None
        assert result.registry is not None

    def test_summary_is_complete(self, tmp_csv: Path) -> None:
        from src.pipeline import run_pipeline

        summary = run_pipeline(path=tmp_csv, persist=False, raise_on_invalid=False).summary()
        for key in (
            "rows_raw", "rows_final", "columns_final", "panel_rows",
            "features_created", "quality_score", "duration_seconds",
        ):
            assert key in summary

    def test_payload_excludes_dataframes(self, tmp_csv: Path) -> None:
        """The AI layer gets aggregates only, never client-level rows."""
        import json

        from src.pipeline import run_pipeline

        result = run_pipeline(path=tmp_csv, persist=False, raise_on_invalid=False)
        payload = result.to_dict()

        # No dataframe may appear anywhere in the payload, at any depth.
        def assert_no_frames(node: object, path: str = "") -> None:
            assert not isinstance(node, pd.DataFrame), f"DataFrame found at {path}"
            assert not isinstance(node, pd.Series), f"Series found at {path}"
            if isinstance(node, dict):
                for key, value in node.items():
                    assert_no_frames(value, f"{path}.{key}")
            elif isinstance(node, (list, tuple)):
                for index, value in enumerate(node):
                    assert_no_frames(value, f"{path}[{index}]")

        assert_no_frames(payload)
        assert "raw" not in payload
        assert "clean" not in payload
        assert json.dumps(payload), "payload must be JSON-serialisable"

    def test_provenance_is_complete(self, tmp_csv: Path) -> None:
        from src.pipeline import run_pipeline

        provenance = run_pipeline(
            path=tmp_csv, persist=False, raise_on_invalid=False
        ).dataset_provenance()
        for key in ("name", "source", "url", "licence", "citation", "period", "currency"):
            assert provenance[key]
        assert "archive.ics.uci.edu" in provenance["url"]

    def test_persists_when_asked(
        self, tmp_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Persistence must work, and must write only inside the temp directory.

        The output directory is redirected so the test cannot overwrite the real
        30,000-row artefacts in ``data/processed``.
        """
        import dataclasses

        import src.config as config_module
        import src.data_processing as processing
        from src.pipeline import run_pipeline

        target = tmp_path / "processed_out"
        redirected = dataclasses.replace(
            config_module.settings, processed_dir=target, models_dir=tmp_path / "models_out"
        )
        # save_processed and the pipeline both read their own module-level alias.
        monkeypatch.setattr(processing, "settings", redirected)
        monkeypatch.setattr("src.pipeline.settings", redirected)

        result = run_pipeline(path=tmp_csv, persist=True, raise_on_invalid=False)

        assert len(result.written_files) >= 1
        for written in result.written_files:
            assert target in written.parents, f"{written} escaped the temp directory"
        assert (target / processing.PROCESSED_WIDE_FILENAME).exists()

    def test_safe_loader_reports_a_missing_file(self) -> None:
        from src.pipeline import safe_load_pipeline_data

        result, error = safe_load_pipeline_data(path="does/not/exist.csv")
        assert result is None
        assert error is not None
        assert "download_data" in error

    def test_safe_loader_message_is_actionable(self) -> None:
        from src.pipeline import safe_load_pipeline_data

        _result, error = safe_load_pipeline_data(path="nope.xls")
        assert "Fix:" in error or "could not be loaded" in error


@pytest.mark.integration
class TestRealPipeline:
    """The pipeline against the real dataset."""

    def test_shapes_are_as_verified(self, real_pipeline) -> None:
        summary = real_pipeline.summary()
        assert summary["rows_raw"] == 30_000
        assert summary["rows_final"] == 30_000
        assert summary["columns_raw"] == 25
        assert summary["panel_rows"] == 180_000

    def test_feature_counts_are_as_verified(self, real_pipeline) -> None:
        summary = real_pipeline.summary()
        assert summary["features_created"] == 37
        assert summary["features_skipped"] == 5

    def test_quality_score_is_as_verified(self, real_pipeline) -> None:
        assert real_pipeline.summary()["quality_score"] == pytest.approx(98.18, abs=0.01)

    def test_validation_passed(self, real_pipeline) -> None:
        assert real_pipeline.validation.is_valid
        assert real_pipeline.validation.warnings == ()

    def test_no_rows_were_lost(self, real_pipeline) -> None:
        assert real_pipeline.cleaning.rows_removed == 0

    def test_expected_roles_are_available(self, real_pipeline) -> None:
        roles = set(real_pipeline.registry.available_role_names())
        for expected in (
            "client_id", "credit_limit", "age", "target",
            "bill_amount", "payment_amount", "payment_status",
            "utilisation", "delinquency", "limit_band", "age_band",
        ):
            assert expected in roles

    def test_income_role_is_absent_and_explained(self, real_pipeline) -> None:
        assert not real_pipeline.registry.has(Role.INCOME)
        assert "income" in real_pipeline.registry.unavailable_explanations()

    def test_payload_is_serialisable_and_compact(self, real_pipeline) -> None:
        import json

        serialised = json.dumps(real_pipeline.to_dict())
        assert len(serialised) < 500_000, "the AI payload must stay small"

    def test_runs_in_reasonable_time(self, real_pipeline) -> None:
        assert real_pipeline.duration_seconds < 60
