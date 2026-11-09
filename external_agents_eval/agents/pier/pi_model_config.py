from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from external_agents_eval.agents.config import DEFAULT_MODEL


DEFAULT_PROVIDER = "deepseek"
SUPPORTED_PROVIDERS = frozenset({"deepseek", "openai", "zai"})
PROVIDER_API_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "zai": "ZAI_API_KEY",
}
PROVIDER_DEFAULT_BASE_URL = {
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    # Pi 0.84.2's native Z.AI provider uses Chat Completions at this endpoint.
    "zai": "https://api.z.ai/api/coding/paas/v4",
}

# The workflow uses these stable stage names in experiment configuration.  Pi
# role names are kept separate so changing the workflow enum does not change
# the on-disk experiment format.
STAGE_NAMES = ("solver", "A", "I", "R", "S", "repair", "json_corrector")
ROLE_TO_STAGE = {
    "solver": "solver",
    "specification_reviewer": "A",
    "patch_reviewer": "I",
    "repair": "repair",
    "json_corrector": "json_corrector",
    "v5_specification_reviewer": "A",
    "v5_implementation_reviewer": "I",
    "v5_regression_reviewer": "R",
    "v5_summary_reviewer": "S",
}


def _validate_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("base_url cannot contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url cannot contain a query or fragment")
    return value.rstrip("/")


@dataclass(frozen=True)
class PiModelSpec:
    provider: str
    model: str
    base_url: str | None = None
    # Optional Pi ``models.json`` fields for models that are newer than the
    # pinned Pi catalog.  The mapping is deliberately kept provider-shaped so
    # we can add a model without pretending it is part of Pi's built-in list.
    model_metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(
        cls,
        value: Any = None,
        *,
        fallback: "PiModelSpec | None" = None,
    ) -> "PiModelSpec":
        if value is None:
            if fallback is not None:
                return fallback
            return cls.from_model(DEFAULT_MODEL)
        if isinstance(value, str):
            return cls.from_model(value, fallback=fallback)
        if not isinstance(value, Mapping):
            raise ValueError("stage model configuration must be an object or model string")

        raw_model = value.get("model")
        raw_provider = value.get("provider")
        raw_base_url = value.get("base_url", value.get("baseUrl"))
        raw_metadata = value.get("model_metadata", value.get("modelMetadata"))
        if raw_metadata is None and fallback is not None:
            raw_metadata = fallback.model_metadata
        if raw_metadata is None:
            raw_metadata = {}
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("model_metadata must be an object when provided")
        if raw_model is None:
            if fallback is None:
                raise ValueError("stage model configuration requires model")
            raw_model = fallback.model
            if raw_provider is None:
                raw_provider = fallback.provider
            if raw_base_url is None:
                raw_base_url = fallback.base_url
        parsed = cls.from_model(
            str(raw_model),
            provider=str(raw_provider) if raw_provider is not None else None,
            fallback=fallback,
        )
        if raw_base_url is None and fallback is not None and raw_provider is None:
            raw_base_url = fallback.base_url
        if raw_base_url is not None:
            parsed = cls(
                provider=parsed.provider,
                model=parsed.model,
                base_url=_validate_base_url(str(raw_base_url)),
                model_metadata=dict(raw_metadata),
            )
        elif raw_metadata:
            parsed = cls(
                provider=parsed.provider,
                model=parsed.model,
                base_url=parsed.base_url,
                model_metadata=dict(raw_metadata),
            )
        return parsed

    @classmethod
    def from_model(
        cls,
        model: str,
        *,
        provider: str | None = None,
        fallback: "PiModelSpec | None" = None,
    ) -> "PiModelSpec":
        model = str(model).strip()
        if not model:
            raise ValueError("model must be non-empty")
        embedded_provider: str | None = None
        if "/" in model:
            embedded_provider, model = model.split("/", 1)
            embedded_provider = embedded_provider.strip().lower()
            if not embedded_provider or not model:
                raise ValueError(f"Invalid provider/model reference: {model!r}")
        resolved_provider = (
            provider
            or embedded_provider
            or (fallback.provider if fallback else DEFAULT_PROVIDER)
        ).lower()
        if embedded_provider and provider and embedded_provider != resolved_provider:
            raise ValueError(
                f"Model provider {embedded_provider!r} conflicts with provider {resolved_provider!r}"
            )
        if resolved_provider not in SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise ValueError(f"Unsupported Pi provider {resolved_provider!r}; supported: {supported}")
        return cls(
            provider=resolved_provider,
            model=model,
            base_url=(fallback.base_url if fallback and resolved_provider == fallback.provider else None),
            model_metadata=(
                dict(fallback.model_metadata)
                if fallback and resolved_provider == fallback.provider
                else {}
            ),
        )

    @property
    def effective_base_url(self) -> str:
        return self.base_url or PROVIDER_DEFAULT_BASE_URL[self.provider]

    @property
    def api_key_env(self) -> str:
        return PROVIDER_API_KEY_ENV[self.provider]

    @property
    def model_ref(self) -> str:
        return f"{self.provider}/{self.model}"

    def to_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "effective_base_url": self.effective_base_url,
        }
        if self.model_metadata:
            payload["model_metadata"] = dict(self.model_metadata)
        return payload


@dataclass(frozen=True)
class PiStageModelConfig:
    default: PiModelSpec
    stages: Mapping[str, PiModelSpec]

    @classmethod
    def from_value(
        cls,
        value: Any = None,
        *,
        fallback_model: str = DEFAULT_MODEL,
    ) -> "PiStageModelConfig":
        if value is None:
            return cls(default=PiModelSpec.from_model(fallback_model), stages={})
        if isinstance(value, (str, Path)):
            text = str(value)
            path = Path(text).expanduser()
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
            else:
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    # A plain model string is useful when constructing this
                    # object directly; CLI callers normally pass JSON.
                    return cls(default=PiModelSpec.from_model(text), stages={})
        if not isinstance(value, Mapping):
            raise ValueError("stage_model_config must be a JSON object")
        default = PiModelSpec.from_value(
            value.get("default"),
            fallback=PiModelSpec.from_model(fallback_model),
        )
        stages: dict[str, PiModelSpec] = {}
        for key, raw in value.items():
            if key == "default":
                continue
            canonical = str(key)
            if canonical not in STAGE_NAMES:
                raise ValueError(
                    f"Unknown Pi stage {canonical!r}; expected one of {', '.join(STAGE_NAMES)}"
                )
            stages[canonical] = PiModelSpec.from_value(raw, fallback=default)
        return cls(default=default, stages=stages)

    def for_stage(self, stage: str) -> PiModelSpec:
        return self.stages.get(stage, self.default)

    def for_role(self, role: Any) -> PiModelSpec:
        role_name = getattr(role, "value", role)
        stage = ROLE_TO_STAGE.get(str(role_name), str(role_name))
        if stage == "json_corrector":
            # Format correction is an auxiliary call.  Keep it on the summary
            # model when configured; otherwise use the workflow default.
            return self.stages.get(stage, self.stages.get("S", self.default))
        return self.for_stage(stage)

    def required_env_keys(self) -> tuple[str, ...]:
        keys = {spec.api_key_env for spec in self.used_specs()}
        return tuple(sorted(keys))

    def domains(self) -> tuple[str, ...]:
        domains = {
            urlparse(spec.effective_base_url).hostname for spec in self.used_specs()
        }
        return tuple(sorted(domain for domain in domains if domain))

    def used_specs(self) -> tuple[PiModelSpec, ...]:
        """Return the specs used by concrete workflow roles."""
        stages = [
            self.for_stage(stage)
            for stage in ("solver", "A", "I", "R", "S", "repair")
        ]
        stages.append(self.for_role("json_corrector"))
        return tuple(stages)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "default": self.default.to_metadata(),
            "stages": {
                stage: self._metadata_spec(stage).to_metadata() for stage in STAGE_NAMES
            },
            "config_sha256": self.digest,
        }

    def _metadata_spec(self, stage: str) -> PiModelSpec:
        if stage == "json_corrector":
            return self.for_role("json_corrector")
        return self.for_stage(stage)

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "default": self.default.to_metadata(),
                "stages": {
                    stage: self._metadata_spec(stage).to_metadata()
                    for stage in STAGE_NAMES
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def models_json(self, spec: PiModelSpec) -> dict[str, Any] | None:
        if not spec.base_url and not spec.model_metadata:
            return None
        provider_config: dict[str, Any] = {
            "baseUrl": spec.effective_base_url,
        }
        if spec.model_metadata:
            model_definition = dict(spec.model_metadata)
            model_definition["id"] = spec.model
            model_definition.setdefault("name", spec.model)
            model_definition.setdefault("api", "openai-completions")
            model_definition.setdefault("reasoning", True)
            model_definition.setdefault("input", ["text"])
            model_definition.setdefault("contextWindow", 128000)
            model_definition.setdefault("maxTokens", 16384)
            provider_config["api"] = model_definition["api"]
            provider_config["models"] = [model_definition]
        return {"providers": {spec.provider: provider_config}}

    def needs_custom_model_config(self, spec: PiModelSpec) -> bool:
        """Whether Pi must receive a models.json entry for this spec."""
        return bool(spec.model_metadata)


def stage_config_json(config: PiStageModelConfig) -> str:
    """Return the compact, secret-free value passed through Pier kwargs."""
    payload: dict[str, Any] = {"default": config.default.to_metadata()}
    payload.update(
        {
            stage: config.for_stage(stage).to_metadata()
            for stage in STAGE_NAMES
            if stage in config.stages
        }
    )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
