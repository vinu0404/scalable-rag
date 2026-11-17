"""Describe processing choices and calculate a repeatable settings hash.

Save the complete configuration along with its fingerprint. A hash alone
cannot tell a future worker which settings to use.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(frozen=True, kw_only=True)
class ComponentSpec:
    """Describe one extractor, chunker, or embedder without creating it.

    driver names its adapter. implementation_version identifies the code's
    behavior. options_json holds output-affecting settings as a JSON object.
    Using a string prevents callers mutating a nested settings dictionary.

    Include model revision, dimensions, prompts, normalization, and other
    output-affecting options where relevant. Do not put credentials here.
    An adapter must later validate its own required options and capabilities.
    """

    driver: str
    implementation_version: str
    options_json: str = field(default="{}", repr=False)

    def __post_init__(self) -> None:
        """Reject blank names, invalid JSON, or options that are not an object.

        Also reject non-finite numbers because they have no standard JSON
        representation. This does not check provider-specific setting names.
        """
        if not isinstance(self.driver, str) or not self.driver.strip():
            raise ValueError("driver must not be blank")
        if (
            not isinstance(self.implementation_version, str)
            or not self.implementation_version.strip()
        ):
            raise ValueError("implementation_version must not be blank")
        if not isinstance(self.options_json, str):
            raise ValueError("options_json must be a JSON object string")
        try:
            options = json.loads(self.options_json)
            if not isinstance(options, dict):
                raise ValueError
            json.dumps(options, allow_nan=False)
        except (ValueError, TypeError, RecursionError):
            raise ValueError(
                "options_json must contain a valid JSON object with finite numbers"
            ) from None

    def to_dict(self) -> dict[str, object]:
        """Return a fresh dictionary for saving the complete component settings.

        Changing this returned dictionary does not change the stored spec.
        """
        return {
            "driver": self.driver,
            "implementation_version": self.implementation_version,
            "options": json.loads(self.options_json),
        }


@dataclass(frozen=True, kw_only=True)
class PipelineConfig:
    """Describe the extraction, chunking, and embedding used for a generation.

    These choices identify reusable processing outputs. Connection addresses,
    API ports, retry delays, and credentials belong in runtime settings instead.
    A matching fingerprint is only one reuse check; source content, tenant,
    and the completeness of saved outputs must also be checked.
    """

    extractor: ComponentSpec
    chunker: ComponentSpec
    embedder: ComponentSpec

    SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        """Require a component description for each stage before hashing it."""
        for name in ("extractor", "chunker", "embedder"):
            if not isinstance(getattr(self, name), ComponentSpec):
                raise ValueError(f"{name} must be a ComponentSpec")

    def to_json(self) -> str:
        """Serialize all settings with sorted keys and consistent spacing.

        This makes JSON key order and whitespace irrelevant to the hash.
        Numeric types stay distinct: 1 and 1.0 are not normalized together.
        """
        return json.dumps(
            {
                "schema_version": self.SCHEMA_VERSION,
                "extractor": self.extractor.to_dict(),
                "chunker": self.chunker.to_dict(),
                "embedder": self.embedder.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def fingerprint(self) -> str:
        """Hash the serialized settings so we can compare processing recipes.

        This hashes configuration, not the file or vectors. It cannot detect
        a provider changing a model behind an unchanged name; use pinned
        revisions where available and change versions when behavior changes.
        """
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
