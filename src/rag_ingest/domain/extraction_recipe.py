"""Validated options of the "document" extractor recipe and its page rules.

Every option here changes extracted output, so all of them are part of the
pipeline fingerprint. The recognizer block records the injected tool's spec
and output-affecting capabilities; changing the tool or model makes a new
build. Runtime limits (budgets, concurrency, rate) are not recipe options.
"""

import json
import math
import unicodedata

DRIVER, VERSION = "document", "1"
MODES = ("text_only", "recognize_unreadable", "recognize_with_images", "recognize_all")
DEFAULTS = {
    "mode": "text_only",
    "unreadable_pages": "keep_empty",
    "min_text_chars": 16,
    "max_garbage_ratio": 0.3,
    "visual_content_bytes": 4096,
    "strict_pdf": False,
    "render_dpi": 200,
    "max_render_pixels": 25_000_000,
    "allow_signed_url": False,
    "min_confidence": None,
    "recognizer": None,
    # False keeps recipes saved before this option behaving exactly as built;
    # new recipes set it to True (see ServiceSettings.extractor_options).
    "normalize_text": False,
}


def _number(value, low, high, integer):
    ok = (type(value) is int) if integer else (type(value) in (int, float))
    return ok and math.isfinite(value) and low <= value <= high


def validate(options):
    """Return complete, checked options; reject unknown keys and bad values."""
    if not isinstance(options, dict) or set(options) - set(DEFAULTS):
        raise ValueError("Unknown document extractor options")
    o = {**DEFAULTS, **options}
    if o["mode"] not in MODES:
        raise ValueError("Unknown extraction mode")
    if o["unreadable_pages"] not in ("keep_empty", "fail"):
        raise ValueError("unreadable_pages must be 'keep_empty' or 'fail'")
    checks = (
        ("min_text_chars", 0, 10_000, True),
        ("max_garbage_ratio", 0, 1, False),
        ("visual_content_bytes", 0, 10**9, True),
        ("render_dpi", 72, 600, True),
        ("max_render_pixels", 100_000, 200_000_000, True),
    )
    for name, low, high, integer in checks:
        if not _number(o[name], low, high, integer):
            raise ValueError(f"Invalid {name}")
    for name in ("strict_pdf", "allow_signed_url", "normalize_text"):
        if type(o[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if o["min_confidence"] is not None and not _number(
        o["min_confidence"], 0, 1, False
    ):
        raise ValueError("min_confidence must be between 0 and 1, or null")
    recognizer = o["recognizer"]
    if (o["mode"] == "text_only") != (recognizer is None):
        raise ValueError(
            "Recognition modes need a recognizer; text_only must not have one"
        )
    if recognizer is not None and (
        not isinstance(recognizer, dict)
        or set(recognizer)
        != {"driver", "implementation_version", "options", "unit", "output_format"}
    ):
        raise ValueError("Invalid recognizer block")
    return o


def options_json(options):
    """Canonical options_json for a ComponentSpec."""
    return json.dumps(validate(options), sort_keys=True, separators=(",", ":"))


def recognizer_block(spec, capabilities):
    """The fingerprinted identity of an injected recognizer."""
    return {
        "driver": spec.driver,
        "implementation_version": spec.implementation_version,
        "options": json.loads(spec.options_json),
        "unit": capabilities.unit,
        "output_format": capabilities.output_format,
    }


def normalize(text):
    """Drop characters that are not text and that storage cannot hold.

    Removes control characters other than tab/newline/carriage return (NUL
    cannot be stored in PostgreSQL text or JSONB) and unpaired surrogates
    (not encodable as UTF-8). All other characters are kept unchanged.
    """
    return "".join(
        ch
        for ch in text
        if ch in "\t\n\r" or unicodedata.category(ch) not in ("Cc", "Cs")
    )


def garbage_ratio(text):
    """Share of visible characters that are replacement, private or control."""
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 0.0
    bad = sum(
        1
        for ch in visible
        if ch == "�" or unicodedata.category(ch) in ("Co", "Cs", "Cn", "Cc")
    )
    return bad / len(visible)


def classify(text, image_count, content_bytes, options, *, is_pdf):
    """Return (method, needs_recognition) for one page under the recipe.

    Usable text: at least min_text_chars visible characters and a garbage
    ratio within limit. Visual content: any image, unknown facts (-1), or a
    content stream larger than visual_content_bytes (e.g. vector-drawn text).
    """
    if not is_pdf:
        return ("text" if text.strip() else "blank"), False
    usable = (
        len(text.strip()) >= options["min_text_chars"]
        and garbage_ratio(text) <= options["max_garbage_ratio"]
    )
    visual = (
        image_count != 0
        or content_bytes < 0
        or content_bytes > options["visual_content_bytes"]
    )
    mode = options["mode"]
    needs = (
        mode == "recognize_all"
        or (
            mode in ("recognize_unreadable", "recognize_with_images")
            and not usable
            and visual
        )
        or (mode == "recognize_with_images" and image_count != 0)
    )
    if needs:
        return "text", True  # method becomes "recognized" once recognized
    if usable or (text.strip() and not visual):
        return "text", False
    return ("unrecognized" if visual else ("text" if text.strip() else "blank")), False
