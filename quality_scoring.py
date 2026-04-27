from query_understanding import STOPWORDS, tokenize

DEFAULT_LENGTH_THRESHOLDS = {
    "short_max": 20,
    "medium_max": 52,
}

DEFAULT_TERM_THRESHOLDS = {
    "sparse_max": 10,
    "moderate_max": 28,
}

FLAG_PENALTIES = {
    "empty": 0.16,
    "no_long_description": 0.0,
    "keyword_only": 0.07,
    "short_description": 0.025,
    "too_short": 0.02,
    "short_context": 0.01,
    "metadata_heavy": 0.03,
    "term_sparse": 0.02,
    "low_information": 0.08,
}

RISKY_SEMANTIC_FLAGS = {
    "empty",
    "keyword_only",
    "short_description",
    "too_short",
    "metadata_heavy",
    "term_sparse",
    "low_information",
}


def dedupe_preserve_order(values):
    seen = set()
    ordered = []
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        ordered.append(clean)
        seen.add(clean)
    return ordered


def estimate_content_term_count(text: str) -> int:
    tokens = {
        token
        for token in tokenize(text or "")
        if len(token) >= 3 and token not in STOPWORDS
    }
    return len(tokens)


def infer_length_bucket(word_count: int, thresholds: dict | None = None) -> str:
    limits = thresholds or DEFAULT_LENGTH_THRESHOLDS
    if word_count <= limits.get("short_max", DEFAULT_LENGTH_THRESHOLDS["short_max"]):
        return "short"
    if word_count <= limits.get("medium_max", DEFAULT_LENGTH_THRESHOLDS["medium_max"]):
        return "medium"
    return "long"


def infer_term_bucket(term_count: int, thresholds: dict | None = None) -> str:
    limits = thresholds or DEFAULT_TERM_THRESHOLDS
    if term_count <= limits.get("sparse_max", DEFAULT_TERM_THRESHOLDS["sparse_max"]):
        return "term_sparse"
    if term_count <= limits.get("moderate_max", DEFAULT_TERM_THRESHOLDS["moderate_max"]):
        return "term_moderate"
    return "term_rich"


def infer_description_style(description: str, keywords: list | None, word_count: int) -> str:
    text = description or ""
    lowered = text.lower()
    colon_count = text.count(":")
    bullet_signals = (
        text.count("- ")
        + lowered.count("keywords:")
        + lowered.count("use cases:")
        + lowered.count("columns:")
    )
    keyword_count = len(keywords or [])
    if word_count <= 40 and (keyword_count >= 3 or colon_count >= 3):
        return "metadata_heavy"
    if bullet_signals >= 2 or colon_count >= 4:
        return "mixed_structured"
    if word_count >= 80:
        return "narrative"
    return "mixed_structured" if keyword_count else "narrative"


def infer_quality_flags(item: dict) -> dict:
    description = item.get("description") or item.get("text") or ""
    keywords = item.get("keywords") or []
    word_count = int(item.get("description_len_words") or len(description.split()))
    term_count = estimate_content_term_count(description)
    raw_flags = dedupe_preserve_order(list(item.get("quality_flags") or []))

    style = infer_description_style(description, keywords, word_count) if description else None
    length_bucket = infer_length_bucket(word_count) if word_count else None
    term_bucket = infer_term_bucket(term_count) if term_count or description else None

    inferred_flags = []
    if not description.strip():
        inferred_flags.append("empty")
    if length_bucket == "short" and not any(flag in raw_flags for flag in ("short_description", "too_short")):
        inferred_flags.append("short_context")
    if style == "metadata_heavy":
        inferred_flags.append("metadata_heavy")
    if term_bucket == "term_sparse":
        inferred_flags.append("term_sparse")
    if ("keyword_only" in raw_flags or style == "metadata_heavy") and length_bucket == "short":
        inferred_flags.append("low_information")
    elif style == "metadata_heavy" and term_bucket == "term_sparse":
        inferred_flags.append("low_information")

    all_flags = dedupe_preserve_order(raw_flags + inferred_flags)
    severity = "info"
    if any(flag in all_flags for flag in ("empty", "keyword_only", "low_information")):
        severity = "risk"
    elif all_flags:
        severity = "warning"

    return {
        "raw_flags": raw_flags,
        "inferred_flags": inferred_flags,
        "all_flags": all_flags,
        "severity": severity,
        "word_count": word_count,
        "term_count": term_count,
        "description_style": style,
        "length_bucket": length_bucket,
        "term_bucket": term_bucket,
    }


def semantic_risk_flags(signal: dict | None) -> list[str]:
    if not signal:
        return []
    return [
        flag
        for flag in dedupe_preserve_order(signal.get("all_flags") or [])
        if flag in RISKY_SEMANTIC_FLAGS
    ]


def should_deemphasize_title(item: dict, signal: dict | None = None) -> bool:
    quality_signal = signal or infer_quality_flags(item)
    risk_flags = semantic_risk_flags(quality_signal)
    if risk_flags:
        return True

    word_count = int(quality_signal.get("word_count") or 0)
    if word_count and word_count <= 12:
        return True
    if (
        word_count and word_count <= 24
        and quality_signal.get("description_style") == "mixed_structured"
        and quality_signal.get("term_bucket") == "term_sparse"
    ):
        return True
    return False


def build_semantic_quality_note(item: dict, signal: dict | None = None) -> str:
    quality_signal = signal or infer_quality_flags(item)
    if not should_deemphasize_title(item, quality_signal):
        return ""

    flags = set(semantic_risk_flags(quality_signal))
    if "empty" in flags:
        reason = "description is empty"
    elif "keyword_only" in flags or "metadata_heavy" in flags:
        reason = "description is mostly keywords or metadata"
    elif "low_information" in flags or "term_sparse" in flags:
        reason = "description has limited semantic detail"
    elif (
        "too_short" in flags
        or "short_description" in flags
        or "short_context" in flags
        or int(quality_signal.get("word_count") or 0) <= 24
    ):
        reason = "description is very short"
    else:
        reason = "description provides weak semantic evidence"
    return f"Limited semantic evidence: {reason}; title terms are de-emphasized."


def compute_quality_adjustment(item: dict) -> dict:
    signal = infer_quality_flags(item)
    penalty = 0.0
    for flag in signal["all_flags"]:
        penalty += FLAG_PENALTIES.get(flag, 0.0)

    bonus = 0.0
    if signal["severity"] == "info":
        if signal["description_style"] == "narrative":
            bonus += 0.008
        if signal["term_bucket"] == "term_rich":
            bonus += 0.006
    if should_deemphasize_title(item, signal):
        penalty += 0.03

    penalty = min(penalty, 0.18)
    confidence = max(0.0, min(1.0, 1.0 - penalty + bonus))

    signal["score_penalty"] = round(penalty, 4)
    signal["score_bonus"] = round(bonus, 4)
    signal["quality_confidence"] = round(confidence, 4)
    return signal
