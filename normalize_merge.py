import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from quality_scoring import (
    build_semantic_quality_note,
    infer_quality_flags,
    should_deemphasize_title,
)
from query_understanding import normalize_text, tokenize

ROOT = Path(__file__).resolve().parent

HF_IN = ROOT / "data" / "processed" / "hf_clean.jsonl"
KG_IN = ROOT / "data" / "processed" / "kaggle_clean.jsonl"
OUT = ROOT / "data" / "processed" / "descriptions_clean.jsonl"


TEXT_FIELDS = (
    "clean_description_basic",
    "clean_description",
    "description",
    "text",
    "raw_description",
    "body",
)

DOMAIN_RULES = {
    "weather_climate": {
        "phrases": ["weather data", "rainfall data", "climate data", "meteorology", "severe weather"],
        "tokens": {
            "weather",
            "climate",
            "rainfall",
            "rain",
            "precipitation",
            "meteorology",
            "storm",
            "temperature",
            "forecast",
            "drought",
        },
    },
    "hydrology_water": {
        "phrases": ["water level", "river flow", "flood data", "runoff dataset"],
        "tokens": {
            "flood",
            "runoff",
            "river",
            "hydrology",
            "water",
            "streamflow",
            "discharge",
            "watershed",
            "groundwater",
        },
    },
    "ocean_marine": {
        "phrases": ["sea surface", "ocean buoy", "marine ecosystem"],
        "tokens": {
            "ocean",
            "marine",
            "aquatic",
            "sea",
            "algae",
            "fish",
            "fishery",
            "buoy",
            "coral",
            "coastal",
        },
    },
    "earth_observation": {
        "phrases": ["satellite imagery", "remote sensing", "earth observation"],
        "tokens": {
            "satellite",
            "remote",
            "sensing",
            "geospatial",
            "gis",
            "earth",
            "landcover",
            "raster",
            "imagery",
        },
    },
    "computer_vision": {
        "phrases": ["image classification", "object detection", "segmentation dataset"],
        "tokens": {
            "image",
            "vision",
            "object",
            "detection",
            "segmentation",
            "ocr",
            "video",
            "classification",
            "bounding",
            "pixels",
        },
    },
    "nlp_text": {
        "phrases": ["text generation", "question answering", "language model"],
        "tokens": {
            "text",
            "language",
            "conversation",
            "dialogue",
            "translation",
            "summarization",
            "qa",
            "question",
            "answering",
            "corpus",
        },
    },
    "audio_speech": {
        "phrases": ["speech recognition", "audio dataset", "acoustic scene"],
        "tokens": {
            "audio",
            "speech",
            "acoustic",
            "voice",
            "speaker",
            "asr",
            "music",
        },
    },
    "health_medical": {
        "phrases": ["medical imaging", "clinical records", "disease prediction"],
        "tokens": {
            "health",
            "medical",
            "clinical",
            "disease",
            "patient",
            "hospital",
            "diagnosis",
            "biomedical",
        },
    },
    "finance_markets": {
        "phrases": ["stock market", "financial time series", "price history"],
        "tokens": {
            "stock",
            "finance",
            "financial",
            "market",
            "price",
            "trading",
            "economy",
            "returns",
        },
    },
}

USE_CASE_RULES = {
    "classification": {"classification", "classifier", "multiclass", "labeling"},
    "detection": {"detection", "detect", "anomaly", "fraud", "wildfire"},
    "forecasting": {"forecast", "forecasting", "prediction", "predict", "time series"},
    "segmentation": {"segmentation", "mask", "pixel", "instance"},
    "retrieval_search": {"retrieval", "search", "ranking", "similarity"},
    "recommendation": {"recommendation", "recommender", "ranking"},
    "ocr_document_ai": {"ocr", "document", "markdown", "pdf"},
    "qa_generation": {"question answering", "qa", "instruction", "generation", "chat"},
    "monitoring_analysis": {"monitoring", "analysis", "tracking", "benchmark"},
}

MODALITY_RULES = {
    "text": {"text", "corpus", "conversation", "document"},
    "tabular": {"csv", "table", "tabular", "spreadsheet"},
    "image": {"image", "images", "vision", "xray", "x-ray", "photo", "satellite"},
    "audio": {"audio", "speech", "voice", "acoustic", "music"},
    "video": {"video", "videos"},
    "time_series": {"time", "timeseries", "temporal", "forecast", "historical"},
    "geospatial": {"geospatial", "gis", "remote", "satellite", "raster", "spatial"},
    "multimodal": {"multimodal", "vision-language", "visionlanguage"},
}

LANGUAGE_RE = re.compile(r"^language:(.+)$", re.IGNORECASE)
TASK_RE = re.compile(r"^task_categories:(.+)$", re.IGNORECASE)
MODALITY_RE = re.compile(r"^modality:(.+)$", re.IGNORECASE)


def unique_preserve_order(values) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        ordered.append(clean)
        seen.add(clean)
    return ordered


def iter_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        print(f"[WARN] Missing input: {path}")
        return

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping invalid JSON in {path.name}:{line_no}: {exc}")


def pick_text(rec: Dict) -> Optional[str]:
    for field in TEXT_FIELDS:
        value = rec.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def pick_ref(rec: Dict, source: str) -> Optional[str]:
    for field in ("source_id", "ref", "id", "name"):
        value = rec.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    title = rec.get("title")
    if isinstance(title, str) and title.strip():
        return f"{source}:{title.strip()}"
    return None


def pick_url(rec: Dict, source: str, ref: Optional[str]) -> Optional[str]:
    url = rec.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip()
    if source == "kaggle" and ref:
        return f"https://www.kaggle.com/datasets/{ref}"
    if source == "huggingface" and ref:
        return f"https://huggingface.co/datasets/{ref}"
    return None


def pick_keywords(rec: Dict) -> list[str]:
    for field in ("keywords", "tags"):
        value = rec.get(field)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [part.strip() for part in value.split(",") if part.strip()]
    return []


def pick_language(rec: Dict) -> str:
    raw = str(rec.get("language") or "").strip().lower()
    if raw:
        return raw

    for tag in pick_keywords(rec):
        match = LANGUAGE_RE.match(tag)
        if match:
            return match.group(1).strip().lower()
    return ""


def metadata_terms(rec: Dict) -> list[str]:
    terms = []
    for tag in pick_keywords(rec):
        lowered = str(tag or "").strip().lower()
        if not lowered:
            continue
        for pattern in (LANGUAGE_RE, TASK_RE, MODALITY_RE):
            match = pattern.match(lowered)
            if match:
                terms.extend(part.strip() for part in match.group(1).split(",") if part.strip())
                break
        else:
            terms.append(lowered)
    return unique_preserve_order(terms)


def pick_license(rec: Dict) -> str:
    value = rec.get("license")
    if isinstance(value, str) and value.strip():
        return value.strip()
    for tag in pick_keywords(rec):
        if tag.startswith("license:"):
            return tag.split(":", 1)[1].strip()
    return ""


def score_rule_set(text: str, tokens: set[str], rule_payload: Dict) -> int:
    score = 0
    for phrase in rule_payload.get("phrases", []):
        if phrase in text:
            score += 2
    for token in rule_payload.get("tokens", set()):
        if token in tokens:
            score += 1
    return score


def infer_domains(
    title: str,
    description: str,
    keywords: list[str],
    metadata: list[str],
    include_title: bool = True,
    min_score: int = 1,
) -> list[str]:
    parts = [description, *keywords, *metadata]
    if include_title:
        parts.insert(0, title)
    combined = normalize_text(" ".join(part for part in parts if part))
    token_set = set(tokenize(combined))
    scores = Counter()
    for domain, payload in DOMAIN_RULES.items():
        score = score_rule_set(combined, token_set, payload)
        if score >= min_score:
            scores[domain] = score
    return [label for label, _ in scores.most_common(3)]


def infer_use_cases(
    title: str,
    description: str,
    keywords: list[str],
    metadata: list[str],
    include_title: bool = True,
    min_score: int = 1,
) -> list[str]:
    parts = [description, *keywords, *metadata]
    if include_title:
        parts.insert(0, title)
    combined = normalize_text(" ".join(part for part in parts if part))
    hits = Counter()
    for label, patterns in USE_CASE_RULES.items():
        for pattern in patterns:
            normalized_pattern = normalize_text(pattern)
            if " " in normalized_pattern and normalized_pattern in combined:
                hits[label] += 2
            else:
                for token in tokenize(normalized_pattern):
                    if token in combined.split():
                        hits[label] += 1
    return [label for label, score in hits.most_common(4) if score >= min_score]


def infer_modalities(
    title: str,
    description: str,
    keywords: list[str],
    metadata: list[str],
    include_title: bool = True,
    min_score: int = 1,
) -> list[str]:
    parts = [description, *keywords, *metadata]
    if include_title:
        parts.insert(0, title)
    combined = normalize_text(" ".join(part for part in parts if part))
    token_set = set(tokenize(combined))
    hits = Counter()
    for label, patterns in MODALITY_RULES.items():
        for pattern in patterns:
            normalized_pattern = normalize_text(pattern)
            if " " in normalized_pattern and normalized_pattern in combined:
                hits[label] += 2
            elif normalized_pattern in token_set:
                hits[label] += 1
    return [label for label, score in hits.most_common(3) if score >= min_score]


def label_to_text(label: str) -> str:
    return label.replace("_", " ")


def build_semantic_summary(
    title: str,
    description: str,
    keywords: list[str],
    domains: list[str],
    use_cases: list[str],
    modalities: list[str],
    language_hint: str,
    semantic_quality_note: str = "",
    include_title: bool = True,
) -> str:
    summary_parts = []
    if semantic_quality_note:
        summary_parts.append("quality: " + semantic_quality_note)
    if domains:
        summary_parts.append("domains: " + ", ".join(label_to_text(label) for label in domains))
    if use_cases:
        summary_parts.append("use cases: " + ", ".join(label_to_text(label) for label in use_cases))
    if modalities:
        summary_parts.append("modalities: " + ", ".join(label_to_text(label) for label in modalities))
    if language_hint:
        summary_parts.append(f"language: {language_hint}")
    topical_keywords = [kw for kw in keywords if ":" not in kw][:8]
    if topical_keywords:
        summary_parts.append("keywords: " + ", ".join(topical_keywords))

    short_desc = " ".join(description.split()[:60]).strip()
    if short_desc:
        summary_parts.append("summary: " + short_desc)

    title_part = f"title: {title.strip()}" if include_title and title.strip() else ""
    return " | ".join(part for part in [title_part, *summary_parts] if part)


NON_TOPICAL_METADATA_PREFIXES = (
    "task_categories:",
    "language:",
    "license:",
    "size_categories:",
    "format:",
    "modality:",
    "library:",
    "region:",
    "arxiv:",
    "doi:",
)

LOW_SIGNAL_TOPIC_TERMS = {
    "en",
    "tr",
    "text",
    "image",
    "audio",
    "video",
    "tabular",
    "dataset",
    "datasets",
    "document",
}


def truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words).strip()
    return " ".join(words[:max_words]).strip()


def extract_summary_fragment(semantic_summary: str) -> str:
    text = str(semantic_summary or "").strip()
    if not text:
        return ""
    marker = "summary:"
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].strip(" |")


def prefer_topic_terms(values: list[str], max_items: int) -> list[str]:
    topical = []
    fallback = []
    for raw_value in values or []:
        value = str(raw_value or "").strip()
        if not value:
            continue
        lowered = value.lower()
        if lowered in LOW_SIGNAL_TOPIC_TERMS:
            continue
        if ":" in value:
            if lowered.startswith(NON_TOPICAL_METADATA_PREFIXES):
                continue
            fallback.append(value)
        else:
            topical.append(value)
    return unique_preserve_order(topical + fallback)[:max_items]


def is_low_information_description(description: str, quality_flags: list[str]) -> bool:
    text = (description or "").strip()
    if not text:
        return True

    lowered = text.lower()
    word_count = len(text.split())
    colon_count = text.count(":")
    bullet_like = text.count("- ") + lowered.count("keywords:") + lowered.count("tags:")
    structured_markers = sum(
        lowered.count(marker)
        for marker in ("task_categories:", "config_name:", "data_files:", "split:", "path:")
    )

    if any(
        flag in set(quality_flags or [])
        for flag in (
            "empty",
            "keyword_only",
            "short_description",
            "too_short",
            "metadata_heavy",
            "low_information",
        )
    ):
        return True
    if word_count <= 20 and (lowered.startswith("keywords:") or colon_count >= 3):
        return True
    if lowered.startswith("---") and word_count <= 60:
        return True
    if structured_markers >= 2:
        return True
    if word_count <= 60 and (colon_count >= 4 or bullet_like >= 3):
        return True
    if word_count <= 40 and bullet_like >= 4:
        return True
    return False


def build_semantic_text(row: Dict) -> str:
    title = str(row.get("title") or "").strip()
    description = str(row.get("description") or row.get("text") or "").strip()
    keywords = list(row.get("keywords") or [])
    metadata = list(row.get("metadata_terms") or [])
    domains = [label_to_text(label) for label in (row.get("inferred_domains") or [])]
    use_cases = [label_to_text(label) for label in (row.get("inferred_use_cases") or [])]
    modalities = [label_to_text(label) for label in (row.get("inferred_modalities") or [])]
    language_hint = str(row.get("language_hint") or "").strip()
    quality_flags = [str(flag).strip() for flag in (row.get("quality_flags") or []) if str(flag).strip()]
    semantic_quality_note = str(row.get("semantic_quality_note") or "").strip()
    title_deemphasized = bool(semantic_quality_note)

    low_information = is_low_information_description(description, quality_flags)
    description_excerpt = truncate_words(description, 90)
    summary_excerpt = truncate_words(extract_summary_fragment(row.get("semantic_summary") or ""), 80)
    topic_terms = prefer_topic_terms(keywords + metadata, 10)
    main_topics = unique_preserve_order(domains + topic_terms[:6])
    keyword_terms = prefer_topic_terms(keywords, 10)
    metadata_terms_selected = prefer_topic_terms(metadata, 8)
    supporting_terms = [term for term in metadata_terms_selected if term not in set(keyword_terms)]

    lines = []
    # Always emit the title so the encoder can see the topic word for short or
    # keyword-only datasets (e.g. "mushrooms" / "Mushroom"). Title-driven false
    # positives are still suppressed at rerank time via title de-emphasis; we
    # were previously dropping the only meaningful semantic anchor.
    if title:
        lines.append(f"Dataset title: {title}")
    if description_excerpt and not low_information:
        lines.append(f"Dataset description: {description_excerpt}")
    elif summary_excerpt and not is_low_information_description(summary_excerpt, quality_flags):
        lines.append(f"Dataset description: {summary_excerpt}")

    if main_topics:
        lines.append("Main topics: " + ", ".join(main_topics))
    if use_cases:
        lines.append("Task/use case: " + ", ".join(use_cases))

    modality_parts = []
    if modalities:
        modality_parts.append(", ".join(modalities))
    if language_hint:
        modality_parts.append(f"language: {language_hint}")
    if modality_parts:
        lines.append("Data modality: " + " | ".join(modality_parts))

    if keyword_terms:
        lines.append("Keywords: " + ", ".join(keyword_terms))
    elif metadata_terms_selected:
        lines.append("Keywords: " + ", ".join(metadata_terms_selected))

    if low_information and supporting_terms:
        lines.append("Supporting metadata: " + ", ".join(supporting_terms[:6]))
    if semantic_quality_note:
        lines.append("Semantic quality: " + semantic_quality_note)
    if quality_flags and low_information:
        lines.append("Description quality: " + ", ".join(unique_preserve_order(quality_flags)))

    return "\n".join(line for line in lines if line.strip())


def normalize(rec: Dict, source_hint: str) -> Optional[Dict]:
    source = (rec.get("source") or source_hint or "unknown").strip().lower()
    title = (rec.get("title") or "Untitled Dataset").strip()
    description = pick_text(rec)
    if not description:
        return None

    ref = pick_ref(rec, source)
    url = pick_url(rec, source, ref)
    keywords = pick_keywords(rec)
    language_hint = pick_language(rec)
    metadata = metadata_terms(rec)
    quality_input = {
        "title": title,
        "description": description,
        "keywords": keywords,
        "quality_flags": rec.get("quality_flags") or [],
        "description_len_words": rec.get("description_len_words") or len(description.split()),
    }
    quality_signal = infer_quality_flags(quality_input)
    title_deemphasized = should_deemphasize_title(quality_input, quality_signal)
    semantic_quality_note = build_semantic_quality_note(quality_input, quality_signal)
    evidence_threshold = 2 if title_deemphasized else 1
    inferred_domains = infer_domains(
        title,
        description,
        keywords,
        metadata,
        include_title=not title_deemphasized,
        min_score=evidence_threshold,
    )
    inferred_use_cases = infer_use_cases(
        title,
        description,
        keywords,
        metadata,
        include_title=not title_deemphasized,
        min_score=evidence_threshold,
    )
    inferred_modalities = infer_modalities(
        title,
        description,
        keywords,
        metadata,
        include_title=not title_deemphasized,
        min_score=evidence_threshold,
    )
    semantic_summary = build_semantic_summary(
        title,
        description,
        keywords,
        inferred_domains,
        inferred_use_cases,
        inferred_modalities,
        language_hint,
        semantic_quality_note=semantic_quality_note,
        include_title=not title_deemphasized,
    )

    # The thesis scope uses title + description as the document text.
    text = f"{title}. {description}".strip()
    row = {
        "source": source,
        "ref": ref,
        "title": title,
        "description": description,
        "text": text,
        "semantic_summary": semantic_summary,
        "semantic_quality_note": semantic_quality_note,
        "url": url,
        "keywords": keywords,
        "language_hint": language_hint,
        "metadata_terms": metadata,
        "inferred_domains": inferred_domains,
        "inferred_use_cases": inferred_use_cases,
        "inferred_modalities": inferred_modalities,
        "license": pick_license(rec),
        "quality_flags": rec.get("quality_flags") or [],
        "description_len_chars": rec.get("description_len_chars") or len(description),
        "description_len_words": rec.get("description_len_words") or len(description.split()),
    }
    # A structured field order helps embeddings focus on description-driven signals
    # like topic, task, and modality instead of over-indexing on short titles.
    row["semantic_text"] = build_semantic_text(row)
    return row


def write_source(path: Path, source_hint: str, out, seen: set) -> Tuple[int, int, int]:
    total = written = skipped = 0

    for rec in iter_jsonl(path) or []:
        total += 1
        norm = normalize(rec, source_hint)
        if not norm:
            skipped += 1
            continue

        key = (norm["source"], norm.get("ref") or norm["title"])
        if key in seen:
            skipped += 1
            continue

        seen.add(key)
        out.write(json.dumps(norm, ensure_ascii=False) + "\n")
        written += 1

    return total, written, skipped


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    totals = {}

    with OUT.open("w", encoding="utf-8", newline="\n") as out:
        for path, source in ((HF_IN, "huggingface"), (KG_IN, "kaggle")):
            total, written, skipped = write_source(path, source, out, seen)
            totals[source] = {"input": total, "written": written, "skipped": skipped}

    total_out = sum(item["written"] for item in totals.values())
    print("[OK] Normalized merge complete")
    for source, counts in totals.items():
        print(
            f"  {source}: input={counts['input']} "
            f"written={counts['written']} skipped={counts['skipped']}"
        )
    print(f"  total_written={total_out}")
    print(f"  output={OUT}")


if __name__ == "__main__":
    main()
