import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

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


def infer_domains(title: str, description: str, keywords: list[str], metadata: list[str]) -> list[str]:
    combined = normalize_text(" ".join([title, description, *keywords, *metadata]))
    token_set = set(tokenize(combined))
    scores = Counter()
    for domain, payload in DOMAIN_RULES.items():
        score = score_rule_set(combined, token_set, payload)
        if score:
            scores[domain] = score
    return [label for label, _ in scores.most_common(3)]


def infer_use_cases(title: str, description: str, keywords: list[str], metadata: list[str]) -> list[str]:
    combined = normalize_text(" ".join([title, description, *keywords, *metadata]))
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
    return [label for label, _ in hits.most_common(4)]


def infer_modalities(title: str, description: str, keywords: list[str], metadata: list[str]) -> list[str]:
    combined = normalize_text(" ".join([title, description, *keywords, *metadata]))
    token_set = set(tokenize(combined))
    hits = Counter()
    for label, patterns in MODALITY_RULES.items():
        for pattern in patterns:
            normalized_pattern = normalize_text(pattern)
            if " " in normalized_pattern and normalized_pattern in combined:
                hits[label] += 2
            elif normalized_pattern in token_set:
                hits[label] += 1
    return [label for label, _ in hits.most_common(3)]


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
) -> str:
    summary_parts = []
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

    title_part = f"title: {title.strip()}" if title.strip() else ""
    return " | ".join(part for part in [title_part, *summary_parts] if part)


def build_semantic_text(
    title: str,
    description: str,
    keywords: list[str],
    domains: list[str],
    use_cases: list[str],
    modalities: list[str],
    language_hint: str,
    semantic_summary: str,
) -> str:
    parts = [
        title.strip(),
        description.strip(),
        semantic_summary,
        "domain concepts: " + ", ".join(label_to_text(label) for label in domains) if domains else "",
        "task concepts: " + ", ".join(label_to_text(label) for label in use_cases) if use_cases else "",
        "modality concepts: " + ", ".join(label_to_text(label) for label in modalities) if modalities else "",
        "language hint: " + language_hint if language_hint else "",
        "keywords: " + ", ".join(keywords[:12]) if keywords else "",
    ]
    return " ".join(part for part in parts if part)


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
    inferred_domains = infer_domains(title, description, keywords, metadata)
    inferred_use_cases = infer_use_cases(title, description, keywords, metadata)
    inferred_modalities = infer_modalities(title, description, keywords, metadata)
    semantic_summary = build_semantic_summary(
        title,
        description,
        keywords,
        inferred_domains,
        inferred_use_cases,
        inferred_modalities,
        language_hint,
    )

    # The thesis scope uses title + description as the document text.
    text = f"{title}. {description}".strip()
    semantic_text = build_semantic_text(
        title,
        description,
        keywords,
        inferred_domains,
        inferred_use_cases,
        inferred_modalities,
        language_hint,
        semantic_summary,
    )

    return {
        "source": source,
        "ref": ref,
        "title": title,
        "description": description,
        "text": text,
        "semantic_text": semantic_text,
        "semantic_summary": semantic_summary,
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
