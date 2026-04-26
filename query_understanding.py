import re
import unicodedata
from collections import Counter


TOKEN_RE = re.compile(r"[a-z0-9]+")
NON_TEXT_RE = re.compile(r"[^a-z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")
LEADING_PATTERNS = [
    re.compile(r"^(i am|i m|im)\s+(looking for|searching for)\s+", re.IGNORECASE),
    re.compile(r"^(i need|i want)\s+", re.IGNORECASE),
    re.compile(r"^(can you find|find me|show me)\s+", re.IGNORECASE),
    re.compile(r"^(bana|ben)\s+", re.IGNORECASE),
]
PHRASE_PATTERNS = [
    re.compile(
        r"\bdatasets?\s+(?:about|for|with|containing|that contain|that contains)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:about|for|with)\s+(.+)", re.IGNORECASE),
    re.compile(r"\bveri(?:\s+seti|leri)?\s+(?:hakkinda|icin|ile ilgili)\s+(.+)", re.IGNORECASE),
    re.compile(r"\b(?:hakkinda|icin|ile ilgili)\s+(.+)", re.IGNORECASE),
]
TAIL_PATTERNS = [
    re.compile(r"\bthat can be used for\b", re.IGNORECASE),
    re.compile(r"\bthat include\b", re.IGNORECASE),
    re.compile(r"\bthat includes\b", re.IGNORECASE),
    re.compile(r"\bprepared for\b", re.IGNORECASE),
    re.compile(r"\bkullanilabilecek\b", re.IGNORECASE),
    re.compile(r"\biceren\b", re.IGNORECASE),
    re.compile(r"\bile ilgili\b", re.IGNORECASE),
]
DATASET_TERMS_RE = re.compile(
    r"\b(datasets?|dataseti|datasetler|data|records?|veri|verisi|veriler|veri seti|veri setleri)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a",
    "about",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "around",
    "as",
    "at",
    "bana",
    "be",
    "ben",
    "bir",
    "bu",
    "by",
    "can",
    "contains",
    "data",
    "dataset",
    "dataseti",
    "datasets",
    "detecting",
    "do",
    "find",
    "for",
    "from",
    "gibi",
    "goster",
    "hakkinda",
    "i",
    "icin",
    "ile",
    "ilgili",
    "im",
    "include",
    "includes",
    "including",
    "is",
    "it",
    "kadar",
    "kullanilabilecek",
    "like",
    "looking",
    "me",
    "my",
    "need",
    "ne",
    "of",
    "olan",
    "on",
    "or",
    "over",
    "prepared",
    "record",
    "records",
    "related",
    "search",
    "searching",
    "show",
    "something",
    "style",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "to",
    "used",
    "ve",
    "veri",
    "veriler",
    "verisi",
    "want",
    "with",
}
TURKISH_TRANSLATION = str.maketrans(
    {
        "\u00c7": "c",
        "\u00e7": "c",
        "\u011e": "g",
        "\u011f": "g",
        "\u0130": "i",
        "\u0131": "i",
        "\u00d6": "o",
        "\u00f6": "o",
        "\u015e": "s",
        "\u015f": "s",
        "\u00dc": "u",
        "\u00fc": "u",
    }
)
TURKISH_CHARACTERS = {chr(codepoint) for codepoint in TURKISH_TRANSLATION.keys()}
TURKISH_MARKER_TOKENS = {
    "arama",
    "analiz",
    "analizi",
    "afet",
    "balik",
    "bana",
    "deniz",
    "deprem",
    "duygu",
    "firtina",
    "goster",
    "hava",
    "icin",
    "iklim",
    "kuraklik",
    "okyanus",
    "sonrasi",
    "sel",
    "su",
    "tahmin",
    "ve",
    "veri",
    "verisi",
    "yardim",
    "yagmur",
    "yagmurlu",
    "yorum",
    "yosun",
}
SHORT_KEEP_TOKENS = {
    "ai",
    "cv",
    "nlp",
    "qa",
    "su",
    "tr",
    "en",
}
PHRASE_RULES = {
    "rainy weather": {
        "domain": "weather_climate",
        "terms": ["weather", "rainfall", "precipitation", "meteorology", "climate"],
    },
    "storm records": {
        "domain": "weather_climate",
        "terms": ["storm", "severe weather", "meteorology", "forecast", "weather archive"],
    },
    "storm forecast": {
        "domain": "weather_climate",
        "terms": ["storm", "forecast", "meteorology", "weather archive", "severe weather"],
    },
    "flood runoff": {
        "domain": "hydrology",
        "terms": ["flood", "runoff", "discharge", "river", "water level", "hydrology"],
    },
    "climate history": {
        "domain": "climate_variability",
        "terms": ["climate", "historical weather", "temperature", "precipitation", "variability"],
    },
    "sea and ocean": {
        "domain": "ocean_climate",
        "terms": ["ocean", "marine", "aquatic", "buoy", "sea surface", "climate"],
    },
    "ocean conditions": {
        "domain": "ocean_climate",
        "terms": ["ocean", "marine", "buoy", "sea surface", "climate"],
    },
    "yagmurlu hava": {
        "domain": "weather_climate",
        "terms": ["weather", "rainfall", "precipitation", "meteorology", "climate"],
    },
    "hava tahmin": {
        "domain": "weather_climate",
        "terms": ["forecast", "weather", "meteorology", "climate", "precipitation"],
    },
    "firtina": {
        "domain": "weather_climate",
        "terms": ["storm", "severe weather", "forecast", "meteorology"],
    },
    "sel ve akis": {
        "domain": "hydrology",
        "terms": ["flood", "runoff", "discharge", "river", "water level", "hydrology"],
    },
    "kuraklik ve iklim": {
        "domain": "climate_variability",
        "terms": ["drought", "climate", "temperature", "precipitation", "variability"],
    },
    "deniz ve okyanus": {
        "domain": "ocean_climate",
        "terms": ["ocean", "marine", "aquatic", "buoy", "sea surface", "climate"],
    },
    "deniz canlilari": {
        "domain": "ocean_climate",
        "terms": ["marine", "ocean", "aquatic", "ecology", "biodiversity", "fishery"],
    },
    "su balik yosun": {
        "domain": "ocean_climate",
        "terms": ["marine", "ocean", "aquatic", "water", "ecology", "algae", "fishery"],
    },
    "deprem sonrasi yardim": {
        "domain": "earth_science",
        "terms": ["earthquake", "seismic", "disaster response", "damage assessment", "relief", "crisis mapping"],
    },
    "deprem yardim": {
        "domain": "earth_science",
        "terms": ["earthquake", "seismic", "disaster response", "damage assessment", "relief"],
    },
    "duygu analizi": {
        "domain": "reviews_sentiment",
        "terms": ["sentiment", "emotion", "reviews", "opinions", "ratings", "classification"],
    },
    "yorum analizi": {
        "domain": "reviews_sentiment",
        "terms": ["reviews", "comments", "opinions", "sentiment", "classification"],
    },
}
TOKEN_RULES = {
    "air": {"domain": "environment", "terms": ["air quality", "atmosphere", "environment"]},
    "audio": {"domain": "audio_speech", "terms": ["audio", "speech", "acoustic"]},
    "balik": {"domain": "ocean_climate", "terms": ["fishery", "aquatic", "marine"]},
    "climate": {"domain": "weather_climate", "terms": ["climate", "weather", "meteorology"]},
    "deepfake": {"domain": "media_authenticity", "terms": ["deepfake", "forgery detection", "audio fake"]},
    "deniz": {"domain": "ocean_climate", "terms": ["marine", "ocean", "sea surface", "aquatic"]},
    "deprem": {"domain": "earth_science", "terms": ["earthquake", "seismic", "geology", "disaster response"]},
    "drought": {"domain": "climate_variability", "terms": ["drought", "climate", "precipitation", "temperature"]},
    "duygu": {"domain": "reviews_sentiment", "terms": ["sentiment", "emotion", "affect", "opinion"]},
    "earthquake": {"domain": "earth_science", "terms": ["earthquake", "seismic", "usgs", "geology"]},
    "ecology": {"domain": "ocean_climate", "terms": ["ecology", "ecosystem", "biodiversity"]},
    "firtina": {"domain": "weather_climate", "terms": ["storm", "severe weather", "forecast", "meteorology"]},
    "fishery": {"domain": "ocean_climate", "terms": ["fishery", "marine", "ocean", "aquatic"]},
    "flood": {"domain": "hydrology", "terms": ["flood", "runoff", "river", "water level", "hydrology"]},
    "forecast": {"domain": "weather_climate", "terms": ["forecast", "meteorology", "weather archive"]},
    "hava": {"domain": "weather_climate", "terms": ["weather", "meteorology", "climate"]},
    "health": {"domain": "health", "terms": ["health", "medical", "clinical"]},
    "iklim": {"domain": "weather_climate", "terms": ["climate", "weather", "meteorology"]},
    "kuraklik": {"domain": "climate_variability", "terms": ["drought", "climate", "precipitation", "temperature"]},
    "marine": {"domain": "ocean_climate", "terms": ["marine", "ocean", "aquatic", "sea surface"]},
    "medical": {"domain": "health", "terms": ["medical", "clinical", "health"]},
    "meteorology": {"domain": "weather_climate", "terms": ["meteorology", "weather", "forecast"]},
    "ocean": {"domain": "ocean_climate", "terms": ["ocean", "marine", "buoy", "sea surface", "aquatic"]},
    "okyanus": {"domain": "ocean_climate", "terms": ["ocean", "marine", "buoy", "sea surface", "aquatic"]},
    "opinion": {"domain": "social_discourse", "terms": ["opinion", "discussion", "comments", "public sentiment"]},
    "pollution": {"domain": "environment", "terms": ["pollution", "environment", "air quality"]},
    "precipitation": {"domain": "weather_climate", "terms": ["precipitation", "rainfall", "weather", "climate"]},
    "price": {"domain": "finance", "terms": ["price", "market", "stock", "financial time series"]},
    "rain": {"domain": "weather_climate", "terms": ["rainfall", "precipitation", "weather", "meteorology"]},
    "rainfall": {"domain": "weather_climate", "terms": ["rainfall", "precipitation", "weather", "climate"]},
    "rainy": {"domain": "weather_climate", "terms": ["rainfall", "precipitation", "weather", "meteorology"]},
    "review": {"domain": "reviews_sentiment", "terms": ["reviews", "sentiment", "ratings", "opinions"]},
    "river": {"domain": "hydrology", "terms": ["river", "water level", "runoff", "discharge"]},
    "runoff": {"domain": "hydrology", "terms": ["runoff", "discharge", "flood", "river", "hydrology"]},
    "sea": {"domain": "ocean_climate", "terms": ["sea surface", "ocean", "marine", "aquatic"]},
    "sel": {"domain": "hydrology", "terms": ["flood", "runoff", "river", "water level", "hydrology"]},
    "sentiment": {"domain": "reviews_sentiment", "terms": ["sentiment", "reviews", "ratings", "classification"]},
    "sonrasi": {"domain": "earth_science", "terms": ["post disaster", "response", "recovery"]},
    "speech": {"domain": "audio_speech", "terms": ["speech", "audio", "recognition", "acoustic"]},
    "stock": {"domain": "finance", "terms": ["stock", "market", "financial", "price history"]},
    "storm": {"domain": "weather_climate", "terms": ["storm", "severe weather", "forecast", "meteorology"]},
    "su": {"domain": "hydrology", "terms": ["water", "hydrology", "aquatic"]},
    "tahmin": {"domain": "weather_climate", "terms": ["forecast", "meteorology", "weather archive"]},
    "water": {"domain": "hydrology", "terms": ["water", "river", "runoff", "hydrology"]},
    "weather": {"domain": "weather_climate", "terms": ["weather", "climate", "meteorology", "forecast"]},
    "yardim": {"domain": "earth_science", "terms": ["relief", "aid", "humanitarian", "response", "crisis"]},
    "yagmur": {"domain": "weather_climate", "terms": ["rainfall", "precipitation", "weather", "meteorology"]},
    "yagmurlu": {"domain": "weather_climate", "terms": ["rainfall", "precipitation", "weather", "meteorology"]},
    "yorum": {"domain": "reviews_sentiment", "terms": ["reviews", "comments", "opinions", "sentiment"]},
    "yosun": {"domain": "ocean_climate", "terms": ["algae", "aquatic", "ecology", "marine"]},
}
COMBO_RULES = [
    {
        "tokens": {"su", "balik"},
        "domain": "ocean_climate",
        "terms": ["marine", "ocean", "aquatic", "water", "fishery"],
    },
    {
        "tokens": {"su", "balik", "yosun"},
        "domain": "ocean_climate",
        "terms": ["marine", "ocean", "aquatic", "water", "ecology", "algae", "fishery"],
    },
    {
        "tokens": {"deniz", "canlilari"},
        "domain": "ocean_climate",
        "terms": ["marine", "ocean", "aquatic", "ecology", "biodiversity", "fishery"],
    },
    {
        "tokens": {"firtina", "tahmin"},
        "domain": "weather_climate",
        "terms": ["storm", "forecast", "meteorology", "severe weather"],
    },
    {
        "tokens": {"kuraklik", "iklim"},
        "domain": "climate_variability",
        "terms": ["drought", "climate", "precipitation", "temperature", "variability"],
    },
    {
        "tokens": {"sel", "akis"},
        "domain": "hydrology",
        "terms": ["flood", "runoff", "discharge", "river", "water level", "hydrology"],
    },
    {
        "tokens": {"deprem", "yardim"},
        "domain": "earth_science",
        "terms": ["earthquake", "seismic", "disaster response", "relief", "damage assessment"],
    },
    {
        "tokens": {"deprem", "sonrasi", "yardim"},
        "domain": "earth_science",
        "terms": ["earthquake", "seismic", "disaster response", "relief", "recovery", "crisis mapping"],
    },
    {
        "tokens": {"duygu", "analizi"},
        "domain": "reviews_sentiment",
        "terms": ["sentiment", "emotion", "reviews", "opinions", "classification"],
    },
    {
        "tokens": {"yorum", "analizi"},
        "domain": "reviews_sentiment",
        "terms": ["reviews", "comments", "opinions", "sentiment", "classification"],
    },
]
TURKISH_SUFFIXES = (
    "lerden",
    "lardan",
    "lerde",
    "larda",
    "lerin",
    "larin",
    "leri",
    "lari",
    "deki",
    "daki",
    "sinin",
    "sine",
    "sina",
    "sini",
    "sunu",
    "sunu",
    "siniz",
    "siniz",
    "imiz",
    "iniz",
    "imiz",
    "inde",
    "inda",
    "imde",
    "unda",
    "unde",
    "leriyle",
    "lariyla",
    "yle",
    "yla",
    "le",
    "la",
    "den",
    "dan",
    "ten",
    "tan",
    "dir",
    "tir",
    "si",
    "i",
    "u",
)
ENGLISH_SUFFIXES = ("s", "es", "ing", "ed")


def ascii_fold(text: str) -> str:
    base = ((text or "").strip()).translate(TURKISH_TRANSLATION).lower()
    folded = unicodedata.normalize("NFKD", base)
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    clean = NON_TEXT_RE.sub(" ", ascii_fold(text))
    return MULTISPACE_RE.sub(" ", clean).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def unique_preserve_order(values):
    seen = set()
    ordered = []
    for value in values:
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def token_variants(token: str) -> list[str]:
    variants = [token]

    for suffix in TURKISH_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            variants.append(token[: -len(suffix)])
            break

    for suffix in ENGLISH_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            variants.append(token[: -len(suffix)])
            break

    return unique_preserve_order(variants)


def detect_language(query: str, normalized: str) -> str:
    raw = (query or "").strip()
    if any(ch in raw for ch in TURKISH_CHARACTERS):
        return "tr"

    normalized_tokens = tokenize(normalized)
    marker_hits = sum(1 for token in normalized_tokens if token in TURKISH_MARKER_TOKENS)
    if marker_hits >= 1:
        return "tr"
    return "en"


def strip_leading_intent(text: str) -> str:
    clean = normalize_text(text)
    for pattern in LEADING_PATTERNS:
        candidate = pattern.sub("", clean).strip()
        if candidate != clean:
            clean = candidate
            break
    return clean


def derive_intent_body(text: str) -> str:
    body = strip_leading_intent(text)

    for pattern in PHRASE_PATTERNS:
        match = pattern.search(body)
        if match:
            body = match.group(1).strip()
            break

    for pattern in TAIL_PATTERNS:
        body = pattern.sub(" ", body).strip()

    body = DATASET_TERMS_RE.sub(" ", body).strip()
    body = MULTISPACE_RE.sub(" ", body)
    return body


def extract_focus_terms(text: str) -> list[str]:
    terms = []
    for token in tokenize(text):
        for variant in token_variants(token):
            if (len(variant) <= 2 and variant not in SHORT_KEEP_TOKENS) or variant in STOPWORDS:
                continue
            terms.append(variant)
    return unique_preserve_order(terms)


def collect_concept_matches(normalized_query: str, focus_terms: list[str]) -> tuple[list[str], list[str]]:
    matched_terms = []
    domain_counts = Counter()
    token_set = set(focus_terms)

    for phrase, payload in PHRASE_RULES.items():
        if phrase in normalized_query:
            matched_terms.extend(payload["terms"])
            domain_counts[payload["domain"]] += 2

    for rule in COMBO_RULES:
        if rule["tokens"].issubset(token_set):
            matched_terms.extend(rule["terms"])
            domain_counts[rule["domain"]] += 2

    for token in focus_terms:
        payload = TOKEN_RULES.get(token)
        if not payload:
            continue
        matched_terms.extend(payload["terms"])
        domain_counts[payload["domain"]] += 1

    primary_domains = [domain for domain, _ in domain_counts.most_common(3)]
    concept_terms = [
        concept_token
        for concept_token in unique_preserve_order(
            token
            for concept in matched_terms
            for token in tokenize(concept)
        )
        if concept_token not in STOPWORDS
    ]
    return concept_terms, primary_domains


def build_semantic_variants(
    query: str,
    intent_body: str,
    raw_focus_terms: list[str],
    concept_terms: list[str],
    domains: list[str],
    detected_language: str,
) -> list[dict]:
    raw_focus_query = " ".join(raw_focus_terms[:8])
    concept_query = " ".join(concept_terms[:10])
    domain_query = " ".join(domain.replace("_", " ") for domain in domains[:2])
    has_projection = bool(concept_terms)
    original_weight = 0.85 if detected_language == "tr" and has_projection else 1.0
    concept_weight = 1.45 if detected_language == "tr" else 1.25
    domain_weight = 1.35 if detected_language == "tr" else 1.15
    bridge_weight = 1.25 if detected_language == "tr" and has_projection else 1.05
    variants = []

    def add(text: str, weight: float):
        clean = (text or "").strip()
        if not clean:
            return
        for item in variants:
            if item["text"] == clean:
                item["weight"] = max(item["weight"], weight)
                return
        variants.append({"text": clean, "weight": weight})

    add(query.strip(), original_weight)
    add(intent_body, 1.0)
    add(raw_focus_query, 1.0)
    add(concept_query, concept_weight)
    add(f"{raw_focus_query} {concept_query}".strip(), bridge_weight)
    add(f"dataset for {raw_focus_query}" if raw_focus_query else "", 1.0)
    add(f"dataset about {concept_query}" if concept_query else "", concept_weight)
    add(
        f"{domain_query} dataset {concept_query}".strip() if domain_query or concept_query else "",
        domain_weight,
    )
    return variants


def build_query_plan(query: str) -> dict:
    normalized = normalize_text(query)
    detected_language = detect_language(query, normalized)
    intent_body = derive_intent_body(query)
    raw_focus_terms = unique_preserve_order(extract_focus_terms(intent_body or normalized))
    concept_terms, domains = collect_concept_matches(normalized, raw_focus_terms)
    domain_terms = unique_preserve_order(
        token
        for domain in domains
        for token in tokenize(domain.replace("_", " "))
        if token not in STOPWORDS
    )
    focus_terms = unique_preserve_order(raw_focus_terms + concept_terms)
    lexical_query = " ".join(focus_terms) if focus_terms else normalized
    concise_intent = " ".join((concept_terms or raw_focus_terms)[:8]) or normalized
    semantic_variants = build_semantic_variants(
        query,
        intent_body,
        raw_focus_terms,
        concept_terms,
        domains,
        detected_language,
    )
    semantic_queries = [item["text"] for item in semantic_variants]

    return {
        "original_query": query.strip(),
        "normalized_query": normalized,
        "detected_language": detected_language,
        "intent_body": intent_body,
        "raw_focus_terms": raw_focus_terms,
        "focus_terms": focus_terms,
        "concept_terms": concept_terms,
        "domains": domains,
        "domain_terms": domain_terms,
        "lexical_query": lexical_query,
        "concise_intent": concise_intent,
        "semantic_variants": semantic_variants,
        "semantic_queries": semantic_queries,
        "is_sentence_query": len(tokenize(query)) >= 6,
    }
