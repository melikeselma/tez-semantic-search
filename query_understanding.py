import re


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
NON_TEXT_RE = re.compile(r"[^a-z0-9\s]", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")
LEADING_PATTERNS = [
    re.compile(r"^(i am|i'm|im)\s+(looking for|searching for)\s+", re.IGNORECASE),
    re.compile(r"^(i need|i want)\s+", re.IGNORECASE),
    re.compile(r"^(can you find|find me|show me)\s+", re.IGNORECASE),
]
PHRASE_PATTERNS = [
    re.compile(r"\bdatasets?\s+(?:about|for|with|containing|that contain|that contains)\s+(.+)", re.IGNORECASE),
    re.compile(r"\b(?:about|for|with)\s+(.+)", re.IGNORECASE),
]
TAIL_PATTERNS = [
    re.compile(r"\bthat can be used for\b", re.IGNORECASE),
    re.compile(r"\bthat include\b", re.IGNORECASE),
    re.compile(r"\bthat includes\b", re.IGNORECASE),
    re.compile(r"\bprepared for\b", re.IGNORECASE),
]
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
    "be",
    "by",
    "can",
    "contains",
    "data",
    "dataset",
    "datasets",
    "detecting",
    "do",
    "find",
    "for",
    "from",
    "i",
    "im",
    "include",
    "includes",
    "including",
    "is",
    "it",
    "like",
    "looking",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "over",
    "prepared",
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
    "want",
    "with",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def unique_preserve_order(values):
    seen = set()
    ordered = []
    for value in values:
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def normalize_text(text: str) -> str:
    clean = NON_TEXT_RE.sub(" ", (text or "").strip().lower())
    return MULTISPACE_RE.sub(" ", clean).strip()


def strip_leading_intent(text: str) -> str:
    clean = normalize_text(text)
    for pattern in LEADING_PATTERNS:
        candidate = pattern.sub("", clean).strip()
        if candidate != clean:
            clean = candidate
            break
    return clean


def extract_focus_terms(text: str) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if len(token) > 2 and token not in STOPWORDS
    ]


def derive_intent_body(text: str) -> str:
    body = strip_leading_intent(text)

    for pattern in PHRASE_PATTERNS:
        match = pattern.search(body)
        if match:
            body = match.group(1).strip()
            break

    for pattern in TAIL_PATTERNS:
        body = pattern.sub(" ", body).strip()

    body = body.replace("datasets", "dataset").replace("dataset", " ").strip()
    body = MULTISPACE_RE.sub(" ", body)
    return body


def build_query_plan(query: str) -> dict:
    normalized = normalize_text(query)
    intent_body = derive_intent_body(query)
    focus_terms = unique_preserve_order(extract_focus_terms(intent_body or normalized))
    lexical_query = " ".join(focus_terms) if focus_terms else normalized

    semantic_queries = unique_preserve_order(
        [
            query.strip(),
            intent_body,
            f"dataset for {lexical_query}" if lexical_query else "",
        ]
    )

    concise_intent = lexical_query or intent_body or normalized
    return {
        "original_query": query.strip(),
        "normalized_query": normalized,
        "intent_body": intent_body,
        "focus_terms": focus_terms,
        "lexical_query": lexical_query,
        "concise_intent": concise_intent,
        "semantic_queries": semantic_queries,
        "is_sentence_query": len(tokenize(query)) >= 6,
    }
