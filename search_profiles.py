from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE_KEY = "minilm"


@dataclass(frozen=True)
class SearchProfile:
    key: str
    label: str
    model_name: str
    description: str
    query_prefix: str = ""
    document_prefix: str = ""
    legacy_layout: bool = False


SEARCH_PROFILES = {
    "minilm": SearchProfile(
        key="minilm",
        label="MiniLM (EN baseline)",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        description="Tezin ana Ingilizce semantic search profili.",
        legacy_layout=True,
    ),
    "e5_base": SearchProfile(
        key="e5_base",
        label="E5 Base (EN retrieval)",
        model_name="intfloat/e5-base-v2",
        description="Retrieval-focused English semantic search profile for dataset discovery.",
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
    "minilm_ft": SearchProfile(
        key="minilm_ft",
        label="MiniLM (domain fine-tuned)",
        model_name=str(ROOT / "models" / "retriever" / "minilm-domain-ft" / "final"),
        description="Hard-negative ve pair egitimi ile alan uyarlamali semantic profil.",
    ),
    "multilingual": SearchProfile(
        key="multilingual",
        label="Multilingual (EN+TR alt kume)",
        model_name="intfloat/multilingual-e5-small",
        description="Ara rapordaki EN+TR alt kume senaryosu icin cok dilli semantic profil.",
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
}


def get_profile(profile_key: str | None) -> SearchProfile:
    key = (profile_key or DEFAULT_PROFILE_KEY).strip().lower()
    aliases = {
        "default": DEFAULT_PROFILE_KEY,
        "en": "minilm",
        "english": "minilm",
        "english_e5": "minilm",
        "e5_en": "minilm",
        "e5_small_v2": "minilm",
        "mini": "minilm",
        "minilm_en": "minilm",
        "e5_base_v2": "e5_base",
        "e5_base_en": "e5_base",
        "e5base": "e5_base",
        "minilm_domain_ft": "minilm_ft",
        "finetuned": "minilm_ft",
        "ft": "minilm_ft",
        "tr": "multilingual",
        "turkish": "multilingual",
        "multilingual_e5": "multilingual",
        "e5": "multilingual",
    }
    key = aliases.get(key, key)
    if key not in SEARCH_PROFILES:
        known = ", ".join(sorted(SEARCH_PROFILES))
        raise ValueError(f"Unknown search profile: {profile_key}. Known profiles: {known}")
    return SEARCH_PROFILES[key]


def list_profiles() -> list[SearchProfile]:
    return [SEARCH_PROFILES[key] for key in ("minilm", "e5_base", "minilm_ft", "multilingual")]


def get_profile_paths(profile_key: str | None) -> dict[str, Path]:
    profile = get_profile(profile_key)

    if profile.legacy_layout:
        index_dir = ROOT / "data" / "index"
        embedding_dir = ROOT / "data" / "embeddings"
    else:
        index_dir = ROOT / "data" / "index" / profile.key
        embedding_dir = ROOT / "data" / "embeddings" / profile.key

    return {
        "index_dir": index_dir,
        "embedding_dir": embedding_dir,
        "index": index_dir / "faiss.index",
        "mappings": index_dir / "mappings.json",
        "index_metadata": index_dir / "index_metadata.json",
        "legacy_meta": embedding_dir / "meta.jsonl",
    }


def prepare_document_text(profile_key: str | None, text: str) -> str:
    profile = get_profile(profile_key)
    clean_text = (text or "").strip()
    return f"{profile.document_prefix}{clean_text}" if profile.document_prefix else clean_text


def prepare_query_text(profile_key: str | None, text: str) -> str:
    profile = get_profile(profile_key)
    clean_text = (text or "").strip()
    return f"{profile.query_prefix}{clean_text}" if profile.query_prefix else clean_text
