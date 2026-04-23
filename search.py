import json
import sys

import faiss
from sentence_transformers import SentenceTransformer

from query_understanding import build_query_plan
from runtime_env import ensure_model_cache_dirs
from search_profiles import DEFAULT_PROFILE_KEY, get_profile, get_profile_paths, prepare_query_text

TOP_K = 5


def configure_output():
    # Prevent Windows console encoding errors when dataset text contains emoji.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_mappings(profile_key: str = DEFAULT_PROFILE_KEY):
    paths = get_profile_paths(profile_key)
    mapping_path = paths["mappings"]
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {mapping_path}\n"
            "Run build_faiss_index.py first."
        )
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_search_engine(profile_key: str = DEFAULT_PROFILE_KEY):
    profile = get_profile(profile_key)
    paths = get_profile_paths(profile.key)
    index_path = paths["index"]

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index file not found: {index_path}\n"
            "Run normalize_merge.py and build_faiss_index.py first."
        )

    ensure_model_cache_dirs()

    print(f"[LOG] Yapay zeka modeli yukleniyor: {profile.model_name}")
    model = SentenceTransformer(profile.model_name)

    print(f"[LOG] Vektor veritabani yukleniyor: {index_path}")
    index = faiss.read_index(str(index_path))

    mappings = load_mappings(profile.key)

    if index.ntotal != len(mappings):
        print(
            "[UYARI] Index ve mapping kayit sayisi farkli: "
            f"index={index.ntotal}, mappings={len(mappings)}"
        )

    return model, index, mappings


def search(
    query: str,
    model,
    index,
    mappings,
    top_k: int = TOP_K,
    profile_key: str = DEFAULT_PROFILE_KEY,
    query_plan: dict | None = None,
):
    plan = query_plan or build_query_plan(query)
    prepared_queries = [
        prepare_query_text(profile_key, text)
        for text in plan.get("semantic_queries") or [query]
        if text.strip()
    ]
    query_vectors = model.encode(prepared_queries, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vectors, min(top_k, index.ntotal))

    fused = {}
    for variant_scores, variant_indices in zip(scores, indices):
        for score, idx in zip(variant_scores, variant_indices):
            if idx == -1:
                continue
            item = mappings.get(str(idx))
            if not item:
                continue
            entry = fused.setdefault(
                str(idx),
                {
                    "item": item,
                    "max_score": float(score),
                    "hits": 0,
                },
            )
            entry["max_score"] = max(entry["max_score"], float(score))
            entry["hits"] += 1

    ranked = []
    for entry in fused.values():
        fused_score = entry["max_score"] + 0.02 * max(entry["hits"] - 1, 0)
        item = dict(entry["item"])
        item["semantic_variant_hits"] = entry["hits"]
        ranked.append((float(fused_score), item))

    ranked.sort(key=lambda row: row[0], reverse=True)
    return ranked[:top_k]


def main():
    configure_output()
    model, index, mappings = load_search_engine()

    print("\n" + "=" * 50)
    print("   SEMANTIK VERI SETI ARAMA MOTORUNA HOS GELDINIZ")
    print("=" * 50)

    while True:
        query = input("\nNe tur bir veri seti ariyorsunuz? (Cikis icin 'q'): ").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        print(f"\n'{query}' icin en alakali sonuclar:")
        print("-" * 50)

        for rank, (score, result) in enumerate(search(query, model, index, mappings), start=1):
            desc = (result.get("text") or "").replace("\n", " ")
            print(f"{rank}. BASLIK: {result.get('title', 'Isimsiz')}")
            print(f"   KAYNAK: {(result.get('source') or 'unknown').upper()}")
            print(f"   SKOR: {score:.4f}")
            print(f"   URL: {result.get('url') or 'N/A'}")
            print(f"   OZET: {desc[:180]}...")
            print("-" * 50)


if __name__ == "__main__":
    main()
