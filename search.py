import json
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "data" / "index" / "faiss.index"
MAPPING_PATH = BASE_DIR / "data" / "index" / "mappings.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def configure_output():
    # Prevent Windows console encoding errors when dataset text contains emoji.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_mappings():
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {MAPPING_PATH}\n"
            "Run build_faiss_index.py first."
        )
    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_search_engine():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index file not found: {INDEX_PATH}\n"
            "Run normalize_merge.py and build_faiss_index.py first."
        )

    print("[LOG] Yapay zeka modeli (MiniLM) yükleniyor...")
    model = SentenceTransformer(MODEL_NAME)

    print("[LOG] Vektör veritabanı yükleniyor...")
    index = faiss.read_index(str(INDEX_PATH))

    mappings = load_mappings()

    if index.ntotal != len(mappings):
        print(
            "[UYARI] Index ve mapping kayıt sayısı farklı: "
            f"index={index.ntotal}, mappings={len(mappings)}"
        )

    return model, index, mappings


def search(query: str, model, index, mappings, top_k: int = TOP_K):
    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vector, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = mappings.get(str(idx))
        if not item:
            continue
        results.append((float(score), item))
    return results


def main():
    configure_output()
    model, index, mappings = load_search_engine()

    print("\n" + "=" * 50)
    print("   SEMANTİK VERİ SETİ ARAMA MOTORUNA HOŞ GELDİNİZ")
    print("=" * 50)

    while True:
        query = input("\nNe tür bir veri seti arıyorsunuz? (Çıkış için 'q'): ").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        print(f"\n'{query}' için en alakalı sonuçlar:")
        print("-" * 50)

        for rank, (score, result) in enumerate(search(query, model, index, mappings), start=1):
            desc = (result.get("text") or "").replace("\n", " ")
            print(f"{rank}. BAŞLIK: {result.get('title', 'İsimsiz')}")
            print(f"   KAYNAK: {(result.get('source') or 'unknown').upper()}")
            print(f"   SKOR: {score:.4f}")
            print(f"   URL: {result.get('url') or 'N/A'}")
            print(f"   ÖZET: {desc[:180]}...")
            print("-" * 50)


if __name__ == "__main__":
    main()
