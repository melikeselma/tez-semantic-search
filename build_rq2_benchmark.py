import json
from pathlib import Path

from search import load_mappings
from search_profiles import DEFAULT_PROFILE_KEY

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "data" / "evaluation"
OUTPUT_PATH = EVAL_DIR / "relevance_judgments_rq2.json"

RQ2_TEMPLATES = [
    {
        "id": "rq2_001",
        "anchor_ref": "stanfordnlp/imdb",
        "target_source": "kaggle",
        "category": "movie_sentiment",
        "topic": "movie_reviews",
        "relevant_refs": [
            "varun08/imdb-dataset",
            "ashirwadsangwan/imdb-dataset",
            "ahmedosamamath/imdb-dataset",
            "justsahil/imdb-dataset-csv-updates-weekly",
        ],
        "notes": "HF movie review sentiment anchor should retrieve Kaggle IMDb-style movie review datasets.",
    },
    {
        "id": "rq2_002",
        "anchor_ref": "jhan21/amazon-beauty-reviews-dataset",
        "target_source": "kaggle",
        "category": "product_reviews",
        "topic": "amazon_reviews",
        "relevant_refs": [
            "humagonen/amazon-reviews-csv",
            "ashishkumarak/amazon-shopping-reviews-daily-updated",
        ],
        "notes": "HF Amazon reviews anchor should retrieve Kaggle Amazon review corpora.",
    },
    {
        "id": "rq2_003",
        "anchor_ref": "McAuley-Lab/Amazon-Reviews-2023",
        "target_source": "kaggle",
        "category": "product_reviews",
        "topic": "amazon_reviews",
        "relevant_refs": [
            "humagonen/amazon-reviews-csv",
            "ashishkumarak/amazon-shopping-reviews-daily-updated",
        ],
        "notes": "Large-scale HF Amazon review anchor should bridge to Kaggle Amazon review datasets.",
    },
    {
        "id": "rq2_004",
        "anchor_ref": "fthbrmnby/turkish_product_reviews",
        "target_source": "kaggle",
        "category": "turkish_sentiment",
        "topic": "turkish_product_reviews",
        "relevant_refs": [
            "furkangozukara/turkish-product-reviews",
        ],
        "notes": "Turkish sentiment anchor should retrieve the Kaggle Turkish e-commerce review dataset.",
    },
    {
        "id": "rq2_005",
        "anchor_ref": "tdiggelm/climate_fever",
        "target_source": "kaggle",
        "category": "climate_discourse",
        "topic": "climate_change",
        "relevant_refs": [
            "asaniczka/public-opinion-on-climate-change-updated-daily",
        ],
        "notes": "Climate-related HF benchmark should retrieve Kaggle climate discussion data despite source differences.",
    },
    {
        "id": "rq2_006",
        "anchor_ref": "Buraaq/quran-audio-text-dataset",
        "target_source": "kaggle",
        "category": "religious_text_audio",
        "topic": "quran_text_audio",
        "relevant_refs": [
            "bentaylor/sacred-texts-for-visualisation",
        ],
        "notes": "Quran audio/text anchor tests whether cross-source retrieval can bridge to Kaggle religious text data.",
    },
    {
        "id": "rq2_007",
        "anchor_ref": "varun08/imdb-dataset",
        "target_source": "huggingface",
        "category": "movie_sentiment",
        "topic": "movie_reviews",
        "relevant_refs": [
            "stanfordnlp/imdb",
        ],
        "notes": "Kaggle IMDb anchor should retrieve the HF IMDb benchmark dataset.",
    },
    {
        "id": "rq2_008",
        "anchor_ref": "humagonen/amazon-reviews-csv",
        "target_source": "huggingface",
        "category": "product_reviews",
        "topic": "amazon_reviews",
        "relevant_refs": [
            "jhan21/amazon-beauty-reviews-dataset",
            "McAuley-Lab/Amazon-Reviews-2023",
        ],
        "notes": "Kaggle Amazon reviews anchor should retrieve semantically matching HF review corpora.",
    },
    {
        "id": "rq2_009",
        "anchor_ref": "ashishkumarak/amazon-shopping-reviews-daily-updated",
        "target_source": "huggingface",
        "category": "product_reviews",
        "topic": "amazon_reviews",
        "relevant_refs": [
            "jhan21/amazon-beauty-reviews-dataset",
            "McAuley-Lab/Amazon-Reviews-2023",
        ],
        "notes": "Daily-updated Kaggle Amazon app-review anchor should still bridge to HF Amazon review datasets.",
    },
    {
        "id": "rq2_010",
        "anchor_ref": "furkangozukara/turkish-product-reviews",
        "target_source": "huggingface",
        "category": "turkish_sentiment",
        "topic": "turkish_product_reviews",
        "relevant_refs": [
            "fthbrmnby/turkish_product_reviews",
        ],
        "notes": "Kaggle Turkish e-commerce reviews should retrieve the HF Turkish sentiment dataset.",
    },
    {
        "id": "rq2_011",
        "anchor_ref": "asaniczka/public-opinion-on-climate-change-updated-daily",
        "target_source": "huggingface",
        "category": "climate_discourse",
        "topic": "climate_change",
        "relevant_refs": [
            "tdiggelm/climate_fever",
        ],
        "notes": "Kaggle climate discussion anchor should retrieve the HF climate-related benchmark.",
    },
    {
        "id": "rq2_012",
        "anchor_ref": "bentaylor/sacred-texts-for-visualisation",
        "target_source": "huggingface",
        "category": "religious_text_audio",
        "topic": "quran_text_audio",
        "relevant_refs": [
            "Buraaq/quran-audio-text-dataset",
            "nazimali/quran",
        ],
        "notes": "Kaggle religious text anchor should retrieve HF Quran datasets.",
    },
    {
        "id": "rq2_013",
        "anchor_ref": "mathurinache/cmu-mosi",
        "target_source": "huggingface",
        "category": "multimodal_sentiment",
        "topic": "sentiment_analysis",
        "relevant_refs": [
            "stanfordnlp/imdb",
            "cornell-movie-review-data/rotten_tomatoes",
            "benjaminvdb/dbrd",
        ],
        "notes": "Kaggle multimodal sentiment anchor should retrieve HF sentiment datasets from the same semantic task family.",
    },
]


def build_record(template, ref_to_item):
    anchor = ref_to_item.get(template["anchor_ref"])
    if not anchor:
        raise ValueError(f"Missing anchor ref in mappings: {template['anchor_ref']}")

    missing_relevant = [
        ref for ref in template["relevant_refs"] if ref_to_item.get(ref) is None
    ]
    if missing_relevant:
        raise ValueError(
            "Missing relevant refs in mappings for "
            f"{template['id']}: {', '.join(missing_relevant)}"
        )

    anchor_source = (anchor.get("source") or "").lower()
    target_source = template["target_source"].lower()
    direction = f"{anchor_source}_to_{target_source}"

    return {
        "id": template["id"],
        "query": anchor.get("text") or "",
        "category": template["category"],
        "topic": template["topic"],
        "benchmark": "cross_source_similarity",
        "query_style": "dataset_description_anchor",
        "language": "en",
        "anchor_ref": template["anchor_ref"],
        "anchor_title": anchor.get("title") or template["anchor_ref"],
        "anchor_source": anchor_source,
        "target_source": target_source,
        "direction": direction,
        "source_filter": target_source,
        "relevant_refs": template["relevant_refs"],
        "notes": template["notes"],
    }


def main():
    mappings = load_mappings(DEFAULT_PROFILE_KEY)
    ref_to_item = {item.get("ref"): item for item in mappings.values()}

    records = [build_record(template, ref_to_item) for template in RQ2_TEMPLATES]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {len(records)} RQ2 judgments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
