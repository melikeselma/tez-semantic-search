"""Fetch metadata for a curated list of popular Kaggle datasets.

Downloads dataset-metadata.json via the Kaggle API for each ref, applies the
same cleaning rules as clean_kaggle.py and appends new records to
data/processed/kaggle_clean.jsonl so the rest of the pipeline (merge_all ->
normalize -> build_faiss_index) can be re-run unchanged.
"""
import html
import json
import re
import tempfile
import time
import unicodedata
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent
KAGGLE_CLEAN_PATH = ROOT / "data" / "processed" / "kaggle_clean.jsonl"

POPULAR_KAGGLE_DATASETS = [
    # UCI classics
    "uciml/iris",
    "uciml/red-wine-quality-cortez-et-al-2009",
    "uciml/pima-indians-diabetes-database",
    "uciml/breast-cancer-wisconsin-data",
    "uciml/adult-census-income",
    "uciml/mushroom-classification",
    # Finance / fraud
    "mlg-ulb/creditcardfraud",
    "arjunbhasin2013/ccdata",
    # Housing / regression
    "camnugent/california-housing-prices",
    # Movies / books / entertainment
    "rounakbanik/the-movies-dataset",
    "tmdb/tmdb-movie-metadata",
    "shivamb/netflix-shows",
    "jealousleopard/goodreadsbooks",
    "grouplens/movielens-20m-dataset",
    "PromptCloudHQ/imdb-data",
    # NLP / sentiment
    "kazanova/sentiment140",
    "crowdflower/twitter-airline-sentiment",
    "snap/amazon-fine-food-reviews",
    "yelp-dataset/yelp-dataset",
    "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
    # Sports
    "karangadiya/fifa19",
    "hugomathien/soccer",
    # Transportation
    "usdot/flight-delays",
    "fivethirtyeight/uber-pickups-in-new-york-city",
    # Healthcare
    "ronitf/heart-disease-uci",
    "fedesoriano/stroke-prediction-dataset",
    # Climate / time series
    "berkeleyearth/climate-change-earth-surface-temperature-data",
    "mczielinski/bitcoin-historical-data",
    # E-commerce / business
    "olistbr/brazilian-ecommerce",
    "carrie1/ecommerce-data",
    "gregorut/videogamesales",
    # Speech / audio
    "mmoreaux/audio-cats-and-dogs",
    # Images
    "moltean/fruits",
    "alxmamaev/flowers-recognition",
    "puneet6060/intel-image-classification",
    "tongpython/cat-and-dog",
    # Misc popular
    "alessiocorrado99/animals10",
    "rtatman/188-million-us-wildfires",
    "noaa/gsod",
    "datasnaek/youtube-new",
    # Flowers / botanical
    "mexwell/orchid-flower-dataset",
    "nunenuh/pytorch-challange-flower-dataset",
    "marquis03/flower-classification",
    "imsparsh/flowers-dataset",
    "rahmasleam/flowers-dataset",
]

# Cleaning regexes — kept aligned with clean_kaggle.py.
RE_CODE = re.compile(r"```.*?```", re.DOTALL)
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
RE_IMAGE_MD = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MULTI_WS = re.compile(r"\s+")

MIN_COMBINED_CHARS = 20


def fix_encoding(text: str) -> str:
    replacements = {
        "Ã¢â‚¬â„¢": "'", "Ã¢â‚¬Å“": '"', "Ã¢â‚¬ï¿½": '"',
        "Ã¢â‚¬â€œ": "-", "Ã¢â‚¬â€": "-", "Ã¢â‚¬Â¦": "...", "Ã‚": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = fix_encoding(text)
    text = RE_CODE.sub(" ", text)
    text = RE_IMAGE_MD.sub(" ", text)
    text = RE_MD_LINK.sub(r"\1", text)
    text = RE_HTML_TAG.sub(" ", text)
    text = RE_URL.sub(" <URL> ", text)
    text = RE_MULTI_WS.sub(" ", text)
    return text.strip()


def join_keywords(keywords) -> str:
    if not isinstance(keywords, list):
        return ""
    cleaned = [clean_text(str(item)) for item in keywords if str(item).strip()]
    return ", ".join([item for item in cleaned if item])


def extract_license(meta: dict) -> str:
    licenses = meta.get("licenses")
    if isinstance(licenses, list) and licenses:
        first = licenses[0]
        if isinstance(first, dict):
            return (first.get("name") or "").strip()
        return str(first).strip()
    return ""


def build_record(ref: str, meta: dict) -> dict | None:
    title = clean_text(meta.get("title", "")) or "Untitled Dataset"
    subtitle = clean_text(meta.get("subtitle", ""))
    description = clean_text(meta.get("description", ""))
    keywords = meta.get("keywords") or []

    notes = []
    parts = []
    if subtitle:
        parts.append(subtitle)
        notes.append("subtitle")
    if description:
        parts.append(description)
        notes.append("description")
    joined_keywords = join_keywords(keywords)
    if joined_keywords:
        parts.append(f"Keywords: {joined_keywords}")
        notes.append("keywords")

    full_description = " ".join(parts).strip()
    if not full_description:
        return None

    combined_text = f"{title}. {full_description}".strip()
    if len(combined_text) < MIN_COMBINED_CHARS:
        return None

    flags = []
    if "description" not in notes:
        flags.append("no_long_description")
    if notes == ["keywords"]:
        flags.append("keyword_only")
    if len(full_description) < 80:
        flags.append("short_description")

    return {
        "source": "kaggle",
        "ref": ref,
        "title": title,
        "description": full_description,
        "text": combined_text,
        "url": f"https://www.kaggle.com/datasets/{ref}",
        "keywords": keywords if isinstance(keywords, list) else [],
        "license": extract_license(meta),
        "cleaning_notes": notes,
        "quality_flags": flags,
        "description_len_chars": len(full_description),
        "description_len_words": len(full_description.split()),
    }


def load_existing_refs() -> set:
    refs = set()
    if not KAGGLE_CLEAN_PATH.exists():
        return refs
    with KAGGLE_CLEAN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("ref"):
                    refs.add(rec["ref"])
            except json.JSONDecodeError:
                continue
    return refs


def fetch_metadata(api: KaggleApi, ref: str) -> dict | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            api.dataset_metadata(ref, path=tmpdir)
        except Exception as exc:
            print(f"[WARN] {ref}: {exc}")
            return None
        path = Path(tmpdir) / "dataset-metadata.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] {ref}: bad metadata file: {exc}")
            return None
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except json.JSONDecodeError:
                return None
        if isinstance(obj, dict) and isinstance(obj.get("info"), dict):
            obj = obj["info"]
        return obj if isinstance(obj, dict) else None


def main():
    api = KaggleApi()
    api.authenticate()

    existing = load_existing_refs()
    print(f"[START] existing kaggle records: {len(existing)}")

    added = skipped = failed = 0
    with KAGGLE_CLEAN_PATH.open("a", encoding="utf-8", newline="\n") as out:
        for ref in POPULAR_KAGGLE_DATASETS:
            if ref in existing:
                skipped += 1
                continue
            meta = fetch_metadata(api, ref)
            if not meta:
                failed += 1
                continue
            rec = build_record(ref, meta)
            if not rec:
                failed += 1
                print(f"[WARN] {ref}: empty description, skipped")
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing.add(ref)
            added += 1
            if added % 5 == 0:
                print(f"[OK] added={added} (last: {ref})")
            time.sleep(0.4)

    print(f"[DONE] added={added}, skipped={skipped}, failed={failed}")
    print(f"[OUT] {KAGGLE_CLEAN_PATH}")


if __name__ == "__main__":
    main()
