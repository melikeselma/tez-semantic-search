import json, re, unicodedata
from pathlib import Path

# --- regexler ---
CODEBLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

# Dataset card YAML header'ını (--- ... ---) kaldırmak için
YAML_FRONTMATTER_RE = re.compile(r"^---.*?---\s*", re.DOTALL)

# HF card içindeki sık metadata satırları (gürültü)
NOISE_LINE_PREFIXES = (
    "license:", "language:", "tags:", "task_categories:", "size_categories:",
    "configs:", "dataset_info:", "pretty_name:", "viewer:", "paperswithcode_id:",
    "annotations_creators:", "multilinguality:", "source_datasets:", "language_creators:",
)

def clean_text(text: str):
    notes, flags = [], []

    if not text or not text.strip():
        return "", notes, ["empty"]

    t = unicodedata.normalize("NFKC", text)
    notes.append("nfkc")

    # YAML frontmatter kaldır
    t2 = YAML_FRONTMATTER_RE.sub("", t)
    if t2 != t:
        notes.append("remove_yaml_frontmatter")
    t = t2

    # kod blokları
    if "```" in t:
        t = CODEBLOCK_RE.sub(" ", t)
        notes.append("strip_codeblock")

    # html
    if "<" in t and ">" in t:
        t2 = HTML_TAG_RE.sub(" ", t)
        if t2 != t:
            notes.append("strip_html")
        t = t2

    # satır satır noise filtrele
    lines = []
    removed = 0
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith(NOISE_LINE_PREFIXES):
            removed += 1
            continue
        lines.append(s)

    if removed:
        notes.append(f"remove_noise_lines:{removed}")

    t = " ".join(lines)

    # url maskele
    if URL_RE.search(t):
        t = URL_RE.sub(" <URL> ", t)
        notes.append("mask_url")

    # whitespace normalize
    t = re.sub(r"\s+", " ", t).strip()
    notes.append("ws_norm")

    # quality flag
    if len(t) < 80:
        flags.append("too_short")

    return t, notes, flags

def load_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def save_jsonl(path, recs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    in_path = "data/raw/huggingface/raw_hf.jsonl"
    out_path = "data/processed/hf_clean.jsonl"

    recs = load_jsonl(in_path)
    cleaned = []

    for r in recs:
        raw = r.get("raw_description", "")
        clean, notes, flags = clean_text(raw)
        r["clean_description_basic"] = clean
        r["cleaning_notes"] = notes
        r["quality_flags"] = list(set(r.get("quality_flags", []) + flags))
        r["description_len_chars"] = len(clean)
        r["description_len_words"] = len(clean.split()) if clean else 0
        cleaned.append(r)

    save_jsonl(out_path, cleaned)
    print("OK ->", out_path)
    print("records:", len(cleaned))

if __name__ == "__main__":
    main()