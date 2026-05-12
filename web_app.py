import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from bm25 import BM25Index, search as bm25_search
from hybrid import DEFAULT_CANDIDATE_K, DEFAULT_SEMANTIC_WEIGHT, search as hybrid_search
from quality_scoring import infer_quality_flags, should_deemphasize_title
from query_understanding import build_query_plan
from search import load_mappings, load_search_engine, search as semantic_search
from search_profiles import DEFAULT_PROFILE_KEY, get_profile, get_profile_paths, list_profiles

BASE_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000
MAX_BODY_BYTES = 16_384
DEFAULT_WEB_PROFILE = DEFAULT_PROFILE_KEY
RQ1_REPORT_DIR = BASE_DIR / "reports" / "evaluation" / "rq1_method_comparison"
RQ1_SUMMARY_PATH = RQ1_REPORT_DIR / "evaluation_summary.json"
RQ1_QUERY_STYLE_PATH = RQ1_REPORT_DIR / "evaluation_by_query_style.json"
RQ1_BENCHMARK_PATH = RQ1_REPORT_DIR / "evaluation_by_benchmark.json"
RQ2_REPORT_DIR = BASE_DIR / "reports" / "evaluation" / "rq2_cross_source"
RQ2_SUMMARY_PATH = RQ2_REPORT_DIR / "evaluation_summary.json"
RQ2_DIRECTION_PATH = RQ2_REPORT_DIR / "evaluation_by_direction.json"
RQ2_TOPIC_PATH = RQ2_REPORT_DIR / "evaluation_by_topic.json"
RQ3_REPORT_DIR = BASE_DIR / "reports" / "evaluation" / "rq3_model_effect"
RQ3_SUMMARY_PATH = RQ3_REPORT_DIR / "evaluation_summary.json"
RQ3_LANGUAGE_PATH = RQ3_REPORT_DIR / "evaluation_by_language.json"
RQ3_BENCHMARK_PATH = RQ3_REPORT_DIR / "evaluation_by_benchmark.json"
RQ3_STUDY_SLICE_PATH = RQ3_REPORT_DIR / "evaluation_by_study_slice.json"
RQ4_REPORT_DIR = BASE_DIR / "reports" / "evaluation" / "rq4_description_quality"
RQ4_DETAILS_PATH = RQ4_REPORT_DIR / "evaluation_details.json"
RQ4_STYLE_PATH = RQ4_REPORT_DIR / "rq4_by_description_style.json"
RQ4_LENGTH_PATH = RQ4_REPORT_DIR / "rq4_by_length_bucket.json"
RQ4_TERM_PATH = RQ4_REPORT_DIR / "rq4_by_term_bucket.json"
RQ4_CORPUS_DISTRIBUTION_PATH = RQ4_REPORT_DIR / "rq4_corpus_feature_distribution.json"
LIVE_EVAL_QUERY_PATH = BASE_DIR / "data" / "evaluation" / "live_semantic_queries.json"
LIVE_EVAL_REPORT_ROOT = BASE_DIR / "reports" / "evaluation" / "live_semantic_queries"
HARD_NEGATIVE_QUERY_SPEC_PATH = BASE_DIR / "data" / "training" / "hard_negative_query_specs.json"
HARD_NEGATIVE_SUMMARY_PATH = BASE_DIR / "data" / "training" / "retriever_hard_negative_summary.json"
METHOD_LABELS = {
    "semantic": "Semantic",
    "bm25": "BM25",
    "hybrid": "Hybrid",
}
METHOD_ROLES = {
    "semantic": "Açıklama tabanlı semantic retrieval",
    "bm25": "Lexical baseline",
    "hybrid": "En güçlü pratik sistem",
}
DIRECTION_LABELS = {
    "huggingface_to_kaggle": "Hugging Face -> Kaggle",
    "kaggle_to_huggingface": "Kaggle -> Hugging Face",
}
TOPIC_LABELS = {
    "amazon_reviews": "Amazon reviews",
    "climate_change": "Climate change",
    "movie_reviews": "Movie reviews",
    "quran_text_audio": "Quran text and audio",
    "sentiment_analysis": "Sentiment analysis",
    "turkish_product_reviews": "Turkish product reviews",
}
STUDY_SLICE_LABELS = {
    "english_main": "English main",
    "cross_source": "Cross-source",
    "tr_subset": "Turkish subset",
}
LANGUAGE_LABELS = {
    "en": "English",
    "tr": "Turkish",
}
PROFILE_ORDER = ("minilm", "e5_base", "minilm_ft", "multilingual")
RQ3_BENCHMARK_PROFILES = ("minilm", "multilingual")
DESCRIPTION_STYLE_LABELS = {
    "metadata_heavy": "Metadata heavy",
    "mixed_structured": "Mixed structured",
    "narrative": "Narrative",
}
LENGTH_BUCKET_LABELS = {
    "short": "Short",
    "medium": "Medium",
    "long": "Long",
}
TERM_BUCKET_LABELS = {
    "term_sparse": "Term sparse",
    "term_moderate": "Term moderate",
    "term_rich": "Term rich",
}
CONTENT_TERM_RE = re.compile(r"[0-9a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}", re.IGNORECASE)
QUALITY_FLAG_META = {
    "empty": {
        "label": "Empty description",
        "severity": "risk",
        "message": "Kayıt açıklama taşımıyor; semantic sinyal çok zayıf.",
    },
    "no_long_description": {
        "label": "No long description",
        "severity": "warning",
        "message": "Ayrıntılı açıklama eksik; kayıt daha çok subtitle veya keyword üzerinden temsil edildi.",
    },
    "keyword_only": {
        "label": "Keyword only",
        "severity": "risk",
        "message": "Kayıt ağırlıkla anahtar kelime veya etiket seviyesinde; doğal dil açıklaması zayıf.",
    },
    "short_description": {
        "label": "Short description",
        "severity": "warning",
        "message": "Açıklama kısa; semantic retrieval için taşıdığı bağlam sınırlı.",
    },
    "too_short": {
        "label": "Too short",
        "severity": "warning",
        "message": "Temizleme asamasinda metin çok kısa gorunmus.",
    },
    "short_context": {
        "label": "Short context",
        "severity": "warning",
        "message": "Canlı tahmine göre açıklama kısa bağlam taşıyor.",
    },
    "metadata_heavy": {
        "label": "Metadata heavy",
        "severity": "warning",
        "message": "Açıklama alan listesi veya metadata ağırlıklı; anlamsal akış zayıf olabilir.",
    },
    "term_sparse": {
        "label": "Term sparse",
        "severity": "warning",
        "message": "Içerik terimi sayısı düşük; semantic ayırt edicilik azalabilir.",
    },
    "low_information": {
        "label": "Low information",
        "severity": "risk",
        "message": "Kısa, metadata ağırlıklı veya keyword benzeri kayıt; retrieval kalitesi için riskli.",
    },
}
CONTENT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "dataset",
    "data",
    "are",
    "into",
    "over",
    "under",
    "between",
    "about",
    "your",
    "their",
    "ve",
    "bir",
    "icin",
    "için",
    "olan",
    "veri",
    "seti",
}


HTML = """<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Semantic Dataset Search</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    :root {
      --bg: #f0edff;
      --bg2: #e8e3ff;
      --ink: #1e1147;
      --muted: #6b5fa0;
      --line: rgba(109, 40, 217, 0.15);
      --panel: #ffffff;
      --accent: #7c3aed;
      --accent2: #0891b2;
      --accent3: #db2777;
      --accent4: #059669;
      --accent5: #ea580c;
      --accent-strong:#6d28d9;
      --warn: #dc2626;
      --soft: rgba(124,58,237,0.08);
      --tag: rgba(8,145,178,0.10);
      --grad: linear-gradient(135deg,#7c3aed,#0891b2);
      --grad2: linear-gradient(135deg,#db2777,#7c3aed);
      --grad3: linear-gradient(135deg,#059669,#0891b2);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: 'Inter','Segoe UI',system-ui,sans-serif;
      background: var(--bg);
      color: var(--ink);
      min-height: 100vh;
      background-image:
        radial-gradient(ellipse at 0% 0%, rgba(124,58,237,.14) 0%,transparent 50%),
        radial-gradient(ellipse at 100% 100%,rgba(8,145,178,.12) 0%,transparent 50%),
        radial-gradient(ellipse at 60% 0%, rgba(219,39,119,.08) 0%,transparent 40%);
    }

    ::-webkit-scrollbar { width:6px; }
    ::-webkit-scrollbar-track { background: var(--bg2); }
    ::-webkit-scrollbar-thumb { background: var(--accent); border-radius:3px; }

    /* ── TOPBAR ─────────────────────────────────────────── */
    main { width:min(1260px,calc(100% - 32px)); margin:0 auto; padding:28px 0 48px; }

    .topbar {
      display:flex; align-items:flex-start; justify-content:space-between;
      gap:18px; margin-bottom:24px;
      background: var(--panel);
      border:1px solid var(--line);
      border-radius:20px;
      padding:20px 24px;
      box-shadow:0 2px 16px rgba(124,58,237,.08);
    }

    .brand { display:flex; align-items:center; gap:14px; margin-bottom:6px; }

    .brand-icon {
      width:52px; height:52px; border-radius:16px;
      background:var(--grad); display:flex; align-items:center;
      justify-content:center; font-size:28px;
      box-shadow:0 4px 20px rgba(124,58,237,.35); flex-shrink:0;
    }

    h1 {
      margin:0; font-size:30px; font-weight:900; line-height:1.1;
      background:var(--grad); -webkit-background-clip:text;
      -webkit-text-fill-color:transparent; background-clip:text;
    }

    .subtitle {
      margin:2px 0 0 66px; color:var(--muted);
      font-size:13px; line-height:1.6; max-width:640px;
    }

    /* stats */
    .stats { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:8px; min-width:220px; }

    .stat {
      border:1px solid var(--line); border-radius:14px;
      background:linear-gradient(135deg,rgba(124,58,237,.06),rgba(8,145,178,.06));
      padding:10px 16px; min-width:96px; text-align:center;
    }

    .stat strong {
      display:block; font-size:22px; font-weight:900;
      background:var(--grad); -webkit-background-clip:text;
      -webkit-text-fill-color:transparent; background-clip:text;
    }

    .stat span { color:var(--muted); font-size:10px; font-weight:700;
      text-transform:uppercase; letter-spacing:.07em; }

    /* ── APP LAYOUT: sidebar + content ─────────────────── */
    .app-layout { display:flex; gap:20px; align-items:flex-start; }

    /* ── SIDEBAR ─────────────────────────────────────────── */
    .sidebar {
      flex:0 0 200px;
      display:flex; flex-direction:column; gap:4px;
      position:sticky; top:20px;
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:20px;
      padding:14px 10px;
      box-shadow:0 2px 16px rgba(124,58,237,.07);
    }

    .sidebar-label {
      font-size:10px; font-weight:800; letter-spacing:.1em;
      text-transform:uppercase; color:var(--muted);
      padding:4px 10px 8px; margin-top:6px;
    }

    .tab-btn {
      width:100%; min-height:auto;
      margin:0; padding:11px 14px;
      border:1px solid transparent;
      border-radius:12px;
      background:transparent;
      color:var(--muted);
      font-size:14px; font-weight:600;
      cursor:pointer; transition:all .18s;
      text-align:left;
      display:flex; align-items:center; gap:10px;
    }

    .tab-btn .tab-icon {
      width:30px; height:30px; border-radius:8px;
      display:flex; align-items:center; justify-content:center;
      font-size:15px; flex-shrink:0;
      background:rgba(124,58,237,.08);
      transition:all .18s;
    }

    .tab-btn .tab-text { display:flex; flex-direction:column; line-height:1.2; }
    .tab-btn .tab-sub { font-size:10px; opacity:.7; font-weight:500; }

    .tab-btn:hover {
      background:var(--soft); color:var(--accent);
      border-color:var(--line);
    }
    .tab-btn:hover .tab-icon { background:rgba(124,58,237,.15); }

    .tab-btn[aria-selected="true"] {
      background:var(--grad); color:#fff;
      border-color:transparent;
      box-shadow:0 4px 16px rgba(124,58,237,.3);
    }
    .tab-btn[aria-selected="true"] .tab-icon {
      background:rgba(255,255,255,.2);
    }
    .tab-btn[aria-selected="true"] .tab-sub { opacity:.85; }

    .tab-panel[hidden] { display:none; }

    /* ── CONTENT AREA ───────────────────────────────────── */
    .content-area { flex:1; min-width:0; }

    /* ── PANEL ──────────────────────────────────────────── */
    .panel {
      border:1px solid var(--line); border-radius:18px;
      background:var(--panel); padding:22px; margin-bottom:16px;
      box-shadow:0 2px 12px rgba(124,58,237,.06);
    }

    .panel h2 {
      margin:0 0 10px; font-size:19px; font-weight:800;
      display:flex; align-items:center; gap:10px;
    }

    .panel-note { margin:0; color:var(--muted); font-size:13px; line-height:1.6; }

    /* ── FORM ───────────────────────────────────────────── */
    form {
      display:grid;
      grid-template-columns:minmax(0,1.8fr) 160px 140px 140px 110px 56px;
      gap:10px; align-items:end; margin-top:16px;
    }

    label {
      display:block; font-size:11px; font-weight:800;
      text-transform:uppercase; letter-spacing:.07em;
      margin-bottom:7px;
    }

    /* Renkli etiketler */
    label[for="query"] { color:var(--accent); }
    label[for="profile"] { color:var(--accent2); }
    label[for="method"] { color:var(--accent3); }
    label[for="source"] { color:var(--accent5); }
    label[for="top-k"] { color:var(--accent4); }

    input, select, button {
      width:100%; min-height:44px; border-radius:10px; font:inherit;
    }

    input, select {
      border:2px solid var(--line);
      background:#faf8ff; color:var(--ink);
      padding:9px 13px;
      transition:border-color .18s, box-shadow .18s;
      font-size:14px;
    }

    input:focus, select:focus {
      outline:none; border-color:var(--accent);
      box-shadow:0 0 0 3px rgba(124,58,237,.15);
    }

    /* Her select'e kendi renk bordürü */
    #profile { border-color:rgba(8,145,178,.35); }
    #method { border-color:rgba(219,39,119,.35); }
    #source { border-color:rgba(234,88,12,.35); }
    #top-k { border-color:rgba(5,150,105,.35); }
    #profile:focus { border-color:var(--accent2); box-shadow:0 0 0 3px rgba(8,145,178,.15); }
    #method:focus { border-color:var(--accent3); box-shadow:0 0 0 3px rgba(219,39,119,.15); }
    #source:focus { border-color:var(--accent5); box-shadow:0 0 0 3px rgba(234,88,12,.15); }
    #top-k:focus { border-color:var(--accent4); box-shadow:0 0 0 3px rgba(5,150,105,.15); }

    select option { background:#fff; color:var(--ink); }

    /* Search butonu, sadece ikon */
    button#submit {
      min-height:44px; border:none;
      background:var(--grad); color:#fff;
      cursor:pointer; font-weight:900; font-size:22px;
      padding:0; border-radius:12px;
      box-shadow:0 4px 16px rgba(124,58,237,.35);
      transition:all .18s; display:flex;
      align-items:center; justify-content:center;
      margin-top:0;
    }
    button#submit:hover { transform:translateY(-2px); box-shadow:0 6px 24px rgba(124,58,237,.5); }
    button#submit:disabled{ opacity:.6; transform:none; cursor:wait; }

    button:not(#submit):not(.example-btn):not(.detail-btn):not(.close-btn):not(.tab-btn) {
      border:none; background:var(--grad); color:#fff;
      cursor:pointer; font-weight:700; padding:0 18px;
      margin-top:20px; border-radius:10px;
      box-shadow:0 4px 14px rgba(124,58,237,.3);
      transition:all .18s;
    }

    /* ── TOGGLES & EXAMPLES ─────────────────────────────── */
    .search-toggle {
      display:flex; flex-wrap:wrap; align-items:center;
      gap:12px; margin-top:14px;
    }

    .toggle {
      display:inline-flex; align-items:center; gap:8px;
      color:var(--muted); font-size:13px; font-weight:600; margin:0;
    }

    .toggle input { width:16px; height:16px; min-height:16px; margin:0; accent-color:var(--accent); }

    .examples { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }

    .example-btn,.detail-btn,.close-btn {
      width:auto; min-height:30px; margin:0;
      border-radius:999px; border:2px solid var(--line);
      background:#fff; color:var(--accent);
      padding:4px 14px; font-size:12px; font-weight:700;
      cursor:pointer; transition:all .18s; box-shadow:none;
    }
    .example-btn:hover,.detail-btn:hover {
      border-color:var(--accent); background:var(--soft); color:var(--accent-strong);
      transform:translateY(-1px); box-shadow:0 3px 10px rgba(124,58,237,.15);
    }
    .close-btn { color:var(--muted); }
    .close-btn:hover { border-color:var(--warn); color:var(--warn); background:#fff5f5; }

    /* ── STATUS ─────────────────────────────────────────── */
    .score-note,.status {
      margin:10px 0 0; color:var(--muted); font-size:13px; line-height:1.5;
    }
    .status { min-height:22px; margin-bottom:12px; }
    .status.error { color:var(--warn); font-weight:700; }

    /* ── RQ CARDS ───────────────────────────────────────── */
    .research-list { display:grid; gap:10px; margin:0; padding:0; list-style:none; }

    .research-item {
      border:1px solid var(--line); border-radius:12px; padding:14px;
      background:linear-gradient(135deg,rgba(124,58,237,.04),rgba(8,145,178,.04));
    }
    .research-item strong { display:block; margin-bottom:6px; color:var(--accent); }
    .research-item p { margin:0; color:var(--muted); line-height:1.5; }

    .rq1-grid {
      display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
      gap:12px; margin-top:14px;
    }

    .rq1-card {
      border:1px solid var(--line); border-radius:14px;
      background:#fff; padding:16px;
      transition:border-color .18s,box-shadow .18s;
      box-shadow:0 1px 6px rgba(124,58,237,.05);
    }
    .rq1-card:hover {
      border-color:rgba(124,58,237,.35);
      box-shadow:0 4px 20px rgba(124,58,237,.12);
      transform:translateY(-1px);
    }
    .rq1-card h3,.rq1-card h4 { margin:0 0 6px; font-size:16px; font-weight:700; color:var(--ink); }
    .rq1-label {
      display:inline-block; margin-bottom:8px; padding:3px 10px;
      border-radius:999px; background:var(--soft); color:var(--accent);
      font-size:10px; font-weight:800; letter-spacing:.06em; text-transform:uppercase;
    }
    .rq1-card p { margin:0 0 10px; color:var(--muted); line-height:1.5; font-size:13px; }
    .rq1-metrics { display:flex; flex-wrap:wrap; gap:8px; }

    .rq1-metric {
      min-width:80px; padding:8px 10px; border-radius:10px;
      background:linear-gradient(135deg,rgba(124,58,237,.07),rgba(8,145,178,.07));
      border:1px solid var(--line);
    }
    .rq1-metric strong {
      display:block; font-size:17px; font-weight:900;
      background:var(--grad); -webkit-background-clip:text;
      -webkit-text-fill-color:transparent; background-clip:text;
    }
    .rq1-metric span { font-size:11px; color:var(--muted); }

    .rq1-banner {
      margin-top:14px; border:1px solid rgba(8,145,178,.25);
      border-radius:12px; background:rgba(8,145,178,.05); padding:14px;
    }
    .rq1-banner strong { display:block; margin-bottom:6px; color:var(--accent2); }
    .rq1-banner p { margin:0; line-height:1.6; color:var(--muted); }

    /* ── RESULTS ─────────────────────────────────────────── */
    .results { display:grid; gap:12px; }

    .result {
      border:1px solid var(--line); border-radius:16px;
      background:var(--panel); padding:18px;
      transition:all .18s; position:relative; overflow:hidden;
      box-shadow:0 1px 8px rgba(124,58,237,.06);
    }
    .result::before {
      content:''; position:absolute; top:0; left:0; right:0;
      height:3px; background:var(--grad); opacity:0; transition:opacity .18s;
    }
    .result:hover {
      border-color:rgba(124,58,237,.3);
      box-shadow:0 6px 24px rgba(124,58,237,.12);
      transform:translateY(-1px);
    }
    .result:hover::before { opacity:1; }

    .result-head {
      display:flex; align-items:flex-start; justify-content:space-between;
      gap:12px; margin-bottom:10px;
    }
    .result h2 { margin:0; font-size:17px; font-weight:700; line-height:1.3; overflow-wrap:anywhere; color:var(--ink); }

    .score {
      flex:0 0 auto; border-radius:8px;
      background:var(--grad); color:#fff;
      padding:6px 11px; font-weight:900; font-size:12px;
      letter-spacing:.03em; box-shadow:0 3px 10px rgba(124,58,237,.3);
    }

    .meta { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }

    .chip {
      display:inline-block; border-radius:999px; padding:3px 10px;
      font-size:11px; font-weight:700; background:var(--tag);
      color:var(--accent2); border:1px solid rgba(8,145,178,.2);
    }
    .chip-warning { background:rgba(234,88,12,.08); color:var(--accent5); border-color:rgba(234,88,12,.2); }
    .chip-risk { background:rgba(220,38,38,.08); color:var(--warn); border-color:rgba(220,38,38,.2); }

    .summary { margin:0 0 12px; color:var(--muted); line-height:1.6; font-size:14px; overflow-wrap:anywhere; }

    a { color:var(--accent); font-weight:600; text-decoration:none; overflow-wrap:anywhere; }
    a:hover { text-decoration:underline; color:var(--accent-strong); }

    .actions { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }

    .empty {
      border:2px dashed rgba(124,58,237,.2); border-radius:14px;
      padding:40px 24px; color:var(--muted);
      background:rgba(124,58,237,.03); text-align:center; font-size:15px;
    }
    .empty::before { content:'🔍'; display:block; font-size:40px; margin-bottom:12px; }

    /* ── MODAL ───────────────────────────────────────────── */
    .modal-backdrop {
      position:fixed; inset:0; display:none;
      align-items:stretch; justify-content:flex-end;
      background:rgba(30,17,71,.45); backdrop-filter:blur(6px); z-index:10;
    }
    .modal-backdrop.open { display:flex; }

    .modal {
      width:min(580px,100%); height:100vh; overflow:auto;
      background:#fff; border-left:1px solid var(--line);
      padding:24px; box-shadow:-16px 0 48px rgba(124,58,237,.18);
    }

    .modal-head {
      display:flex; justify-content:space-between; gap:14px;
      align-items:flex-start; margin-bottom:16px;
    }
    .modal h2 { margin:0; font-size:20px; font-weight:800; line-height:1.3; overflow-wrap:anywhere; color:var(--ink); }

    .modal dl { display:grid; grid-template-columns:130px minmax(0,1fr); gap:8px 14px; margin:0 0 14px; }
    .modal dt { color:var(--muted); font-weight:700; font-size:13px; }
    .modal dd { margin:0; overflow-wrap:anywhere; font-size:13px; }

    .modal-section { border-top:1px solid var(--line); padding-top:14px; margin-top:14px; }
    .modal-section h3 {
      margin:0 0 10px; font-size:10px; font-weight:800;
      color:var(--accent2); text-transform:uppercase; letter-spacing:.08em;
    }

    .flag-list { display:grid; gap:8px; }
    .flag-item {
      border:1px solid var(--line); border-radius:10px;
      background:rgba(124,58,237,.04); padding:12px;
    }
    .flag-item strong { display:block; margin-bottom:4px; color:var(--ink); }
    .flag-item p { margin:0; color:var(--muted); line-height:1.5; font-size:13px; }
    .flag-meta {
      display:inline-block; margin-top:6px; color:var(--muted);
      font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.05em;
    }

    .why-box {
      border-radius:10px; background:var(--soft);
      border:1px solid var(--line); color:var(--ink);
      padding:12px; line-height:1.6; font-size:13px;
    }
    .modal-text { white-space:pre-wrap; line-height:1.6; overflow-wrap:anywhere; font-size:13px; color:var(--muted); }

    /* ── RESPONSIVE ──────────────────────────────────────── */
    @media (max-width:980px) {
      .app-layout { flex-direction:column; }
      .sidebar { flex:none; width:100%; flex-direction:row; flex-wrap:wrap;
                    position:static; }
      .tab-btn { flex:1 1 auto; text-align:center; justify-content:center; }
      form { grid-template-columns:1fr 1fr; }
      .rq1-grid { grid-template-columns:1fr; }
    }

    @media (max-width:680px) {
      main { width:min(100% - 20px,1260px); padding-top:16px; }
      .topbar { display:block; }
      .stats { justify-content:flex-start; margin-top:12px; }
      h1 { font-size:24px; }
      form { grid-template-columns:1fr; }
      .result-head { display:block; }
      .score { display:inline-block; margin-top:8px; }
      .modal dl { grid-template-columns:1fr; }
      .modal { width:100%; }
    }

    /* ── RQ INFO CARDS ──────────────────────────────────── */
    .rq-hero {
      display:flex; align-items:center; gap:14px;
      margin-bottom:20px; padding-bottom:16px;
      border-bottom:2px solid var(--line);
    }
    .rq-hero-icon {
      font-size:42px; line-height:1; flex-shrink:0;
    }
    .rq-hero h2 {
      margin:0 0 4px; font-size:24px; font-weight:900;
      background:var(--grad); -webkit-background-clip:text;
      -webkit-text-fill-color:transparent; background-clip:text;
    }
    .rq-hero-sub { color:var(--muted); font-size:14px; font-weight:500; }

    .rq-cluster { margin-bottom:20px; }
    .rq-cluster-label {
      display:inline-flex; align-items:center; gap:7px;
      font-size:11px; font-weight:800; text-transform:uppercase;
      letter-spacing:.09em; margin-bottom:10px;
      padding:4px 12px; border-radius:999px;
    }
    .rq-cluster-label.purple { background:rgba(124,58,237,.1); color:var(--accent); }
    .rq-cluster-label.cyan { background:rgba(8,145,178,.1); color:var(--accent2); }
    .rq-cluster-label.green { background:rgba(5,150,105,.1); color:var(--accent4); }
    .rq-cluster-label.pink { background:rgba(219,39,119,.1); color:var(--accent3); }

    .rq-cluster-body {
      font-size:15px; font-weight:500; color:var(--ink);
      line-height:1.65; padding:14px 18px;
      background:#fff; border-radius:14px;
      border-left:4px solid;
      box-shadow:0 2px 10px rgba(124,58,237,.06);
    }
    .rq-cluster-body.purple { border-color:var(--accent); }
    .rq-cluster-body.cyan { border-color:var(--accent2); }
    .rq-cluster-body.green { border-color:var(--accent4); }
    .rq-cluster-body.pink { border-color:var(--accent3); }

    .rq-cluster-body strong { color:var(--ink); }

    .rq-pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
    .rq-pill {
      padding:5px 13px; border-radius:999px;
      font-size:12px; font-weight:700; border:2px solid;
    }
    .rq-pill.purple { background:rgba(124,58,237,.08); color:var(--accent); border-color:rgba(124,58,237,.2); }
    .rq-pill.cyan { background:rgba(8,145,178,.08); color:var(--accent2); border-color:rgba(8,145,178,.2); }
    .rq-pill.green { background:rgba(5,150,105,.08); color:var(--accent4); border-color:rgba(5,150,105,.2); }
    .rq-pill.pink { background:rgba(219,39,119,.08); color:var(--accent3); border-color:rgba(219,39,119,.2); }
    .rq-pill.orange { background:rgba(234,88,12,.08); color:var(--accent5); border-color:rgba(234,88,12,.2); }

    .panel h2.rq-title {
      font-size:22px; font-weight:900; margin:0 0 18px;
      display:flex; align-items:center; gap:10px;
    }

    /* ── FORM YENİDEN YAPILANMA ─────────────────────────── */
    form { display:block; }

    .search-query-row { margin-bottom:14px; }
    .search-query-row input {
      min-height:62px; font-size:17px; font-weight:500;
      border-width:2px; border-radius:14px;
      padding:14px 20px; letter-spacing:.01em;
    }

    .search-options-row {
      display:grid;
      grid-template-columns:1fr 1fr 1fr 1fr 52px;
      gap:10px; align-items:end;
    }
    .search-options-row button#submit {
      min-height:46px; border-radius:12px;
      font-size:22px; margin-top:0;
    }

    /* ── LOADING OVERLAY ─────────────────────────────────── */
    @keyframes spin { to { transform:rotate(360deg); } }
    @keyframes pulse-dot {
      0%,100% { opacity:.3; transform:scale(.7); }
      50% { opacity:1; transform:scale(1); }
    }

    #loading {
      display:none; flex-direction:column;
      align-items:center; justify-content:center;
      gap:18px; padding:48px 24px; text-align:center;
    }
    #loading.active { display:flex; }

    .loading-ring {
      position:relative; width:64px; height:64px;
    }
    .loading-ring svg {
      width:64px; height:64px;
      animation:spin 1.1s linear infinite;
    }
    .loading-ring circle {
      fill:none; stroke-width:5;
      stroke-linecap:round;
    }
    .loading-ring .ring-track { stroke:rgba(124,58,237,.12); }
    .loading-ring .ring-fill {
      stroke:url(#spinGrad);
      stroke-dasharray:120 200;
      stroke-dashoffset:0;
    }
    .loading-dots {
      display:flex; gap:8px; align-items:center;
    }
    .loading-dots span {
      width:8px; height:8px; border-radius:50%;
      display:inline-block; animation:pulse-dot 1.2s ease-in-out infinite;
    }
    .loading-dots span:nth-child(1){ background:var(--accent); animation-delay:0s; }
    .loading-dots span:nth-child(2){ background:var(--accent2); animation-delay:.2s; }
    .loading-dots span:nth-child(3){ background:var(--accent3); animation-delay:.4s; }
    .loading-text {
      font-size:18px; font-weight:800;
      background:var(--grad); -webkit-background-clip:text;
      -webkit-text-fill-color:transparent; background-clip:text;
    }
    .loading-sub { font-size:13px; color:var(--muted); margin-top:2px; }
    .results.faded { opacity:.35; pointer-events:none; transition:opacity .2s; }

    /* ── CHARTS ──────────────────────────────────────────── */
    .chart-box {
      background:#fff; border:1px solid var(--line);
      border-radius:16px; padding:20px 22px; margin-bottom:14px;
      box-shadow:0 2px 12px rgba(124,58,237,.06);
    }
    .chart-box-title {
      font-size:12px; font-weight:800; text-transform:uppercase;
      letter-spacing:.07em; color:var(--muted); margin-bottom:14px;
      display:flex; align-items:center; gap:7px;
    }
    .charts-row {
      display:grid; grid-template-columns:1fr 1fr;
      gap:14px; margin-bottom:14px;
    }
    @media(max-width:760px){ .charts-row{ grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <div>
        <div class="brand">
          <div class="brand-icon">&#128269;</div>
          <h1>Semantic Dataset Search</h1>
        </div>
        <p class="subtitle">Tezdeki semantic data discovery problemini canlı olarak test edin model, yöntem ve kaynak seçimi ile karşılaştırmalı analiz yapın.</p>
      </div>
      <div class="stats" id="stats"></div>
    </div>

    <div class="app-layout">
      <nav class="sidebar" role="tablist" aria-label="Main views">
        <div class="sidebar-label">&#128269; Gezinti</div>
        <button class="tab-btn" type="button" role="tab" aria-selected="true" aria-controls="tab-search" id="tab-btn-search" data-tab="search">
          <span class="tab-icon">&#128270;</span>
          <span class="tab-text">Arama<span class="tab-sub">Canlı Sorgula</span></span>
        </button>
        <div class="sidebar-label">&#128202; Arastirma</div>
        <button class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="tab-rq1" id="tab-btn-rq1" data-tab="rq1">
          <span class="tab-icon">&#9889;</span>
          <span class="tab-text">RQ1<span class="tab-sub">Yöntem Karşılaştırma</span></span>
        </button>
        <button class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="tab-rq2" id="tab-btn-rq2" data-tab="rq2">
          <span class="tab-icon">&#127760;</span>
          <span class="tab-text">RQ2<span class="tab-sub">Kaynaklar Arası</span></span>
        </button>
        <button class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="tab-rq3" id="tab-btn-rq3" data-tab="rq3">
          <span class="tab-icon">&#129302;</span>
          <span class="tab-text">RQ3<span class="tab-sub">Model Etkisi</span></span>
        </button>
        <button class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="tab-rq4" id="tab-btn-rq4" data-tab="rq4">
          <span class="tab-icon">&#128202;</span>
          <span class="tab-text">RQ4<span class="tab-sub">Açıklama Kalitesi</span></span>
        </button>
      </nav>
      <div class="content-area">

    <section class="tab-panel" id="tab-search" role="tabpanel" aria-labelledby="tab-btn-search">
      <section class="panel" aria-label="Search">
        <h2><span style="background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">&#128269;</span> Canlı Arama</h2>
          <p class="panel-note">Aynı sorguda farklı profil ve yöntemleri karşılaştırın model, kaynak ve reranking seçeneklerini değiştirerek sistem davranışını canlı gözlemleyin.</p>
        <form id="search-form">
          <div class="search-query-row">
            <label for="query">&#128270; Sorgu</label>
            <input id="query" name="query" autocomplete="off" placeholder="&#128269; Ne tür bir veri seti arıyorsunuz? Örneğin: tarihsel deprem olayları ve sismik aktivite verileri..." required>
          </div>
          <div class="search-options-row">
            <div>
              <label for="profile">&#129504; Profil</label>
              <select id="profile" name="profile"></select>
            </div>
            <div>
              <label for="method">&#9889; Yöntem</label>
              <select id="method" name="method">
                <option value="semantic" selected>Semantic</option>
                <option value="bm25">BM25</option>
                <option value="hybrid">Hybrid</option>
              </select>
            </div>
            <div>
              <label for="source">&#127760; Kaynak</label>
              <select id="source" name="source">
                <option value="all">Tumü</option>
                <option value="kaggle">Kaggle</option>
                <option value="huggingface">Hugging Face</option>
              </select>
            </div>
            <div>
              <label for="top-k">&#128200; Top K</label>
              <select id="top-k" name="top_k">
                <option value="5">Top 5</option>
                <option value="10">Top 10</option>
                <option value="15">Top 15</option>
              </select>
            </div>
            <div>
              <button id="submit" type="submit" title="Ara">&#128269;</button>
            </div>
          </div>
        </form>
        <div class="search-toggle" aria-label="Semantic rerank options">
          <label class="toggle" for="cross-encoder-toggle">
            <input id="cross-encoder-toggle" name="enable_cross_encoder" type="checkbox">
            <span>&#129504; Cross-encoder semantic rerank</span>
          </label>
        </div>
        <div class="examples" id="examples" aria-label="Sample queries">
          <button class="example-btn" type="button" data-query="I want to study how weather affects crop production.">&#127789; weather + crop</button>
          <button class="example-btn" type="button" data-query="I want to research diabetes prediction using patient records.">&#128137; diabetes</button>
          <button class="example-btn" type="button" data-query="I want to find a dataset of historical earthquake events and seismic activity records.">&#127979; earthquake</button>
        </div>
      </section>

      <!--
      <section class="panel" id="query-insight-panel" aria-label="Query Insight" hidden>
        <h2>Query Insight</h2>
        <p class="panel-note" id="query-insight-note">Sorgunun semantic planini, algılanan aspect'lerini ve evaluation set ile iliskisini burada görürsün.</p>
        <div class="rq1-grid" id="query-insight-grid"></div>
        <div class="rq1-banner" id="query-insight-banner"></div>
      </section>

      <section class="panel" aria-label="Live Evaluation Set">
        <h2>Live Evaluation Set</h2>
        <p class="panel-note">Canlı testlerde kullandığımız sorgular artık tekrar üretilebilir evaluation set olarak tutuluyor. Aynı sorguları farklı semantic profile'lar ile karşılaştırabilirsin.</p>
        <div class="rq1-grid" id="live-eval-grid"></div>
        <div class="examples" id="live-eval-queries" aria-label="Live evaluation queries"></div>
        <div class="rq1-banner" id="live-eval-banner"></div>
      </section>

      <section class="panel" aria-label="Retriever Adaptation">
        <h2>Retriever Adaptation</h2>
        <p class="panel-note">Curated hard-negative set, semantik olarak yakın ama yanlış veri setlerini ince ayar sırasında aşağı itmeyi hedefler. Bu panel son üretilen adaptation set özetini gösterir.</p>
        <div class="rq1-grid" id="adaptation-grid"></div>
        <div class="rq1-grid" id="adaptation-spec-grid"></div>
        <div class="rq1-banner" id="adaptation-banner"></div>
      </section>
      -->
      <section class="panel" id="query-insight-panel" aria-label="Query Insight" hidden>
        <p id="query-insight-note" hidden></p>
        <div class="rq1-grid" id="query-insight-grid" hidden></div>
        <div class="rq1-banner" id="query-insight-banner" hidden></div>
      </section>
      <div id="live-eval-grid" hidden></div>
      <div id="live-eval-queries" hidden></div>
      <div id="live-eval-banner" hidden></div>
      <div id="adaptation-grid" hidden></div>
      <div id="adaptation-spec-grid" hidden></div>
      <div id="adaptation-banner" hidden></div>

      <p id="status" class="status">&#9989; Sorgunuzu girin.</p>
      <div id="confidence-banner" class="rq1-banner" hidden></div>

      <div id="loading">
        <svg style="position:absolute;width:0;height:0">
          <defs>
            <linearGradient id="spinGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#7c3aed"/>
              <stop offset="100%" stop-color="#0891b2"/>
            </linearGradient>
          </defs>
        </svg>
        <div class="loading-ring">
          <svg viewBox="0 0 64 64">
            <circle class="ring-track" cx="32" cy="32" r="26"/>
            <circle class="ring-fill" cx="32" cy="32" r="26"/>
          </svg>
        </div>
        <div>
          <div class="loading-text">Arama yapılıyor...</div>
          <div class="loading-sub">Vektör veritabanı taranıyor</div>
        </div>
        <div class="loading-dots">
          <span></span><span></span><span></span>
        </div>
      </div>

      <section id="results" class="results" aria-live="polite">
        <div class="empty">&#128269; Yukarıdaki alana bir sorgu girin, sonuçlar burada listelenecek.</div>
      </section>
    </section>

    <section class="tab-panel" id="tab-rq1" role="tabpanel" aria-labelledby="tab-btn-rq1" hidden>
      <section class="panel" aria-label="RQ1">
        <h2 class="rq-title">&#9889; RQ1 Semantic, BM25'i Geçer mi?</h2>

        <div class="rq-cluster">
          <div class="rq-cluster-label purple">&#10067; Arastirma Sorusu</div>
          <div class="rq-cluster-body purple">
            Anlam tabanlı (dense) retrieval, kelime eşleme (BM25) yöntemini <strong>P@1, MRR ve nDCG@5</strong> metriklerinde anlamlı ölçüde geçiyor mu?
            <div style="margin-top:10px; font-size:12px; line-height:1.6; color:var(--muted)">
              <div><strong>P@1</strong> (Precision at 1): İlk sırada doğru veri seti çıkma oranı. 1.00 = her sorguda en üst sonuç doğru.</div>
              <div><strong>MRR</strong> (Mean Reciprocal Rank): Doğru sonucun sırasının tersinin ortalaması. Doğru cevap 1. sıradaysa 1.0, 2. sıradaysa 0.5, 3. sıradaysa 0.33... Doğru cevabın listede ne kadar yukarıda olduğunu özetler.</div>
              <div><strong>nDCG@5</strong> (Normalized Discounted Cumulative Gain): İlk 5 sonucun sıralama kalitesi. Hem doğru sonucun olup olmadığını hem de kaçıncı sırada olduğunu birlikte değerlendirir; 1.00 ideal sıralama.</div>
            </div>
          </div>
        </div>

        <div class="rq-cluster">
          <div class="rq-cluster-label cyan">&#9881; Yöntem</div>
          <div class="rq-cluster-body cyan">
            <strong>34 etiketli sorgu</strong>, 1.248 kayıtlık corpus üzerinde 3 yöntemle karşılaştırıldı.
            <div class="rq-pills" style="margin-top:10px">
              <span class="rq-pill cyan">BM25</span>
              <span class="rq-pill purple">Semantic (FAISS)</span>
              <span class="rq-pill green">Hybrid + Rerank</span>
            </div>
          </div>
        </div>

        <div class="rq-cluster">
          <div class="rq-cluster-label green">&#9989; Anahtar Bulgu</div>
          <div class="rq-cluster-body green">
            <strong>Hybrid kazandı.</strong> Cümle tipi sorgularda Semantic de BM25'u geçiyor; anlam tabanlı veri seti keşfi mümkün.
            <div class="rq-pills" style="margin-top:10px">
              <span class="rq-pill green">Hybrid P@1 = 0.91</span>
              <span class="rq-pill green">MRR = 0.95</span>
              <span class="rq-pill green">nDCG@5 = 0.79</span>
              <span class="rq-pill orange">BM25 P@1 = 0.74</span>
            </div>
          </div>
        </div>

        <div class="chart-box">
          <div class="chart-box-title">&#128202; Yöntem Karşılaştırması P@1 / MRR / nDCG@5</div>
          <canvas id="chart-rq1-methods" height="110"></canvas>
        </div>
        <div class="charts-row">
          <div class="chart-box">
            <div class="chart-box-title">&#128285; Keyword Sorgu Turu</div>
            <canvas id="chart-rq1-keyword" height="160"></canvas>
          </div>
          <div class="chart-box">
            <div class="chart-box-title">&#128172; Cümle Sorgu Turu</div>
            <canvas id="chart-rq1-sentence" height="160"></canvas>
          </div>
        </div>
        <div class="rq1-grid" id="rq1-overview"></div>
        <div class="rq1-grid" id="rq1-style-grid"></div>
        <div class="rq1-banner" id="rq1-banner"></div>
      </section>

      <section class="panel" id="rq1-live-panel" aria-label="RQ1 canlı karşılaştırma" hidden>
        <h2 class="rq-title">&#9889; Canlı Karşılaştırma</h2>
        <p class="panel-note" id="rq1-live-note">Arama sekmesinde girdiğiniz sorgu 3 yöntemle eşit koşullarda çalıştırıldı.</p>
        <div class="rq1-banner" id="rq1-guidance"></div>
        <div class="rq1-grid" id="rq1-live-grid"></div>
      </section>
    </section>

    <section class="tab-panel" id="tab-rq2" role="tabpanel" aria-labelledby="tab-btn-rq2" hidden>
      <section class="panel" aria-label="RQ2">
        <h2 class="rq-title">&#127760; RQ2 Kaggle &#8596; Hugging Face Koprusu</h2>

        <div class="rq-cluster">
          <div class="rq-cluster-label purple">&#10067; Arastirma Sorusu</div>
          <div class="rq-cluster-body purple">
            Kaggle'da aranan bir veri setinin karşılığı <strong>Hugging Face'te</strong> bulunabilir mi? Peki tersi? İki platform arasındaki anlamsal kopru ne kadar saglamdir?
          </div>
        </div>

        <div class="rq-cluster">
          <div class="rq-cluster-label cyan">&#9881; Yöntem</div>
          <div class="rq-cluster-body cyan">
            Manuel eslestirilen veri seti ciftleri, <strong>2 yönde</strong> (HF&#8594;Kaggle ve Kaggle&#8594;HF) test edildi.
            <div class="rq-pills" style="margin-top:10px">
              <span class="rq-pill cyan">6 Konu Kategorisi</span>
              <span class="rq-pill purple">2 Yön</span>
              <span class="rq-pill green">Hit@5 ve nDCG@5</span>
            </div>
          </div>
        </div>

        <div class="rq-cluster">
          <div class="rq-cluster-label green">&#9989; Anahtar Bulgu</div>
          <div class="rq-cluster-body green">
            <strong>Semantic ve Hybrid, BM25'u platformlar arası koprulamada acik ara geci.</strong> BM25 "Daily Updated" gibi platforma özgü kelimelere takilirken embedding modeller konunun ozunu yakalıyor.
          </div>
        </div>

        <div class="charts-row">
          <div class="chart-box">
            <div class="chart-box-title">&#127760; Platform Yönüne Göre nDCG@5</div>
            <canvas id="chart-rq2-direction" height="160"></canvas>
          </div>
          <div class="chart-box">
            <div class="chart-box-title">&#127919; Konuya Göre Hit@5</div>
            <canvas id="chart-rq2-topics" height="160"></canvas>
          </div>
        </div>
        <div class="rq1-grid" id="rq2-composition-grid"></div>
        <div class="rq1-grid" id="rq2-overview"></div>
        <div class="rq1-grid" id="rq2-direction-grid"></div>
        <div class="rq1-grid" id="rq2-topic-grid"></div>
        <div class="rq1-banner" id="rq2-banner"></div>
      </section>

      <section class="panel" id="rq2-live-panel" aria-label="RQ2 canlı karşılaştırma" hidden>
        <h2 class="rq-title">&#127760; Canlı Karşılaştırma</h2>
        <p class="panel-note" id="rq2-live-note">Kaynak filtresini tek platforma aldiginizdda cross-source davranış bu panelde izlenebilir.</p>
        <div class="rq1-banner" id="rq2-guidance"></div>
        <div class="rq1-grid" id="rq2-live-grid"></div>
      </section>
    </section>

    <section class="tab-panel" id="tab-rq3" role="tabpanel" aria-labelledby="tab-btn-rq3" hidden>
      <section class="panel" aria-label="RQ3">
        <h2 class="rq-title">&#129302; RQ3 Hangi Model Daha İyi?</h2>

        <div class="rq-cluster">
          <div class="rq-cluster-label purple">&#10067; Arastirma Sorusu</div>
          <div class="rq-cluster-body purple">
            Encoder model seçimi sonucu ne kadar değiştiriyor? <strong>MiniLM, E5, MiniLM-FT, Multilingual</strong> hangisi ne zaman kazanıyor?
          </div>
        </div>

        <div class="rq-cluster">
          <div class="rq-cluster-label cyan">&#9881; Test Edilen Modeller</div>
          <div class="rq-cluster-body cyan">
            4 model, aynı FAISS pipeline ve aynı sorgu seti üzerinde karşılaştırıldı.
            <div class="rq-pills" style="margin-top:10px">
              <span class="rq-pill purple">MiniLM EN Baseline</span>
              <span class="rq-pill cyan">E5 Base EN Retrieval</span>
              <span class="rq-pill pink">MiniLM-FT Fine-tuned</span>
              <span class="rq-pill green">Multilingual EN+TR</span>
            </div>
          </div>
        </div>

        <div class="rq-cluster">
          <div class="rq-cluster-label green">&#9989; Anahtar Bulgular</div>
          <div class="rq-cluster-body green">
            <strong>Tek kazanan yok</strong> dile ve açıklama tarzına göre değişiyor.
            <div class="rq-pills" style="margin-top:10px">
              <span class="rq-pill green">TR sorgu &#8594; Multilingual zorunlu</span>
              <span class="rq-pill cyan">EN uzun metin &#8594; E5 one çıkıyor</span>
              <span class="rq-pill purple">Genel EN &#8594; MiniLM yeterli</span>
              <span class="rq-pill pink">Yakın ayirim &#8594; MiniLM-FT iyileşme</span>
            </div>
          </div>
        </div>

        <div class="charts-row">
          <div class="chart-box">
            <div class="chart-box-title">&#129302; Dile Göre Model Performansi (Semantic nDCG@5)</div>
            <canvas id="chart-rq3-language" height="160"></canvas>
          </div>
          <div class="chart-box">
            <div class="chart-box-title">&#127775; Kesit Bazlı Karşılaştırma (Hybrid nDCG@5)</div>
            <canvas id="chart-rq3-slices" height="160"></canvas>
          </div>
        </div>
        <div class="rq1-grid" id="rq3-composition-grid"></div>
        <div class="rq1-grid" id="rq3-overview-grid"></div>
        <div class="rq1-grid" id="rq3-slice-grid"></div>
        <div class="rq1-grid" id="rq3-language-grid"></div>
        <div class="rq1-banner" id="rq3-banner"></div>
      </section>

      <section class="panel" id="rq3-live-panel" aria-label="RQ3 canlı karşılaştırma" hidden>
        <h2 class="rq-title">&#129302; Canlı Karşılaştırma</h2>
        <p class="panel-note" id="rq3-live-note">Profil değiştirip aynı sorguyu yeniden çalıştırarak modeller arası sıra farkı gözlemlenebilir.</p>
        <div class="rq1-banner" id="rq3-guidance"></div>
        <div class="rq1-grid" id="rq3-live-grid"></div>
      </section>
    </section>

    <section class="tab-panel" id="tab-rq4" role="tabpanel" aria-labelledby="tab-btn-rq4" hidden>
      <section class="panel" aria-label="RQ4">
        <h2 class="rq-title">&#128202; RQ4 Zayıf Açıklama = Kaybolan Veri Seti mi?</h2>

        <div class="rq-cluster">
          <div class="rq-cluster-label purple">&#10067; Arastirma Sorusu</div>
          <div class="rq-cluster-body purple">
            Açıklaması kısa ya da yalnizca etiket listesinden oluşan veri setleri, sistem tarafından <strong>hic bulunamıyor mu?</strong>
          </div>
        </div>

        <div class="rq-cluster">
          <div class="rq-cluster-label cyan">&#9881; Segmentasyon Kriterleri</div>
          <div class="rq-cluster-body cyan">
            1.248 kayıt 3 farklı eksende gruplandırıldı ve her grupta Hit@5 ayrı ayrı ölçüldü.
            <div class="rq-pills" style="margin-top:10px">
              <span class="rq-pill cyan">&#128207; Uzunluk: Kısa / Orta / Uzun</span>
              <span class="rq-pill purple">&#128221; Tarz: Anlatımsal / Karışık / Metadata</span>
              <span class="rq-pill pink">&#128269; Terim: Seyrek / Orta / Zengin</span>
            </div>
          </div>
        </div>

        <div class="rq-cluster">
          <div class="rq-cluster-label green">&#9989; Anahtar Bulgular</div>
          <div class="rq-cluster-body green">
            Açıklama kalitesi direkt retrieval başarısını etkiliyor. Başlıkların her zaman dahil edilmesiyle uzun-açıklama bucket'inda büyük iyileşme.
            <div class="rq-pills" style="margin-top:10px">
              <span class="rq-pill green">Uzun açıklama Hit@5 = 0.77</span>
              <span class="rq-pill orange">Kısa açıklama Hit@5 &asymp; 0.00</span>
              <span class="rq-pill green">Bug fix: 0.64 &#8594; 0.77</span>
            </div>
          </div>
        </div>

        <div class="chart-box">
          <div class="chart-box-title">&#128207; Açıklama Uzunluguna Göre Hit@5</div>
          <canvas id="chart-rq4-length" height="110"></canvas>
        </div>
        <div class="charts-row">
          <div class="chart-box">
            <div class="chart-box-title">&#128221; Anlatım Tarzına Göre Hit@5</div>
            <canvas id="chart-rq4-style" height="160"></canvas>
          </div>
          <div class="chart-box">
            <div class="chart-box-title">&#128269; Terim Zenginligine Göre Hit@5</div>
            <canvas id="chart-rq4-terms" height="160"></canvas>
          </div>
        </div>
        <div class="rq1-grid" id="rq4-composition-grid"></div>
        <div class="rq1-grid" id="rq4-style-grid"></div>
        <div class="rq1-grid" id="rq4-length-grid"></div>
        <div class="rq1-grid" id="rq4-term-grid"></div>
        <div class="rq1-banner" id="rq4-banner"></div>
      </section>

      <section class="panel" id="rq4-live-panel" aria-label="RQ4 canlı karşılaştırma" hidden>
        <h2 class="rq-title">&#128202; Canlı Karşılaştırma</h2>
        <p class="panel-note" id="rq4-live-note">Sonuçların açıklama uzunluğu ve tarzı bu panelde gösterilir; "title-deemphasized" rozeti izlenebilir.</p>
        <div class="rq1-banner" id="rq4-guidance"></div>
        <div class="rq1-grid" id="rq4-live-grid"></div>
      </section>
    </section>
      </div><!-- /.content-area -->
    </div><!-- /.app-layout -->
  </main>

  <div class="modal-backdrop" id="detail-modal" aria-hidden="true">
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div class="modal-head">
        <h2 id="modal-title">Detail</h2>
        <button class="close-btn" id="modal-close" type="button">Close</button>
      </div>
      <dl id="modal-meta"></dl>
      <section class="modal-section">
        <h3>Why this dataset?</h3>
        <div class="why-box" id="modal-why"></div>
      </section>
      <section class="modal-section">
        <h3>Semantic view</h3>
        <div class="why-box" id="modal-semantic-note"></div>
        <div class="modal-text" id="modal-semantic-text"></div>
      </section>
      <section class="modal-section">
        <h3>Quality flags</h3>
        <div class="flag-list" id="modal-flags"></div>
      </section>
      <section class="modal-section">
        <h3>Full description</h3>
        <div class="modal-text" id="modal-text"></div>
      </section>
    </div>
  </div>

  <script>
    const tabButtons = Array.from(document.querySelectorAll(".tab-btn"));
    const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
    const form = document.querySelector("#search-form");
    const queryInput = document.querySelector("#query");
    const profileInput = document.querySelector("#profile");
    const methodInput = document.querySelector("#method");
    const sourceInput = document.querySelector("#source");
    const topKInput = document.querySelector("#top-k");
    const crossEncoderToggle = document.querySelector("#cross-encoder-toggle");
    const submitButton = document.querySelector("#submit");
    const loadingEl = document.querySelector("#loading");
    const statusEl = document.querySelector("#status");
    const resultsEl = document.querySelector("#results");
    const statsEl = document.querySelector("#stats");
    const examplesEl = document.querySelector("#examples");
    const profileNoteEl = document.querySelector("#profile-note");
    const queryInsightPanelEl = document.querySelector("#query-insight-panel");
    const queryInsightNoteEl = document.querySelector("#query-insight-note");
    const queryInsightGridEl = document.querySelector("#query-insight-grid");
    const queryInsightBannerEl = document.querySelector("#query-insight-banner");
    const liveEvalGridEl = document.querySelector("#live-eval-grid");
    const liveEvalQueriesEl = document.querySelector("#live-eval-queries");
    const liveEvalBannerEl = document.querySelector("#live-eval-banner");
    const adaptationGridEl = document.querySelector("#adaptation-grid");
    const adaptationSpecGridEl = document.querySelector("#adaptation-spec-grid");
    const adaptationBannerEl = document.querySelector("#adaptation-banner");
    const rq1OverviewEl = document.querySelector("#rq1-overview");
    const rq1StyleEl = document.querySelector("#rq1-style-grid");
    const rq1BannerEl = document.querySelector("#rq1-banner");
    const rq1LivePanelEl = document.querySelector("#rq1-live-panel");
    const rq1GuidanceEl = document.querySelector("#rq1-guidance");
    const rq1LiveGridEl = document.querySelector("#rq1-live-grid");
    const rq1LiveNoteEl = document.querySelector("#rq1-live-note");
    const rq2CompositionEl = document.querySelector("#rq2-composition-grid");
    const rq2OverviewEl = document.querySelector("#rq2-overview");
    const rq2DirectionEl = document.querySelector("#rq2-direction-grid");
    const rq2TopicEl = document.querySelector("#rq2-topic-grid");
    const rq2BannerEl = document.querySelector("#rq2-banner");
    const rq2LivePanelEl = document.querySelector("#rq2-live-panel");
    const rq2GuidanceEl = document.querySelector("#rq2-guidance");
    const rq2LiveGridEl = document.querySelector("#rq2-live-grid");
    const rq2LiveNoteEl = document.querySelector("#rq2-live-note");
    const rq3CompositionEl = document.querySelector("#rq3-composition-grid");
    const rq3OverviewEl = document.querySelector("#rq3-overview-grid");
    const rq3SliceEl = document.querySelector("#rq3-slice-grid");
    const rq3LanguageEl = document.querySelector("#rq3-language-grid");
    const rq3BannerEl = document.querySelector("#rq3-banner");
    const rq3LivePanelEl = document.querySelector("#rq3-live-panel");
    const rq3GuidanceEl = document.querySelector("#rq3-guidance");
    const rq3LiveGridEl = document.querySelector("#rq3-live-grid");
    const rq3LiveNoteEl = document.querySelector("#rq3-live-note");
    const rq4CompositionEl = document.querySelector("#rq4-composition-grid");
    const rq4StyleEl = document.querySelector("#rq4-style-grid");
    const rq4LengthEl = document.querySelector("#rq4-length-grid");
    const rq4TermEl = document.querySelector("#rq4-term-grid");
    const rq4BannerEl = document.querySelector("#rq4-banner");
    const rq4LivePanelEl = document.querySelector("#rq4-live-panel");
    const rq4GuidanceEl = document.querySelector("#rq4-guidance");
    const rq4LiveGridEl = document.querySelector("#rq4-live-grid");
    const rq4LiveNoteEl = document.querySelector("#rq4-live-note");
    const detailModal = document.querySelector("#detail-modal");
    const modalTitle = document.querySelector("#modal-title");
    const modalMeta = document.querySelector("#modal-meta");
    const modalWhy = document.querySelector("#modal-why");
    const modalSemanticNote = document.querySelector("#modal-semantic-note");
    const modalSemanticText = document.querySelector("#modal-semantic-text");
    const modalFlags = document.querySelector("#modal-flags");
    const modalText = document.querySelector("#modal-text");
    const modalClose = document.querySelector("#modal-close");

    const methodLabels = {
      semantic: "Semantic",
      bm25: "BM25",
      hybrid: "Hybrid"
    };

    const sourceLabels = {
      all: "All sources",
      kaggle: "Kaggle",
      huggingface: "Hugging Face"
    };

    const directionLabels = {
      huggingface_to_kaggle: "Hugging Face -> Kaggle",
      kaggle_to_huggingface: "Kaggle -> Hugging Face"
    };

    const topicLabels = {
      amazon_reviews: "Amazon reviews",
      climate_change: "Climate change",
      movie_reviews: "Movie reviews",
      quran_text_audio: "Quran text and audio",
      sentiment_analysis: "Sentiment analysis",
      turkish_product_reviews: "Turkish product reviews"
    };

    const studySliceLabels = {
      english_main: "English main",
      cross_source: "Cross-source",
      tr_subset: "Turkish subset"
    };

    const languageLabels = {
      en: "English",
      tr: "Turkish"
    };

    const descriptionStyleLabels = {
      metadata_heavy: "Metadata heavy",
      mixed_structured: "Mixed structured",
      narrative: "Narrative"
    };

    const lengthBucketLabels = {
      short: "Short",
      medium: "Medium",
      long: "Long"
    };

    const termBucketLabels = {
      term_sparse: "Term sparse",
      term_moderate: "Term moderate",
      term_rich: "Term rich"
    };

    let currentResults = [];
    let currentMethod = "hybrid";
    let metadataPayload = null;
    let activeTab = "search";

    const escapeHtml = (value) => String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

    function formatScore(score) {
      const numeric = Number(score);
      return Number.isFinite(numeric) ? numeric.toFixed(4) : "-";
    }

    function formatList(value, emptyLabel = "-") {
      if (Array.isArray(value)) {
        return value.length ? value.join(", ") : emptyLabel;
      }
      return value || emptyLabel;
    }

    function formatProfileList(profiles, emptyLabel = "-") {
      if (!Array.isArray(profiles) || !profiles.length) return emptyLabel;
      return profiles.map((profileKey) => profileLabel(profileKey)).join(", ");
    }

    function qualityChipClass(status) {
      if (status === "risk") return "chip chip-risk";
      if (status === "warning") return "chip chip-warning";
      return "chip";
    }

    function renderQualityFlags(details) {
      if (!Array.isArray(details) || !details.length) {
        return '<div class="empty">Bu kayıt için aktif kalite flagi yok.</div>';
      }
      return details.map((detail) => `
        <article class="flag-item">
          <strong>${escapeHtml(detail.label || detail.code || "Flag")}</strong>
          <p>${escapeHtml(detail.message || "")}</p>
          <span class="flag-meta">${escapeHtml(detail.severity || "info")} / ${escapeHtml(detail.origin || "stored")}</span>
        </article>
      `).join("");
    }

    function scoreLabel(method) {
      if (method === "bm25") return "BM25";
      if (method === "hybrid") return "HYB";
      return "SEM";
    }

    function methodRole(method) {
      const roles = {
        semantic: "Açıklama tabanlı semantic retrieval",
        bm25: "Lexical baseline",
        hybrid: "Semantic + lexical birlikte"
      };
      return roles[method] || "";
    }

    function methodBadge(method) {
      return method === "hybrid" ? "En güçlü pratik sistem" : method === "semantic" ? "Semantic signal" : "Lexical signal";
    }

    function getProfileMeta(profileKey) {
      return metadataPayload?.profiles?.find((profile) => profile.key === profileKey) || null;
    }

    function renderStats(profileKey) {
      if (!metadataPayload) return;
      const stats = metadataPayload.profile_stats?.[profileKey] || metadataPayload.profile_stats?.[metadataPayload.default_profile] || {};
      const sources = stats.source_counts || {};
      const items = [
        ["Indexed", stats.indexed_rows ?? "-"],
        ["Kaggle", sources.kaggle ?? "-"],
        ["Hugging Face", sources.huggingface ?? "-"]
      ];
      statsEl.innerHTML = items.map(([label, value]) => `
        <div class="stat"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>
      `).join("");

      if (profileNoteEl) {
        const profile = getProfileMeta(profileKey);
        if (profile) {
          const retrievalHint = profileKey === "e5_base"
            ? "Retrieval-style query:/passage: prefixes aktif."
            : profileKey === "multilingual"
              ? "Çok dilli query:/passage: prefix akışı aktif."
              : "Structured semantic_text ana sinyal olarak kullanılıyor.";
          const rerankHint = crossEncoderToggle?.checked
            ? " Cross-encoder semantic rerank acik; top adaylar ince query-document alignment ile tekrar siralaniyor."
            : " Cross-encoder semantic rerank kapali.";
          profileNoteEl.textContent = `${profile.label}: ${profile.description} ${retrievalHint} Düşük-bilgi kayıtlarda title etkisi semantic kalite notu ile azaltıldı.${rerankHint}`;
        }
      }
    }

    function findMethodRow(rows, method) {
      return (rows || []).find((row) => row.method === method) || null;
    }

    function findProfileMethodRow(rows, profile, method) {
      return (rows || []).find((row) => row.profile === profile && row.method === method) || null;
    }

    function profileLabel(profileKey) {
      return getProfileMeta(profileKey)?.label || profileKey || "-";
    }

    function formatDirectionLabel(direction) {
      return directionLabels[direction] || direction || "-";
    }

    function formatTopicLabel(topic) {
      if (topicLabels[topic]) return topicLabels[topic];
      return String(topic || "-")
        .split("_")
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
    }

    function formatStudySliceLabel(slice) {
      return studySliceLabels[slice] || slice || "-";
    }

    function formatLanguageLabel(language) {
      return languageLabels[language] || language || "-";
    }

    function formatDescriptionStyleLabel(style) {
      return descriptionStyleLabels[style] || style || "-";
    }

    function formatLengthBucketLabel(bucket) {
      return lengthBucketLabels[bucket] || bucket || "-";
    }

    function formatTermBucketLabel(bucket) {
      return termBucketLabels[bucket] || bucket || "-";
    }

    function normalizeLooseText(value) {
      return String(value || "")
        .normalize("NFKD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .replace(/\s+/g, " ")
        .trim();
    }

    function findLiveEvalQuery(queryText) {
      const normalized = normalizeLooseText(queryText);
      return (metadataPayload?.live_eval?.queries || []).find((row) => normalizeLooseText(row.query_text) === normalized) || null;
    }

    function countObjectValues(value) {
      return Object.values(value || {}).reduce((total, item) => total + Number(item || 0), 0);
    }

    function activateTab(tabKey) {
      activeTab = tabKey;
      tabButtons.forEach((button) => {
        const isActive = button.dataset.tab === tabKey;
        button.setAttribute("aria-selected", isActive ? "true" : "false");
      });
      tabPanels.forEach((panel) => {
        panel.hidden = panel.id !== `tab-${tabKey}`;
      });
    }

    function rq1MetricCard(label, value) {
      return `
        <div class="rq1-metric">
          <strong>${escapeHtml(value)}</strong>
          <span>${escapeHtml(label)}</span>
        </div>
      `;
    }

    function renderRQ1Overview() {
      const rq1 = metadataPayload?.rq1;
      if (!rq1) return;

      const methods = ["bm25", "semantic", "hybrid"];
      const overallCards = methods.map((method) => {
        const row = findMethodRow(rq1.overall, method);
        if (!row) return "";
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(methodBadge(method))}</span>
            <h3>${escapeHtml(methodLabels[method])}</h3>
            <p>${escapeHtml(methodRole(method))}</p>
            <div class="rq1-metrics">
              ${rq1MetricCard("nDCG@5", Number(row.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("MRR", Number(row.mean_mrr).toFixed(3))}
              ${rq1MetricCard("P@1", Number(row.mean_precision_at_1).toFixed(3))}
            </div>
          </article>
        `;
      }).join("");
      rq1OverviewEl.innerHTML = overallCards;

      const keywordRows = rq1.by_query_style?.keyword || [];
      const sentenceRows = rq1.by_query_style?.sentence || [];
      const keywordHybrid = findMethodRow(keywordRows, "hybrid");
      const keywordBm25 = findMethodRow(keywordRows, "bm25");
      const sentenceHybrid = findMethodRow(sentenceRows, "hybrid");
      const sentenceSemantic = findMethodRow(sentenceRows, "semantic");
      const sentenceBm25 = findMethodRow(sentenceRows, "bm25");

      rq1StyleEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Keyword Queries</span>
          <h4>Kısa ve anahtar kelime odakli aramalar</h4>
          <p>BM25 burada hala anlamlı bir baseline. Ama genel ranking kalitesinde hybrid one geçiyor.</p>
          <div class="rq1-metrics">
            ${keywordBm25 ? rq1MetricCard("BM25 MRR", Number(keywordBm25.mean_mrr).toFixed(3)) : ""}
            ${keywordHybrid ? rq1MetricCard("Hybrid nDCG@5", Number(keywordHybrid.mean_ndcg_at_k).toFixed(3)) : ""}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Sentence Queries</span>
          <h4>Doğal dil ve ihtiyac anlatan cümleler</h4>
          <p>RQ1'in asil kazanci burada. Semantic ve özellikle hybrid, BM25'ten daha güçlü davranıyor.</p>
          <div class="rq1-metrics">
            ${sentenceSemantic ? rq1MetricCard("Semantic nDCG@5", Number(sentenceSemantic.mean_ndcg_at_k).toFixed(3)) : ""}
            ${sentenceBm25 ? rq1MetricCard("BM25 nDCG@5", Number(sentenceBm25.mean_ndcg_at_k).toFixed(3)) : ""}
            ${sentenceHybrid ? rq1MetricCard("Hybrid nDCG@5", Number(sentenceHybrid.mean_ndcg_at_k).toFixed(3)) : ""}
          </div>
        </article>
      `;

      rq1BannerEl.innerHTML = `
        <strong>RQ1 özet sonucu</strong>
        <p>Tezdeki ana İngilizce benchmarkta <b>hybrid</b> en güçlü sistem oldu. Ancak sentence-style sorgularda salt <b>semantic retrieval</b> bile BM25'i geçiyor. Bu da veri seti açıklamalarıyla semantic discovery yapılabildiğini gösteriyor.</p>
      `;
    }

    function selectTopicInsight(rq2, kind, excludeTopics = []) {
      const entries = Object.entries(rq2?.by_topic || {}).map(([topic, rows]) => {
        const semantic = findMethodRow(rows, "semantic");
        const bm25 = findMethodRow(rows, "bm25");
        const hybrid = findMethodRow(rows, "hybrid");
        let score = Number.NEGATIVE_INFINITY;
        if (kind === "strong") {
          score = semantic?.mean_ndcg_at_k ?? hybrid?.mean_ndcg_at_k ?? score;
        } else if (kind === "hard") {
          score = -(semantic?.mean_ndcg_at_k ?? hybrid?.mean_ndcg_at_k ?? Number.POSITIVE_INFINITY);
        } else if (kind === "gap") {
          score = (semantic?.mean_ndcg_at_k ?? Number.NEGATIVE_INFINITY) - (bm25?.mean_ndcg_at_k ?? 0);
        }
        return { topic, rows, semantic, bm25, hybrid, score };
      });

      return entries
        .filter((entry) => !excludeTopics.includes(entry.topic) && Number.isFinite(entry.score))
        .sort((left, right) => right.score - left.score)[0] || null;
    }

    function renderRQ2Overview() {
      const rq2 = metadataPayload?.rq2;
      if (!rq2) return;

      rq2CompositionEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Test seti buyuklugu</span>
          <h4>${escapeHtml(String(rq2.query_count ?? "-"))} eşleşme çifti</h4>
          <p>Manuel olarak eşleştirilmiş (Kaggle, Hugging Face) veri seti ciftleri. Her cift bir test sorgusu üretir.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Yön sayısı", String(Object.keys(rq2.direction_counts || {}).length))}
            ${rq1MetricCard("Konu sayısı", String(Object.keys(rq2.topic_counts || {}).length))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Yön dağılımı</span>
          <h4>İki yön de test ediliyor</h4>
          <p>Her iki yöne de bakmasak, sadece bir platformun açıklama tarzına özgü sonuç çıkarmış olurduk.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("HF'den Kaggle'a", String(rq2.direction_counts?.huggingface_to_kaggle ?? "-"))}
            ${rq1MetricCard("Kaggle'dan HF'ye", String(rq2.direction_counts?.kaggle_to_huggingface ?? "-"))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Konu cesitliligi</span>
          <h4>${escapeHtml(String(countObjectValues(rq2.topic_counts)))} toplam eşleşme</h4>
          <p>Birden fazla konu var ki sonucumuz tek bir alana özgü çıkmasın.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("En genis konu", formatTopicLabel(rq2.largest_topic || "-"))}
            ${rq1MetricCard("En dar konu", formatTopicLabel(rq2.smallest_topic || "-"))}
          </div>
        </article>
      `;

      const methods = ["bm25", "semantic", "hybrid"];
      rq2OverviewEl.innerHTML = methods.map((method) => {
        const row = findMethodRow(rq2.overall, method);
        if (!row) return "";
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(methodBadge(method))}</span>
            <h3>${escapeHtml(methodLabels[method])}</h3>
            <p>İki platform arası sorgularda ${escapeHtml(methodRole(method))} davranışı.</p>
            <div class="rq1-metrics">
              ${rq1MetricCard("Top-5'te yakalama", Number(row.mean_hit_rate_at_k).toFixed(3))}
              ${rq1MetricCard("Siralama kalitesi (nDCG@5)", Number(row.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("Top-1 isabet", Number(row.mean_precision_at_1).toFixed(3))}
            </div>
          </article>
        `;
      }).join("");

      const directions = ["huggingface_to_kaggle", "kaggle_to_huggingface"];
      rq2DirectionEl.innerHTML = directions.map((direction) => {
        const rows = rq2.by_direction?.[direction] || [];
        const semantic = findMethodRow(rows, "semantic");
        const bm25 = findMethodRow(rows, "bm25");
        const hybrid = findMethodRow(rows, "hybrid");
        const best = [...rows].sort((left, right) => (right.mean_ndcg_at_k ?? 0) - (left.mean_ndcg_at_k ?? 0))[0];
        if (!best) return "";
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(formatDirectionLabel(direction))}</span>
            <h4>Bu yönde en iyi yöntem</h4>
            <p>Bu yönde sorguladığımızda en yüksek skoru <b>${escapeHtml(methodLabels[best.method] || best.method)}</b> alıyor. Platformların açıklama stili (Kaggle daha pazarlamaya yakın, HF daha teknik) sonucu etkiliyor.</p>
            <div class="rq1-metrics">
              ${semantic ? rq1MetricCard("Semantic siralama", Number(semantic.mean_ndcg_at_k).toFixed(3)) : ""}
              ${hybrid ? rq1MetricCard("Hybrid Top-5'te yakalama", Number(hybrid.mean_hit_rate_at_k).toFixed(3)) : ""}
              ${bm25 ? rq1MetricCard("BM25 siralama", Number(bm25.mean_ndcg_at_k).toFixed(3)) : ""}
            </div>
          </article>
        `;
      }).join("");

      const strongest = selectTopicInsight(rq2, "strong");
      const hardest = selectTopicInsight(rq2, "hard", strongest ? [strongest.topic] : []);
      const gap = selectTopicInsight(
        rq2,
        "gap",
        [strongest?.topic, hardest?.topic].filter(Boolean)
      );
      const topicInsights = [
        strongest && {
          label: "En kolay konu",
          title: formatTopicLabel(strongest.topic),
          copy: "Bu konuda iki platformun veri seti açıklamaları ortak terim/yapı kullanıyor. Embedding modeli konunun kendisini yakaladığı için platformlar arası geçiş sorunsuz oluyor.",
          metrics: [
            strongest.semantic ? ["Semantic siralama", Number(strongest.semantic.mean_ndcg_at_k).toFixed(3)] : null,
            strongest.hybrid ? ["Hybrid Top-5'te yakalama", Number(strongest.hybrid.mean_hit_rate_at_k).toFixed(3)] : null
          ].filter(Boolean)
        },
        hardest && {
          label: "En zor konu",
          title: formatTopicLabel(hardest.topic),
          copy: "Bu konuda platformların yazım tarzları birbirinden çok ayrışıyor. BM25 anahtar kelime eşitliği bulamıyor; semantic ve hybrid ancak kısmen toparlıyor.",
          metrics: [
            hardest.semantic ? ["Semantic siralama", Number(hardest.semantic.mean_ndcg_at_k).toFixed(3)] : null,
            hardest.bm25 ? ["BM25 siralama", Number(hardest.bm25.mean_ndcg_at_k).toFixed(3)] : null
          ].filter(Boolean)
        },
        gap && {
          label: "Anlamsal aramanın en çok fark yarattigi konu",
          title: formatTopicLabel(gap.topic),
          copy: "BM25'in (anahtar kelime) yetersiz kaldigi, anlamsal aramanın (Semantic/Hybrid) acik fark yarattigi konu. RQ2'nin tez argümanı burada en net görülüyor.",
          metrics: [
            gap.semantic ? ["Semantic siralama", Number(gap.semantic.mean_ndcg_at_k).toFixed(3)] : null,
            gap.bm25 ? ["BM25 siralama", Number(gap.bm25.mean_ndcg_at_k).toFixed(3)] : null
          ].filter(Boolean)
        }
      ].filter(Boolean);

      rq2TopicEl.innerHTML = topicInsights.map((insight) => `
        <article class="rq1-card">
          <span class="rq1-label">${escapeHtml(insight.label)}</span>
          <h4>${escapeHtml(insight.title)}</h4>
          <p>${escapeHtml(insight.copy)}</p>
          <div class="rq1-metrics">
            ${insight.metrics.map(([label, value]) => rq1MetricCard(label, value)).join("")}
          </div>
        </article>
      `).join("");

      rq2BannerEl.innerHTML = `
        <strong>RQ2 özet</strong>
        <p>İki platform arasında doğru veri setini bulma işinde anlamsal arama (Semantic ve Hybrid), anahtar kelime aramasından (BM25) belirgin şekilde daha başarılı. Tezin sorusuna doğrudan cevap: <b>evet, embedding'ler kaynaktan bağımsız olarak konu temsilini taşıyor</b>. Uygulamada en yüksek skoru veren konfigürasyon Hybrid; ancak konuya ve yöne göre değişiyor.</p>
      `;
    }

    function renderRQ3Overview() {
      const rq3 = metadataPayload?.rq3;
      if (!rq3) return;
      const liveProfiles = (metadataPayload?.profiles || []).filter((profile) => profile.is_ready);
      const benchmarkProfiles = rq3.benchmark_profiles || [];

      rq3CompositionEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Test seti buyuklugu</span>
          <h4>${escapeHtml(String(rq3.query_count ?? "-"))} test sorgusu</h4>
          <p>Tüm modeller bu aynı sorgu listesi üzerinde çalıştırıldı. Aynı sorguyu farklı modellerle deneyince adil karşılaştırma yapılabiliyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Alt kesit sayısı", String(Object.keys(rq3.study_slice_counts || {}).length))}
            ${rq1MetricCard("Dil sayısı", String(Object.keys(rq3.language_counts || {}).length))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Tezde ölçülmüş modeller</span>
          <h4>${escapeHtml(String(benchmarkProfiles.length || 0))} model raporlu</h4>
          <p>Tezde ölçülmüş karşılaştırma raporlari su an MiniLM ve çok-dilli model için mevcut. Ek modeller (E5 Base, MiniLM-FT) canlı arama tarafında test edilebiliyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Modeller", formatProfileList(benchmarkProfiles))}
            ${rq1MetricCard("Varsayılan", profileLabel(metadataPayload.default_profile))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Canlı arama için hazır modeller</span>
          <h4>${escapeHtml(String(liveProfiles.length || 0))} model</h4>
          <p>Live Search alanından model değiştirip aynı sorguyu tekrar çalıştırın; sıra değişikliğini görerek model etkisini canlı olarak izleyin.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Hazır", String(liveProfiles.length || 0))}
            ${rq1MetricCard("İngilizce sorgu", String(rq3.language_counts?.en ?? "-"))}
            ${rq1MetricCard("Türkçe sorgu", String(rq3.language_counts?.tr ?? "-"))}
          </div>
        </article>
      `;

      rq3OverviewEl.innerHTML = ["hybrid", "semantic"].map((method) => {
        const minilmRow = findProfileMethodRow(rq3.overall, "minilm", method);
        const multilingualRow = findProfileMethodRow(rq3.overall, "multilingual", method);
        if (!minilmRow || !multilingualRow) return "";
        const winner = (minilmRow.mean_ndcg_at_k ?? 0) >= (multilingualRow.mean_ndcg_at_k ?? 0) ? "minilm" : "multilingual";
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(methodLabels[method])}</span>
            <h4>Genel ortalamada ${escapeHtml(profileLabel(winner))} onde</h4>
            <p>Aynı test setinde iki modelin genel siralama kalitesi (nDCG@5) karşılaştırılıyor. Yüksek olan, sonuçları daha iyi sırada çıkarıyor.</p>
            <div class="rq1-metrics">
              ${rq1MetricCard("MiniLM siralama", Number(minilmRow.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("Multilingual siralama", Number(multilingualRow.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("One çıkan", winner === "minilm" ? "MiniLM" : "Multilingual")}
            </div>
          </article>
        `;
      }).join("");

      rq3SliceEl.innerHTML = ["english_main", "cross_source", "tr_subset"].map((slice) => {
        const minilmHybrid = findProfileMethodRow(rq3.by_study_slice?.[slice], "minilm", "hybrid");
        const multilingualHybrid = findProfileMethodRow(rq3.by_study_slice?.[slice], "multilingual", "hybrid");
        const minilmSemantic = findProfileMethodRow(rq3.by_study_slice?.[slice], "minilm", "semantic");
        const multilingualSemantic = findProfileMethodRow(rq3.by_study_slice?.[slice], "multilingual", "semantic");
        if (!minilmHybrid || !multilingualHybrid || !minilmSemantic || !multilingualSemantic) return "";
        const winner = (minilmSemantic.mean_ndcg_at_k ?? 0) >= (multilingualSemantic.mean_ndcg_at_k ?? 0) ? "minilm" : "multilingual";
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(formatStudySliceLabel(slice))}</span>
            <h4>Bu kesitte ${escapeHtml(profileLabel(winner))} semantic'te onde</h4>
            <p>Bu sorgu kesitinde (orn. yalnız İngilizce, iki kaynak arası, ya da Türkçe alt küme) hangi modelin daha iyi sıraladığını gösteriyor.</p>
            <div class="rq1-metrics">
              ${rq1MetricCard("MiniLM semantic", Number(minilmSemantic.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("Multilingual semantic", Number(multilingualSemantic.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("MiniLM hybrid", Number(minilmHybrid.mean_ndcg_at_k).toFixed(3))}
            </div>
          </article>
        `;
      }).join("");

      rq3LanguageEl.innerHTML = ["en", "tr"].map((language) => {
        const minilmRow = findProfileMethodRow(rq3.by_language?.[language], "minilm", "semantic");
        const multilingualRow = findProfileMethodRow(rq3.by_language?.[language], "multilingual", "semantic");
        if (!minilmRow || !multilingualRow) return "";
        const winner = (minilmRow.mean_ndcg_at_k ?? 0) >= (multilingualRow.mean_ndcg_at_k ?? 0) ? "minilm" : "multilingual";
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(formatLanguageLabel(language))}</span>
            <h4>${escapeHtml(profileLabel(winner))} bu dil için daha uygun</h4>
            <p>Sorgu dili değiştikçe hangi modelin daha iyi çalıştığı değişiyor RQ3'un en pratik mesaji burada.</p>
            <div class="rq1-metrics">
              ${rq1MetricCard("MiniLM semantic", Number(minilmRow.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("Multilingual semantic", Number(multilingualRow.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("Tercih", winner === "minilm" ? "MiniLM" : "Multilingual")}
            </div>
          </article>
        `;
      }).join("");

      rq3BannerEl.innerHTML = `
        <strong>RQ3 özet</strong>
        <p>Tek bir model her durumda kazanmiyor: tezdeki ölçülmüş karşılaştırmada İngilizce ana kesitte <b>MiniLM</b> yeterli, Türkçe alt kümede <b>çok-dilli model</b> belirgin şekilde gerekli. Canlı aramada ek olarak <b>E5 Base</b> (retrieval için eğitilmiş İngilizce model) ve alan-uyarlanmis <b>MiniLM-FT</b> aynı sorguyu denemek için hazır. Pratik tavsiye: İngilizce için MiniLM ile basla, sorgu Türkçe ise multilingual'a geç.</p>
      `;
    }

    function renderRQ4Overview() {
      const rq4 = metadataPayload?.rq4;
      if (!rq4) return;

      const semanticLong = findMethodRow(rq4.by_length?.long, "semantic");
      const semanticShort = findMethodRow(rq4.by_length?.short, "semantic");
      const semanticNarrative = findMethodRow(rq4.by_style?.narrative, "semantic");
      const semanticMetadata = findMethodRow(rq4.by_style?.metadata_heavy, "semantic");
      const semanticRich = findMethodRow(rq4.by_term?.term_rich, "semantic");
      const semanticSparse = findMethodRow(rq4.by_term?.term_sparse, "semantic");
      const hybridSparse = findMethodRow(rq4.by_term?.term_sparse, "hybrid");
      const hybridMixed = findMethodRow(rq4.by_style?.mixed_structured, "hybrid");

      rq4CompositionEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Test seti buyuklugu</span>
          <h4>${escapeHtml(String(rq4.query_count ?? "-"))} İngilizce sorgu</h4>
          <p>RQ4, mevcut test sorgularını bu kez "veri seti açıklaması nasil yazılmış" perspektifinden tekrar yorumluyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("İngilizce ana kesit", String(rq4.study_slice_counts?.english_main ?? "-"))}
            ${rq1MetricCard("İki kaynak arası", String(rq4.study_slice_counts?.cross_source ?? "-"))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Eşikler (kac kelime?)</span>
          <h4>Kısa / orta / uzun siniri</h4>
          <p>Açıklamayı uzunluk ve terim zenginligine göre ayirirken kullanılan eşikler. Bu sayede "kısa açıklamalı veri setleri" gibi bir gruba bakabiliyoruz.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Kısa &le;", String(rq4.thresholds?.length_word_thresholds?.short_max ?? "-"))}
            ${rq1MetricCard("Orta &le;", String(rq4.thresholds?.length_word_thresholds?.medium_max ?? "-"))}
            ${rq1MetricCard("Zayıf terim &le;", String(rq4.thresholds?.term_thresholds?.sparse_max ?? "-"))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Corpus dağılımı</span>
          <h4>Açıklama tarzları dengeli</h4>
          <p>Corpus'ta bu uc tarz da bulunuyor; sonuç tek bir uc gruba bağlı kalmadi. Yorumlama için yeterli temsil var.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Etiket ağırlıklı", `${Math.round((rq4.distribution?.description_style?.metadata_heavy ?? 0) * 100)}%`)}
            ${rq1MetricCard("Karışık", `${Math.round((rq4.distribution?.description_style?.mixed_structured ?? 0) * 100)}%`)}
            ${rq1MetricCard("Anlatımsal", `${Math.round((rq4.distribution?.description_style?.narrative ?? 0) * 100)}%`)}
          </div>
        </article>
      `;

      rq4StyleEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Anlatım tarzı etkisi</span>
          <h4>Cümlelerle yazılmış açıklamalar daha iyi çalışıyor</h4>
          <p>Cümlelerle anlatılmış (anlatımsal) açıklamalar embedding'e daha çok bağlam taşıyor. Sadece etiket listesi gibi yazılmış açıklamalar semantic sinyali zayıflatıyor; bu durumda Hybrid daha güvenli kaliyor.</p>
          <div class="rq1-metrics">
            ${semanticNarrative ? rq1MetricCard("Anlatımsal Top-5'te yakalama", Number(semanticNarrative.hit_rate_at_k).toFixed(3)) : ""}
            ${semanticMetadata ? rq1MetricCard("Etiket ağırlıklı Top-5'te yakalama", Number(semanticMetadata.hit_rate_at_k).toFixed(3)) : ""}
            ${hybridMixed ? rq1MetricCard("Hybrid karışık Top-1 isabet", Number(hybridMixed.top1_rate).toFixed(3)) : ""}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Açıklama uzunluğu etkisi</span>
          <h4>Uzun açıklamalar daha kolay yakalanıyor</h4>
          <p>Açıklama uzadikca embedding modeline daha fazla bağlam düşüyor; sistem o veri setini sırada yukarı çıkarabiliyor. Kısa açıklamalı veri setlerinde semantic ve hybrid'in ikisi de zorlaniyor.</p>
          <div class="rq1-metrics">
            ${semanticLong ? rq1MetricCard("Uzun Top-5'te yakalama", Number(semanticLong.hit_rate_at_k).toFixed(3)) : ""}
            ${semanticShort ? rq1MetricCard("Kısa Top-5'te yakalama", Number(semanticShort.hit_rate_at_k).toFixed(3)) : ""}
            ${semanticLong ? rq1MetricCard("Uzun Top-1 isabet", Number(semanticLong.top1_rate).toFixed(3)) : ""}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Terim zenginligi etkisi</span>
          <h4>Daha çeşitli kelime kullanan açıklamalar daha iyi temsil ediliyor</h4>
          <p>Aciklamada ne kadar farklı içerikli kelime gecerse semantic eşleşme o kadar güçlü oluyor. Kelime kısmir kalan veri setlerinde Hybrid'in BM25 yari daha iyi sonuç veriyor yani anahtar-kelime araması devreye girip "elinde ne varsa onunla bul" diyor.</p>
          <div class="rq1-metrics">
            ${semanticRich ? rq1MetricCard("Zengin Top-5'te yakalama", Number(semanticRich.hit_rate_at_k).toFixed(3)) : ""}
            ${semanticSparse ? rq1MetricCard("Zayıf Top-5'te yakalama", Number(semanticSparse.hit_rate_at_k).toFixed(3)) : ""}
            ${hybridSparse ? rq1MetricCard("Hybrid zayıf Top-5", Number(hybridSparse.hit_rate_at_k).toFixed(3)) : ""}
          </div>
        </article>
      `;

      rq4LengthEl.innerHTML = "";
      rq4TermEl.innerHTML = "";

      rq4BannerEl.innerHTML = `
        <strong>RQ4 özet</strong>
        <p>Sonucun kalitesi sadece seçilen modele bağlı değil corpus'taki <b>açıklamaların yazım kalitesi</b> de en az model kadar belirleyici. Pratik mesaj: <b>uzun, anlatımsal ve terim zengini</b> açıklamalar embedding'i güçlendiriyor; açıklama zayıf oldugunda Hybrid (semantic + BM25) bir güvenlik ağı olarak devreye giriyor. Bu turda eklenen başlık-her-zaman-indekste fixi sayesinde "mushrooms" gibi tek-kelime açıklamalı veri setleri de artık bulunabilir.</p>
      `;
    }

    function renderLiveEvalOverview() {
      const liveEval = metadataPayload?.live_eval;
      if (!liveEval) return;

      const readyReports = Object.values(liveEval.profile_exports || {}).filter((row) => row?.available);
      const queryLanguages = Array.from(new Set((liveEval.queries || []).map((row) => row.language).filter(Boolean)));
      const topQueryButtons = (liveEval.queries || []).slice(0, 9).map((row) => `
        <button class="example-btn" type="button" data-query="${escapeHtml(row.query_text)}">${escapeHtml(row.query_id || row.language || "query")}</button>
      `).join("");

      liveEvalGridEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Query Set</span>
          <h4>${escapeHtml(String(liveEval.query_count || 0))} evaluation query</h4>
          <p>Canlı semantic arama denemeleri artık sabit bir evaluation slice olarak saklaniyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Languages", queryLanguages.join(", ") || "-")}
            ${rq1MetricCard("Profiles exported", String(readyReports.length))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Manual Label Ready</span>
          <h4>Top-10 export yapısı hazır</h4>
          <p>CSV/JSON export dosyalari sonraki adimda 0/1/2 relevance etiketi ile doldurulabilecek şekilde tutuluyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Fields", "query + rank + score")}
            ${rq1MetricCard("Review", "manual_relevance")}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Profiles</span>
          <h4>${escapeHtml(formatProfileList(readyReports.map((row) => row.profile)))}</h4>
          <p>Aynı query set, baseline ve geliştirilmiş semantic profile'larla tekrar kosulabiliyor.</p>
          <div class="rq1-metrics">
            ${readyReports[0] ? rq1MetricCard("Top K", String(readyReports[0].top_k || "-")) : ""}
            ${readyReports.some((row) => row.enable_cross_encoder) ? rq1MetricCard("Cross-encoder", "available") : rq1MetricCard("Cross-encoder", "optional")}
          </div>
        </article>
      `;
      liveEvalQueriesEl.innerHTML = topQueryButtons || '<div class="empty">Evaluation query set bulunamadı.</div>';
      liveEvalBannerEl.innerHTML = `
        <strong>Thesis evaluation için neden önemli?</strong>
        <p>Bu query set, semantic arama iyilestirmelerini canlı goze dayalı yorumdan çıkarıp tekrarlanabilir bir karşılaştırmaya cevirir. MiniLM, E5 Base ve diger profile'lar aynı sorgular üzerinde ölçülebilir.</p>
      `;
    }

    function renderAdaptationOverview() {
      const adaptation = metadataPayload?.retriever_adaptation;
      if (!adaptation) return;

      const summary = adaptation.summary || {};
      const specs = adaptation.query_specs || [];
      adaptationGridEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Triples</span>
          <h4>${escapeHtml(String(summary.triples ?? 0))} hard-negative triple</h4>
          <p>Sentence-transformers fine-tuning için hazırlanan query, positive_text, negative_text satirlari.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Queries", String(summary.queries ?? 0))}
            ${rq1MetricCard("Unique positives", String(summary.unique_positive_refs ?? 0))}
            ${rq1MetricCard("Unique negatives", String(summary.unique_negative_refs ?? 0))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Selection Quality</span>
          <h4>Curated positive güveni</h4>
          <p>Strong ve weak pozitif ayrımı, corpus'taki eksik veya zayıf açıklamaları daha temkinli kullanmak için saklaniyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Strong", String(summary.positive_selection_quality?.strong ?? 0))}
            ${rq1MetricCard("Weak", String(summary.positive_selection_quality?.weak ?? 0))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Negative Logic</span>
          <h4>Yanlış ama yakın adaylar</h4>
          <p>Semantic komsuluk içinde kalan ama query'nin tüm aspect'lerini karşılamayan veri setleri hard negative olarak seçiliyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Neg clue", String(summary.negative_reasons?.negative_clue_match ?? 0))}
            ${rq1MetricCard("Partial overlap", String(summary.negative_reasons?.partial_semantic_overlap ?? 0))}
            ${rq1MetricCard("Missing aspects", String(summary.negative_reasons?.missing_query_aspects ?? 0))}
          </div>
        </article>
      `;
      adaptationSpecGridEl.innerHTML = specs.slice(0, 4).map((spec) => `
        <article class="rq1-card">
          <span class="rq1-label">${escapeHtml(spec.query_id || "spec")}</span>
          <h4>${escapeHtml(spec.query_text || "-")}</h4>
          <p><strong>Positive clues:</strong> ${escapeHtml(formatList(spec.core_positive_clues || spec.positive_clues || []))}</p>
          <p><strong>Hard negatives:</strong> ${escapeHtml(formatList(spec.hard_negative_clues || []))}</p>
        </article>
      `).join("");
      adaptationBannerEl.innerHTML = `
        <strong>Adaptation mantigi</strong>
        <p>Bu set, semantic retriever'a "price" ile "stock price" veya "urban change" ile generic image dataset arasındaki farkı ogretmek için eklendi. Amaç keyword'e geçmek değil; semantik ayirim gücünü arttırmak.</p>
      `;
    }

    function renderQueryInsight(payload) {
      // Query Insight panel is intentionally hidden in the trimmed UI.
      if (queryInsightPanelEl) queryInsightPanelEl.hidden = true;
      return;

      // Legacy render kept below in case the panel is re-enabled.
      const plan = payload?.query_plan;
      if (!plan) {
        queryInsightPanelEl.hidden = true;
        return;
      }

      const liveEvalMatch = findLiveEvalQuery(payload.query);
      const aspects = plan.semantic_aspects || [];
      const variants = plan.semantic_variants || [];
      const domains = plan.domains || [];
      const conceptPhrases = plan.concept_phrases || [];
      const topResult = (payload.results || [])[0] || null;
      queryInsightPanelEl.hidden = false;
      queryInsightNoteEl.textContent = "Bu özet, query understanding ve multi-aspect semantic scoring tarafında hangi sinyallerin aktif olduğunu gösterir.";
      queryInsightGridEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Intent</span>
          <h4>${escapeHtml(plan.concise_intent || payload.query || "-")}</h4>
          <p>Sorgunun temizlenmis ve retrieval odakli semantic niyet ifadesi.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Language", formatLanguageLabel(plan.detected_language))}
            ${rq1MetricCard("Sentence query", plan.is_sentence_query ? "yes" : "no")}
            ${rq1MetricCard("Multi-aspect", plan.has_multi_aspect_intent ? "yes" : "no")}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Aspects</span>
          <h4>${escapeHtml(aspects.map((aspect) => aspect.label || aspect.key).join(", ") || "-")}</h4>
          <p>Multi-part query'lerde aday veri setlerinin birden fazla semantic aspect'i kapsayıp kapsamadigi ayrıca skorlanır.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Aspect count", String(aspects.length))}
            ${rq1MetricCard("Domains", formatList(domains))}
            ${rq1MetricCard("Concept phrases", String(conceptPhrases.length))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Variants</span>
          <h4>${escapeHtml(String(variants.length || 0))} semantic variant</h4>
          <p>Original query, cleaned intent ve domain/task odakli semantic variants birlikte encode edilir.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Top-1 aspect coverage", topResult?.aspect_coverage != null ? formatScore(topResult.aspect_coverage) : "-")}
            ${rq1MetricCard("Top-1 aspect hits", topResult?.semantic_aspect_hits ?? "-")}
            ${rq1MetricCard("Top-1 variant hits", topResult?.semantic_variant_hits ?? "-")}
          </div>
        </article>
      `;

      const expectedNote = liveEvalMatch ? `Beklenen domain: ${formatList(liveEvalMatch.expected_domain)}. Beklenen task: ${formatList(liveEvalMatch.expected_task)}. Beklenen modality: ${formatList(liveEvalMatch.expected_modality)}.` : "Bu sorgu curated live evaluation set içinde değil; yalnizca query-plan sinyalleri gösteriliyor.";
      const failureNote = liveEvalMatch ? ` Bilinen failure pattern: ${formatList(liveEvalMatch.known_failure_patterns)}.` : "";
      const aspectNote = aspects.length
        ? ` Aspect texts: ${aspects.map((aspect) => aspect.text).join(" | ")}.`
        : "";
      queryInsightBannerEl.innerHTML = `
        <strong>Query-plan yorumu</strong>
        <p>${escapeHtml(expectedNote + failureNote + aspectNote)}</p>
      `;
    }

    function populateProfileOptions() {
      const profiles = metadataPayload?.profiles || [];
      const fallbackProfile = profiles.find((profile) => profile.is_ready)?.key || metadataPayload.default_profile;
      profileInput.innerHTML = profiles.map((profile) => `
        <option value="${escapeHtml(profile.key)}" ${profile.key === fallbackProfile ? "selected" : ""} ${profile.is_ready ? "" : "disabled"}>
          ${escapeHtml(profile.label)}${profile.is_ready ? "" : " (index missing)"}
        </option>
      `).join("");
    }

    function similarityExplanation(method, score, result = null) {
      const formattedScore = formatScore(score);
      if (method === "hybrid") {
        if (result?.cross_encoder_model) {
          return `Bu sonuç hybrid aday havuzu içinden cross-encoder semantic rerank ile tekrar hizalandi. Hybrid skor ${formattedScore}.`;
        }
        return `Bu sonuç semantic ve BM25 skorlarinin normalize edilip birleştirilmesiyle geldi. Hybrid skor ${formattedScore}.`;
      }
      if (method === "bm25") {
        return `Bu sonuç başlık, açıklama ve anahtar kelime eşleşmeleri güçlü olduğu için listelendi. BM25 skoru ${formattedScore}.`;
      }
      if (result?.cross_encoder_model) {
        return `Bu sonuç önce FAISS semantic retrieval ile aday olarak geldi, sonra query ve dataset açıklaması cross-encoder ile semantik olarak tekrar eşleştirildi. Semantic skor ${formattedScore}.`;
      }
      return `Bu sonuç sorgu ile dataset açıklaması anlamsal olarak yakın olduğu için listelendi. Semantic skor ${formattedScore}.`;
    }

    function renderConfidenceBanner(results) {
      const banner = document.querySelector("#confidence-banner");
      if (!banner) return;
      if (!results || !results.length) {
        banner.hidden = true;
        banner.innerHTML = "";
        return;
      }
      const top = results[0] || {};
      const confidence = top.query_confidence || "moderate";
      const isOod = !!top.query_is_ood;
      const weakCount = results.filter((r) => r.weak_match).length;
      if (confidence === "strong" && weakCount === 0) {
        banner.hidden = true;
        banner.innerHTML = "";
        return;
      }
      let headline;
      let body;
      const allWeak = weakCount === results.length;
      const onlyOne = results.length === 1;
      if (allWeak && onlyOne && isOod) {
        headline = "Corpus bu sorguyu kapsamiyor";
        body = "Sorgudaki ana topik kelimesi corpus'taki hiçbir veri setinde yok. Aşağıdaki tek sonuç, embedding uzayında en yakın komsudur; gerçek bir ilgi garantisi değildir. Curated query expansion vocabulary'sine veya corpus'a yeni kayıt eklemeden bu sorgu için daha iyi bir sonuç çıkması beklenmemelidir.";
      } else if (confidence === "weak") {
        headline = "Zayıf eşleşme";
        body = isOod
          ? "Bu sorgu için curated semantic kurallara düşen bir alan yok ve corpus'taki en yakın kayıtlar bile cosine olarak düşük skor verdi. Sonuçlar yalnizca yakınlık sirasidir; gerçek bir ilgi garantisi vermez."
          : "En iyi adayin retrieval skoru zayıf eşik altında. Sonuçları hedef veri seti olarak değil, en yakın komsular olarak okuyun.";
      } else {
        headline = "Orta seviye güven";
        body = isOod
          ? `Bu sorgu için domain/concept sinyali yok; ${weakCount} sonuç weak-match olarak isaretlendi. Top-1 score yorumu için bu rozeti dikkate alin.`
          : `${weakCount} sonuç weak-match olarak isaretlendi. Yorumda bu rozetleri dikkate alin.`;
      }
      banner.hidden = false;
      banner.innerHTML = `<strong>${headline}</strong><p>${body}</p>`;
    }

    function renderResults(results, method) {
      currentResults = results;
      currentMethod = method;

      renderConfidenceBanner(results);

      if (!results.length) {
        resultsEl.innerHTML = '<div class="empty">Bu sorgu için sonuç bulunamadı.</div>';
        return;
      }

      resultsEl.innerHTML = results.map((result, index) => {
        const text = String(result.semantic_summary || result.description || result.text || "");
        const summary = text.length > 360 ? `${text.slice(0, 360)}...` : text;
        const url = result.url ? `<a href="${escapeHtml(result.url)}" target="_blank" rel="noreferrer">Open source page</a>` : "URL yok";
        const semanticRiskNote = result.semantic_quality_note
          ? `<p class="score-note">${escapeHtml(result.semantic_quality_note)}</p>`
          : "";
        return `
          <article class="result">
            <div class="result-head">
              <h2>${index + 1}. ${escapeHtml(result.title || "Isimsiz veri seti")}</h2>
              <span class="score">${scoreLabel(method)} ${formatScore(result.score)}</span>
            </div>
            <div class="meta">
              <span class="chip">${escapeHtml((result.source || "unknown").toUpperCase())}</span>
              <span class="chip">${escapeHtml(result.ref || "ref yok")}</span>
              ${result.weak_match ? '<span class="chip chip-risk">Weak match</span>' : ""}
              ${result.title_deemphasized ? '<span class="chip chip-warning">Title de-emphasized</span>' : ""}
              ${result.cross_encoder_enabled && !result.cross_encoder_model && result.cross_encoder_error ? '<span class="chip chip-warning">Cross-encoder unavailable</span>' : ""}
              ${result.quality_flag_count ? `<span class="${qualityChipClass(result.quality_status)}">${escapeHtml(result.quality_flag_summary || "Flag var")}</span>` : ""}
            </div>
            <p class="summary">${escapeHtml(summary)}</p>
            ${semanticRiskNote}
            <div class="actions">
              <button class="detail-btn" type="button" data-result-index="${index}">Detail</button>
              ${url}
            </div>
          </article>
        `;
      }).join("");
    }

    function renderRQ1Live(payload) {
      const rq1 = payload.rq1;
      if (!rq1) {
        rq1LivePanelEl.hidden = true;
        return;
      }

      rq1LivePanelEl.hidden = false;
      rq1LiveNoteEl.textContent = `Bu panel, aynı sorgunun BM25, Semantic ve Hybrid ile nasil davrandığını canlı olarak gösterir. Algılanan sorgu tipi: ${rq1.query_style_label}.`;
      rq1GuidanceEl.innerHTML = `
        <strong>Yöntem önerisi</strong>
        <p>${escapeHtml(rq1.guidance_text || "")}</p>
      `;

      const cards = (rq1.snapshots || []).map((snapshot) => {
        const top = snapshot.top_result;
        const benchmark = snapshot.benchmark_metrics;
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(methodBadge(snapshot.method))}</span>
            <h3>${escapeHtml(methodLabels[snapshot.method] || snapshot.method)}</h3>
            <p>${escapeHtml(methodRole(snapshot.method))}</p>
            <div class="rq1-metrics">
              ${benchmark ? rq1MetricCard("Benchmark nDCG@5", Number(benchmark.mean_ndcg_at_k).toFixed(3)) : ""}
              ${benchmark ? rq1MetricCard("Benchmark P@1", Number(benchmark.mean_precision_at_1).toFixed(3)) : ""}
              ${top ? rq1MetricCard("Live top score", formatScore(top.score)) : ""}
            </div>
            <p><strong>Top-1:</strong> ${escapeHtml(top?.title || "Sonuç yok")}</p>
            <p>${escapeHtml(top?.source ? top.source.toUpperCase() : "")}${top?.ref ? ` / ${top.ref}` : ""}</p>
          </article>
        `;
      }).join("");
      rq1LiveGridEl.innerHTML = cards;
    }

    function renderRQ2Live(payload) {
      const rq2 = payload.rq2;
      if (!rq2) {
        rq2LivePanelEl.hidden = true;
        return;
      }

      rq2LivePanelEl.hidden = false;
      if (!rq2.enabled) {
        rq2LiveNoteEl.textContent = "Canlı cross-source yorum için hedef kaynağı tek platforma indir.";
        rq2GuidanceEl.innerHTML = `
          <strong>RQ2 canlı modu hazır</strong>
          <p>Source filtresini <b>Kaggle</b> ya da <b>Hugging Face</b> olarak seç. Boylece sistem, sorguyu diger platformdan bu hedef kaynağa kopruleme senaryosu olarak yorumlayıp ilgili yön benchmarki ile karşılaştırır.</p>
        `;
        rq2LiveGridEl.innerHTML = `
          <article class="rq1-card">
            <span class="rq1-label">How To Use</span>
            <h3>Target source seç</h3>
            <p><b>Kaggle</b> seçildiğinde canlı panel <b>Hugging Face -> Kaggle</b> yönünü, <b>Hugging Face</b> seçildiğinde ise <b>Kaggle -> Hugging Face</b> yönünü baz alir.</p>
            <div class="rq1-metrics">
              ${rq1MetricCard("Current mode", "All sources")}
              ${rq1MetricCard("Needed", "Single source")}
            </div>
          </article>
        `;
        return;
      }

      rq2LiveNoteEl.textContent = `Seçilen hedef kaynak ${rq2.target_source_label}. Bu kurulum RQ2 benchmarkindaki ${rq2.direction_label} yönüne en yakın canlı senaryodur.`;
      rq2GuidanceEl.innerHTML = `
        <strong>Cross-source yorum</strong>
        <p>${escapeHtml(rq2.guidance_text || "")}</p>
      `;

      rq2LiveGridEl.innerHTML = (rq2.snapshots || []).map((snapshot) => {
        const top = snapshot.top_result;
        const benchmark = snapshot.benchmark_metrics;
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(methodBadge(snapshot.method))}</span>
            <h3>${escapeHtml(methodLabels[snapshot.method] || snapshot.method)}</h3>
            <p>${escapeHtml(methodRole(snapshot.method))}</p>
            <div class="rq1-metrics">
              ${benchmark ? rq1MetricCard("Benchmark nDCG@5", Number(benchmark.mean_ndcg_at_k).toFixed(3)) : ""}
              ${benchmark ? rq1MetricCard("Benchmark Bridge@5", Number(benchmark.mean_hit_rate_at_k).toFixed(3)) : ""}
              ${top ? rq1MetricCard("Live top score", formatScore(top.score)) : ""}
            </div>
            <p><strong>Top-1:</strong> ${escapeHtml(top?.title || "Sonuç yok")}</p>
            <p>${escapeHtml(top?.source ? top.source.toUpperCase() : "")}${top?.ref ? ` / ${top.ref}` : ""}</p>
          </article>
        `;
      }).join("");
    }

    function renderRQ3Live(payload) {
      const rq3 = payload.rq3;
      if (!rq3) {
        rq3LivePanelEl.hidden = true;
        return;
      }

      rq3LivePanelEl.hidden = false;
      rq3LiveNoteEl.textContent = `Algılanan sorgu dili ${rq3.language_label}. Aktif kesit: ${rq3.primary_slice_label}. Karşılaştırma: ${profileLabel(rq3.selected_profile)} vs ${profileLabel(rq3.alternative_profile)}. ${rq3.benchmark_scope === "paired" ? "Depolanmış benchmark metrikleri mevcut." : "Bu karşılaştırma canlı profil davranışı üzerinden okunmali."}`;
      rq3GuidanceEl.innerHTML = `
        <strong>Model yorumu</strong>
        <p>${escapeHtml(rq3.guidance_text || "")}</p>
      `;

      rq3LiveGridEl.innerHTML = (rq3.snapshots || []).map((snapshot) => {
        const top = snapshot.top_result;
        const primary = snapshot.primary_metrics;
        const language = snapshot.language_metrics;
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(snapshot.is_selected ? "Active profile" : "Comparison profile")}</span>
            <h3>${escapeHtml(snapshot.profile_label || snapshot.profile)}</h3>
            <p>${escapeHtml(snapshot.profile_description || "")}</p>
            <div class="rq1-metrics">
              ${primary ? rq1MetricCard("Slice nDCG@5", Number(primary.mean_ndcg_at_k).toFixed(3)) : rq1MetricCard("Slice nDCG@5", "live only")}
              ${language ? rq1MetricCard("Lang nDCG@5", Number(language.mean_ndcg_at_k).toFixed(3)) : rq1MetricCard("Lang nDCG@5", "live only")}
              ${top ? rq1MetricCard("Live top score", formatScore(top.score)) : ""}
            </div>
            <p><strong>Top-1:</strong> ${escapeHtml(top?.title || "Sonuç yok")}</p>
            <p>${escapeHtml(top?.source ? top.source.toUpperCase() : "")}${top?.ref ? ` / ${top.ref}` : ""}</p>
          </article>
        `;
      }).join("");
    }

    function renderRQ4Live(payload) {
      const rq4 = payload.rq4;
      if (!rq4) {
        rq4LivePanelEl.hidden = true;
        return;
      }

      rq4LivePanelEl.hidden = false;
      rq4LiveNoteEl.textContent = `Canlı kalite yorumu ${rq4.compare_method_label} methodu etrafında kuruluyor. Algılanan sorgu dili: ${rq4.language_label}.`;
      rq4GuidanceEl.innerHTML = `
        <strong>Belge kalite yorumu</strong>
        <p>${escapeHtml(rq4.guidance_text || "")}</p>
      `;

      rq4LiveGridEl.innerHTML = (rq4.snapshots || []).map((snapshot) => {
        const top = snapshot.top_result;
        const lengthMetrics = snapshot.length_metrics;
        const styleMetrics = snapshot.style_metrics;
        const termMetrics = snapshot.term_metrics;
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(methodLabels[snapshot.method] || snapshot.method)}</span>
            <h3>${escapeHtml(top?.title || "Sonuç yok")}</h3>
            <p>${escapeHtml(snapshot.description_style_label || "-")} / ${escapeHtml(snapshot.length_bucket_label || "-")} / ${escapeHtml(snapshot.term_bucket_label || "-")}</p>
            <div class="rq1-metrics">
              ${lengthMetrics ? rq1MetricCard("Len Hit@5", Number(lengthMetrics.hit_rate_at_k).toFixed(3)) : ""}
              ${styleMetrics ? rq1MetricCard("Style Hit@5", Number(styleMetrics.hit_rate_at_k).toFixed(3)) : ""}
              ${termMetrics ? rq1MetricCard("Term Hit@5", Number(termMetrics.hit_rate_at_k).toFixed(3)) : ""}
            </div>
            <p><strong>Top-1:</strong> ${escapeHtml(top?.source ? top.source.toUpperCase() : "")}${top?.ref ? ` / ${top.ref}` : ""}</p>
          </article>
        `;
      }).join("");
    }

    function openDetail(index) {
      const result = currentResults[index];
      if (!result) return;

      modalTitle.textContent = result.title || "Isimsiz veri seti";
      const urlValue = result.url
        ? `<a href="${escapeHtml(result.url)}" target="_blank" rel="noreferrer">${escapeHtml(result.url)}</a>`
        : "URL yok";
      const metaItems = [
        ["Başlık", result.title || "Isimsiz veri seti"],
        ["Profile", result.profile_label || "-"],
        ["Method", methodLabels[currentMethod] || currentMethod],
        ["Skor", formatScore(result.score)],
        ["Kaynak", (result.source || "unknown").toUpperCase()],
        ["Referans", result.ref || "ref yok"],
        ["URL", urlValue],
        ["Keywords", formatList(result.keywords)],
        ["Domains", formatList(result.inferred_domains)],
        ["Use cases", formatList(result.inferred_use_cases)],
        ["Modalities", formatList(result.inferred_modalities)],
        ["Language hint", result.language_hint || "-"],
        ["License", result.license || "-"],
        ["Description length", `${result.description_len_chars ?? "-"} chars / ${result.description_len_words ?? "-"} words`],
        ["Quality status", result.quality_flag_summary || "Flag yok"]
      ];
      if (currentMethod === "hybrid") {
        metaItems.push(["Semantic component", formatScore(result.semantic_score)]);
        metaItems.push(["BM25 component", formatScore(result.bm25_score)]);
      }
      modalMeta.innerHTML = metaItems.map(([label, value]) => `
        <dt>${escapeHtml(label)}</dt><dd>${label === "URL" ? value : escapeHtml(value)}</dd>
      `).join("");
      modalWhy.textContent = similarityExplanation(currentMethod, result.score, result);
      modalSemanticNote.textContent = result.semantic_quality_note || "Structured semantic_text önce description, topic, use case, modality ve keywords sinyalini kullanır; başlık tek basina belirleyici değildir.";
      modalSemanticText.textContent = result.semantic_text || result.semantic_summary || result.description || result.text || "";
      modalFlags.innerHTML = renderQualityFlags(result.quality_flag_details);
      modalText.textContent = result.description || result.text || "";
      detailModal.classList.add("open");
      detailModal.setAttribute("aria-hidden", "false");
      modalClose.focus();
    }

    function closeDetail() {
      detailModal.classList.remove("open");
      detailModal.setAttribute("aria-hidden", "true");
    }

    async function loadMetadata() {
      const response = await fetch("/api/metadata");
      if (!response.ok) throw new Error("Metadata yüklenemedi.");
      metadataPayload = await response.json();
      populateProfileOptions();
      renderStats(profileInput.value || metadataPayload.default_profile);
      renderLiveEvalOverview();
      renderAdaptationOverview();
      renderRQ1Overview();
      renderRQ2Overview();
      renderRQ3Overview();
      renderRQ4Overview();
    }

    resultsEl.addEventListener("click", (event) => {
      const button = event.target.closest(".detail-btn");
      if (!button) return;
      openDetail(Number(button.dataset.resultIndex));
    });

    modalClose.addEventListener("click", closeDetail);
    detailModal.addEventListener("click", (event) => {
      if (event.target === detailModal) closeDetail();
    });

    tabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        activateTab(button.dataset.tab);
        if (button.dataset.tab === "search") {
          queryInput.focus();
        }
      });
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeDetail();
    });

    examplesEl.addEventListener("click", (event) => {
      const button = event.target.closest(".example-btn");
      if (!button) return;
      queryInput.value = button.dataset.query || "";
      form.requestSubmit();
    });

    liveEvalQueriesEl.addEventListener("click", (event) => {
      const button = event.target.closest(".example-btn");
      if (!button) return;
      queryInput.value = button.dataset.query || "";
      form.requestSubmit();
    });

    profileInput.addEventListener("change", () => {
      renderStats(profileInput.value);
    });

    crossEncoderToggle.addEventListener("change", () => {
      renderStats(profileInput.value);
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const query = queryInput.value.trim();
      const profile = profileInput.value;
      const method = methodInput.value;
      const source = sourceInput.value;
      const topK = Number(topKInput.value);
      const enableCrossEncoder = Boolean(crossEncoderToggle.checked);
      if (!query) return;

      submitButton.disabled = true;
      statusEl.className = "status";
      statusEl.textContent = "";
      loadingEl.classList.add("active");
      resultsEl.classList.add("faded");

      try {
        const response = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            profile,
            method,
            source,
            top_k: topK,
            enable_cross_encoder: enableCrossEncoder
          })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Arama tamamlanamadi.");

        renderQueryInsight(payload);
        renderResults(payload.results || [], payload.method);
        renderRQ1Live(payload);
        renderRQ2Live(payload);
        renderRQ3Live(payload);
        renderRQ4Live(payload);
        renderStats(payload.profile.key);
        const sourceLabel = sourceLabels[payload.source] || "All sources";
        const conciseIntent = payload.query_plan?.concise_intent;
        const intentNote = conciseIntent ? ` Intent: ${conciseIntent}.` : "";
        const rerankNote = enableCrossEncoder ? " Cross-encoder semantic rerank acik." : "";
        statusEl.textContent = `${payload.profile.label} / ${methodLabels[payload.method] || payload.method} / ${sourceLabel} ile "${payload.query}" için ${payload.results.length} sonuç listelendi.${intentNote}${rerankNote}`;
      } catch (error) {
        statusEl.className = "status error";
        statusEl.textContent = error.message;
      } finally {
        submitButton.disabled = false;
        loadingEl.classList.remove("active");
        resultsEl.classList.remove("faded");
      }
    });


    // ── CHART RENDERING ──────────────────────────────────────────
    const CC = {
      purple: { bg:'rgba(124,58,237,.78)', border:'#7c3aed' },
      cyan: { bg:'rgba(8,145,178,.78)', border:'#0891b2' },
      green: { bg:'rgba(5,150,105,.78)', border:'#059669' },
      orange: { bg:'rgba(234,88,12,.78)', border:'#ea580c' },
      pink: { bg:'rgba(219,39,119,.78)', border:'#db2777' },
      amber: { bg:'rgba(245,158,11,.78)', border:'#f59e0b' },
    };
    const MCLR = { bm25:CC.orange, semantic:CC.purple, hybrid:CC.green };
    const chartReg = {};

    const BASE_OPTS = {
      responsive:true, maintainAspectRatio:true,
      plugins:{
        legend:{ position:'top', labels:{ usePointStyle:true, padding:14,
          font:{ size:11, weight:'700' }, color:'#1e1147' } },
        tooltip:{ cornerRadius:8, padding:10,
          titleFont:{ weight:'700' }, bodyFont:{ size:12 } }
      },
      scales:{
        x:{ grid:{ display:false }, ticks:{ font:{ size:11 }, color:'#6b5fa0' } },
        y:{ grid:{ color:'rgba(124,58,237,.07)' },
            ticks:{ font:{ size:11 }, color:'#6b5fa0' }, min:0, max:1 }
      }
    };

    function mkChart(id, type, data, extraOpts={}) {
      const el = document.getElementById(id);
      if (!el) return;
      if (chartReg[id]) chartReg[id].destroy();
      const opts = JSON.parse(JSON.stringify(BASE_OPTS));
      if (type === 'bar' && extraOpts.indexAxis === 'y') {
        opts.scales = {
          x:{ grid:{ color:'rgba(124,58,237,.07)' }, ticks:{ font:{ size:11 }, color:'#6b5fa0' }, min:0, max:1 },
          y:{ grid:{ display:false }, ticks:{ font:{ size:11 }, color:'#6b5fa0' } }
        };
      }
      if (extraOpts.noLegend) { opts.plugins.legend.display = false; delete extraOpts.noLegend; }
      Object.assign(opts, extraOpts);
      chartReg[id] = new Chart(el.getContext('2d'), { type, data, options: opts });
    }

    function drawRQ1Charts() {
      const rq1 = metadataPayload?.rq1; if (!rq1) return;
      const methods = ['bm25','semantic','hybrid'];
      const mLabel = { bm25:'BM25', semantic:'Semantic', hybrid:'Hybrid' };
      // Ana metrik karşılaştırması
      mkChart('chart-rq1-methods','bar',{
        labels:['P@1','MRR','nDCG@5'],
        datasets: methods.map(m => {
          const r = findMethodRow(rq1.overall, m);
          return { label: mLabel[m],
            data: r ? [+r.mean_precision_at_1, +r.mean_mrr, +r.mean_ndcg_at_k] : [0,0,0],
            backgroundColor: MCLR[m].bg, borderColor: MCLR[m].border,
            borderWidth:2, borderRadius:8 };
        })
      });
      // Keyword sorgu turu
      const kRows = rq1.by_query_style?.keyword || [];
      mkChart('chart-rq1-keyword','bar',{
        labels: methods.map(m=>mLabel[m]),
        datasets:[{ label:'nDCG@5',
          data: methods.map(m=>{ const r=findMethodRow(kRows,m); return r?+r.mean_ndcg_at_k:0; }),
          backgroundColor: methods.map(m=>MCLR[m].bg),
          borderColor: methods.map(m=>MCLR[m].border),
          borderWidth:2, borderRadius:8 }]
      },{ noLegend:true });
      // Cümle sorgu turu
      const sRows = rq1.by_query_style?.sentence || [];
      mkChart('chart-rq1-sentence','bar',{
        labels: methods.map(m=>mLabel[m]),
        datasets:[{ label:'nDCG@5',
          data: methods.map(m=>{ const r=findMethodRow(sRows,m); return r?+r.mean_ndcg_at_k:0; }),
          backgroundColor: methods.map(m=>MCLR[m].bg),
          borderColor: methods.map(m=>MCLR[m].border),
          borderWidth:2, borderRadius:8 }]
      },{ noLegend:true });
    }

    function drawRQ2Charts() {
      const rq2 = metadataPayload?.rq2; if (!rq2) return;
      const dirs = ['huggingface_to_kaggle','kaggle_to_huggingface'];
      const dLbl = ['HF → Kaggle','Kaggle → HF'];
      const methods = ['bm25','semantic','hybrid'];
      const mLabel = { bm25:'BM25', semantic:'Semantic', hybrid:'Hybrid' };
      mkChart('chart-rq2-direction','bar',{
        labels: dLbl,
        datasets: methods.map(m=>({
          label: mLabel[m],
          data: dirs.map(d=>{ const r=findMethodRow(rq2.by_direction?.[d]||[],m); return r?+r.mean_ndcg_at_k:0; }),
          backgroundColor: MCLR[m].bg, borderColor: MCLR[m].border,
          borderWidth:2, borderRadius:8 }))
      });
      const topics = Object.keys(rq2.by_topic||{});
      mkChart('chart-rq2-topics','bar',{
        labels: topics.map(t=>formatTopicLabel(t).substring(0,13)),
        datasets:[
          { label:'Semantic Hit@5',
            data: topics.map(t=>{ const r=findMethodRow(rq2.by_topic[t],'semantic'); return r?+(r.mean_hit_rate_at_k??r.mean_ndcg_at_k):0; }),
            backgroundColor:CC.purple.bg, borderColor:CC.purple.border, borderWidth:2, borderRadius:6 },
          { label:'Hybrid Hit@5',
            data: topics.map(t=>{ const r=findMethodRow(rq2.by_topic[t],'hybrid'); return r?+(r.mean_hit_rate_at_k??r.mean_ndcg_at_k):0; }),
            backgroundColor:CC.green.bg, borderColor:CC.green.border, borderWidth:2, borderRadius:6 }
        ]
      });
    }

    function drawRQ3Charts() {
      const rq3 = metadataPayload?.rq3; if (!rq3) return;
      const profiles = ['minilm','multilingual'];
      const pClr = [CC.purple, CC.cyan];
      const langs = ['en','tr'];
      const lLbl = ['🇬🇧 EN','🇹🇷 TR'];
      mkChart('chart-rq3-language','bar',{
        labels: lLbl,
        datasets: profiles.map((p,i)=>({
          label: profileLabel(p),
          data: langs.map(lang=>{
            const rows = rq3.by_language?.[lang]||[];
            const r = findProfileMethodRow(rows,p,'semantic') || findProfileMethodRow(rows,'semantic',p);
            return r?+r.mean_ndcg_at_k:0;
          }),
          backgroundColor:pClr[i].bg, borderColor:pClr[i].border, borderWidth:2, borderRadius:8 }))
      });
      const slices = ['english_main','cross_source','tr_subset'];
      const sLbl = ['EN Ana','2 Kaynak','TR Alt'];
      mkChart('chart-rq3-slices','bar',{
        labels: sLbl,
        datasets: profiles.map((p,i)=>({
          label: profileLabel(p),
          data: slices.map(sl=>{
            const rows = rq3.by_study_slice?.[sl]||[];
            const r = findProfileMethodRow(rows,p,'hybrid') || findProfileMethodRow(rows,'hybrid',p);
            return r?+r.mean_ndcg_at_k:0;
          }),
          backgroundColor:pClr[i].bg, borderColor:pClr[i].border, borderWidth:2, borderRadius:8 }))
      });
    }

    function drawRQ4Charts() {
      const rq4 = metadataPayload?.rq4; if (!rq4) return;
      const lenBuckets = ['short','medium','long'];
      const lenLbl = ['❌ Kısa','🟡 Orta','✅ Uzun'];
      mkChart('chart-rq4-length','bar',{
        labels: lenLbl,
        datasets:[
          { label:'Semantic Hit@5',
            data: lenBuckets.map(b=>{ const r=findMethodRow(rq4.by_length?.[b]||[],'semantic'); return r?+(r.hit_rate_at_k??0):0; }),
            backgroundColor:CC.purple.bg, borderColor:CC.purple.border, borderWidth:2, borderRadius:8 },
          { label:'Hybrid Hit@5',
            data: lenBuckets.map(b=>{ const r=findMethodRow(rq4.by_length?.[b]||[],'hybrid'); return r?+(r.hit_rate_at_k??0):0; }),
            backgroundColor:CC.green.bg, borderColor:CC.green.border, borderWidth:2, borderRadius:8 }
        ]
      });
      const styles = ['narrative','mixed_structured','metadata_heavy'];
      const sLbl = ['Anlatımsal','Karışık','Metadata'];
      mkChart('chart-rq4-style','bar',{
        labels: sLbl,
        datasets:[{ label:'Semantic Hit@5',
          data: styles.map(s=>{ const r=findMethodRow(rq4.by_style?.[s]||[],'semantic'); return r?+(r.hit_rate_at_k??0):0; }),
          backgroundColor:[CC.green.bg,CC.cyan.bg,CC.orange.bg],
          borderColor:[CC.green.border,CC.cyan.border,CC.orange.border],
          borderWidth:2, borderRadius:8 }]
      },{ noLegend:true });
      const terms = ['term_sparse','term_moderate','term_rich'];
      const tLbl = ['Seyrek','Orta','Zengin'];
      mkChart('chart-rq4-terms','bar',{
        labels: tLbl,
        datasets:[{ label:'Semantic Hit@5',
          data: terms.map(t=>{ const r=findMethodRow(rq4.by_term?.[t]||[],'semantic'); return r?+(r.hit_rate_at_k??0):0; }),
          backgroundColor:[CC.orange.bg,CC.amber.bg,CC.purple.bg],
          borderColor:[CC.orange.border,CC.amber.border,CC.purple.border],
          borderWidth:2, borderRadius:8 }]
      },{ noLegend:true });
    }

    function drawAllCharts() {
      if (typeof Chart === 'undefined') { console.warn('Chart.js yüklenemedi'); return; }
      drawRQ1Charts();
      drawRQ2Charts();
      drawRQ3Charts();
      drawRQ4Charts();
    }
    loadMetadata()
      .then(() => {
        activateTab("search");
        queryInput.focus();
        setTimeout(drawAllCharts, 200);
      })
      .catch((error) => {
        statusEl.className = "status error";
        statusEl.textContent = error.message;
      });
  </script>
</body>
</html>
"""


def configure_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_json_file(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def group_rows_by(rows, key_name):
    grouped = {}
    for row in rows:
        key = row.get(key_name)
        if not key:
            continue
        grouped.setdefault(key, []).append(row)
    return grouped


def detect_query_language(query: str) -> str:
    text = (query or "").lower()
    if any(char in text for char in "çğıöşü"):
        return "tr"
    turkish_markers = {
        "ve",
        "bir",
        "icin",
        "için",
        "ile",
        "olan",
        "veri",
        "seti",
        "yorum",
        "turkce",
        "türkçe",
    }
    tokens = {token for token in text.replace(",", " ").replace(".", " ").split() if token}
    return "tr" if tokens.intersection(turkish_markers) else "en"


def estimate_content_term_count(text: str) -> int:
    tokens = {
        token.lower()
        for token in CONTENT_TERM_RE.findall(text or "")
        if token.lower() not in CONTENT_STOPWORDS
    }
    return len(tokens)


def infer_length_bucket(word_count: int, thresholds: dict) -> str:
    short_max = thresholds.get("short_max", 20)
    medium_max = thresholds.get("medium_max", 52)
    if word_count <= short_max:
        return "short"
    if word_count <= medium_max:
        return "medium"
    return "long"


def infer_term_bucket(term_count: int, thresholds: dict) -> str:
    sparse_max = thresholds.get("sparse_max", 10)
    moderate_max = thresholds.get("moderate_max", 28)
    if term_count <= sparse_max:
        return "term_sparse"
    if term_count <= moderate_max:
        return "term_moderate"
    return "term_rich"


def infer_description_style(description: str, keywords: list, word_count: int) -> str:
    text = description or ""
    lowered = text.lower()
    colon_count = text.count(":")
    bullet_signals = text.count("- ") + text.count("•") + lowered.count("keywords:") + lowered.count("use cases:") + lowered.count("columns:")
    keyword_count = len(keywords or [])
    if word_count <= 40 and (keyword_count >= 3 or colon_count >= 3):
        return "metadata_heavy"
    if bullet_signals >= 2 or colon_count >= 4:
        return "mixed_structured"
    if word_count >= 80:
        return "narrative"
    return "mixed_structured" if keyword_count else "narrative"


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def build_quality_flag_payload(item: dict) -> dict:
    description = item.get("description") or item.get("text") or ""
    keywords = item.get("keywords") or []
    word_count = int(item.get("description_len_words") or len(description.split()))
    term_count = estimate_content_term_count(description)
    raw_flags = dedupe_preserve_order(list(item.get("quality_flags") or []))
    quality_signal = infer_quality_flags(item)

    style = infer_description_style(description, keywords, word_count) if description else None
    length_bucket = infer_length_bucket(
        word_count,
        {"short_max": 20, "medium_max": 52},
    ) if word_count else None
    term_bucket = infer_term_bucket(
        term_count,
        {"sparse_max": 10, "moderate_max": 28},
    ) if term_count or description else None

    inferred_flags = []
    if not description.strip():
        inferred_flags.append("empty")
    if length_bucket == "short" and not any(flag in raw_flags for flag in ("short_description", "too_short")):
        inferred_flags.append("short_context")
    if style == "metadata_heavy":
        inferred_flags.append("metadata_heavy")
    if term_bucket == "term_sparse":
        inferred_flags.append("term_sparse")
    if ("keyword_only" in raw_flags or style == "metadata_heavy") and length_bucket == "short":
        inferred_flags.append("low_information")
    elif style == "metadata_heavy" and term_bucket == "term_sparse":
        inferred_flags.append("low_information")

    all_flags = dedupe_preserve_order(raw_flags + inferred_flags)
    details = []
    severity_rank = {"info": 0, "warning": 1, "risk": 2}
    top_severity = "info"

    for code in all_flags:
        meta = QUALITY_FLAG_META.get(code)
        if not meta:
            meta = {
                "label": code.replace("_", " ").title(),
                "severity": "warning",
                "message": "Kayıt bu kalite bayragi ile isaretlenmis.",
            }
        severity = meta["severity"]
        if severity_rank[severity] > severity_rank[top_severity]:
            top_severity = severity
        details.append(
            {
                "code": code,
                "label": meta["label"],
                "severity": severity,
                "message": meta["message"],
                "origin": "stored" if code in raw_flags else "inferred",
            }
        )

    summary_map = {
        "info": "Temiz kayıt",
        "warning": "Dikkat gerekiyor",
        "risk": "Riskli kayıt",
    }

    return {
        "quality_flag_codes": all_flags,
        "quality_flag_details": details,
        "quality_flag_count": len(details),
        "quality_status": top_severity if details else "info",
        "quality_flag_summary": summary_map[top_severity] if details else "Flag yok",
        "description_style": style,
        "length_bucket": length_bucket,
        "term_bucket": term_bucket,
        "title_deemphasized": should_deemphasize_title(item, quality_signal),
        "semantic_quality_note": item.get("semantic_quality_note") or "",
    }


def load_app_metadata():
    profile_entries = []
    profile_stats = {}

    for profile in list_profiles():
        profile_paths = get_profile_paths(profile.key)
        metadata = load_json_file(profile_paths["index_metadata"], {})
        is_ready = profile_paths["index"].exists() and profile_paths["mappings"].exists()
        profile_stats[profile.key] = metadata
        profile_entries.append(
            {
                "key": profile.key,
                "label": profile.label,
                "model_name": profile.model_name,
                "description": profile.description,
                "indexed_rows": metadata.get("indexed_rows"),
                "source_counts": metadata.get("source_counts") or {},
                "is_ready": is_ready,
            }
        )

    default_stats = profile_stats.get(DEFAULT_WEB_PROFILE, {})
    return {
        "default_profile": DEFAULT_WEB_PROFILE,
        "profiles": profile_entries,
        "profile_stats": profile_stats,
        "indexed_rows": default_stats.get("indexed_rows"),
        "source_counts": default_stats.get("source_counts") or {},
        "live_eval": load_live_eval_metadata(),
        "retriever_adaptation": load_retriever_adaptation_metadata(),
        "rq1": load_rq1_metadata(),
        "rq2": load_rq2_metadata(),
        "rq3": load_rq3_metadata(),
        "rq4": load_rq4_metadata(),
    }


def load_rq1_metadata():
    overall = load_json_file(RQ1_SUMMARY_PATH, [])
    by_query_style_rows = load_json_file(RQ1_QUERY_STYLE_PATH, [])
    by_benchmark_rows = load_json_file(RQ1_BENCHMARK_PATH, [])
    return {
        "overall": overall,
        "by_query_style": group_rows_by(by_query_style_rows, "query_style"),
        "by_benchmark": by_benchmark_rows,
    }


def load_rq2_metadata():
    overall = load_json_file(RQ2_SUMMARY_PATH, [])
    by_direction_rows = load_json_file(RQ2_DIRECTION_PATH, [])
    by_topic_rows = load_json_file(RQ2_TOPIC_PATH, [])
    direction_counts = {}
    for row in by_direction_rows:
        direction = row.get("direction")
        if not direction or direction in direction_counts:
            continue
        direction_counts[direction] = row.get("queries", 0)

    topic_counts = {}
    for row in by_topic_rows:
        topic = row.get("topic")
        if not topic or topic in topic_counts:
            continue
        topic_counts[topic] = row.get("queries", 0)

    largest_topic = None
    smallest_topic = None
    if topic_counts:
        sorted_topics = sorted(topic_counts.items(), key=lambda item: (item[1], item[0]))
        smallest_topic = sorted_topics[0][0]
        largest_topic = sorted_topics[-1][0]

    return {
        "overall": overall,
        "by_direction": group_rows_by(by_direction_rows, "direction"),
        "by_topic": group_rows_by(by_topic_rows, "topic"),
        "query_count": overall[0].get("queries") if overall else 0,
        "direction_counts": direction_counts,
        "topic_counts": topic_counts,
        "largest_topic": largest_topic,
        "smallest_topic": smallest_topic,
    }


def load_rq3_metadata():
    overall = load_json_file(RQ3_SUMMARY_PATH, [])
    by_language_rows = load_json_file(RQ3_LANGUAGE_PATH, [])
    by_benchmark_rows = load_json_file(RQ3_BENCHMARK_PATH, [])
    by_study_slice_rows = load_json_file(RQ3_STUDY_SLICE_PATH, [])

    language_counts = {}
    for row in by_language_rows:
        language = row.get("language")
        if not language or language in language_counts:
            continue
        language_counts[language] = row.get("queries", 0)

    study_slice_counts = {}
    for row in by_study_slice_rows:
        study_slice = row.get("study_slice")
        if not study_slice or study_slice in study_slice_counts:
            continue
        study_slice_counts[study_slice] = row.get("queries", 0)

    benchmark_counts = {}
    for row in by_benchmark_rows:
        benchmark = row.get("benchmark")
        if not benchmark or benchmark in benchmark_counts:
            continue
        benchmark_counts[benchmark] = row.get("queries", 0)

    return {
        "overall": overall,
        "by_language": group_rows_by(by_language_rows, "language"),
        "by_benchmark": group_rows_by(by_benchmark_rows, "benchmark"),
        "by_study_slice": group_rows_by(by_study_slice_rows, "study_slice"),
        "benchmark_profiles": list(RQ3_BENCHMARK_PROFILES),
        "query_count": overall[0].get("queries") if overall else 0,
        "language_counts": language_counts,
        "study_slice_counts": study_slice_counts,
        "benchmark_counts": benchmark_counts,
    }


def load_rq4_metadata():
    details = load_json_file(RQ4_DETAILS_PATH, [])
    by_style_rows = load_json_file(RQ4_STYLE_PATH, [])
    by_length_rows = load_json_file(RQ4_LENGTH_PATH, [])
    by_term_rows = load_json_file(RQ4_TERM_PATH, [])
    corpus_distribution = load_json_file(RQ4_CORPUS_DISTRIBUTION_PATH, {})

    unique_queries = set()
    study_slice_counts = {}
    for row in details:
        query_id = row.get("query_id")
        study_slice = row.get("study_slice")
        if query_id:
            unique_queries.add(query_id)
        if query_id and study_slice:
            study_slice_counts.setdefault(study_slice, set()).add(query_id)

    distribution = {}
    for row in corpus_distribution.get("distribution", []):
        feature = row.get("feature")
        bucket = row.get("bucket")
        share = row.get("share")
        if not feature or not bucket:
            continue
        distribution.setdefault(feature, {})[bucket] = share

    return {
        "query_count": len(unique_queries),
        "study_slice_counts": {key: len(value) for key, value in study_slice_counts.items()},
        "by_style": group_rows_by(by_style_rows, "description_style"),
        "by_length": group_rows_by(by_length_rows, "length_bucket"),
        "by_term": group_rows_by(by_term_rows, "term_bucket"),
        "distribution": distribution,
        "thresholds": corpus_distribution.get("thresholds", {}),
    }


def load_live_eval_metadata():
    queries = load_json_file(LIVE_EVAL_QUERY_PATH, [])
    profile_exports = {}
    for profile in list_profiles():
        report_path = LIVE_EVAL_REPORT_ROOT / profile.key / "top10_results.json"
        payload = load_json_file(report_path, {})
        profile_exports[profile.key] = {
            "available": bool(payload),
            "profile": profile.key,
            "profile_label": profile.label,
            "top_k": payload.get("top_k"),
            "query_count": payload.get("query_count"),
            "enable_rerank": payload.get("enable_rerank"),
            "enable_cross_encoder": payload.get("enable_cross_encoder"),
            "enable_quality_penalty": payload.get("enable_quality_penalty"),
            "enable_tr_fusion": payload.get("enable_tr_fusion"),
        }
    return {
        "query_count": len(queries),
        "queries": queries,
        "profile_exports": profile_exports,
    }


def load_retriever_adaptation_metadata():
    return {
        "summary": load_json_file(HARD_NEGATIVE_SUMMARY_PATH, {}),
        "query_specs": load_json_file(HARD_NEGATIVE_QUERY_SPEC_PATH, []),
    }


class AppState:
    search_engines = {}
    bm25_indices = {}
    metadata = {}
    lock = threading.Lock()


def ensure_search_engine(profile_key):
    profile = get_profile(profile_key)
    cached = AppState.search_engines.get(profile.key)
    if cached is not None:
        return cached

    with AppState.lock:
        cached = AppState.search_engines.get(profile.key)
        if cached is not None:
            return cached
        print(f"[WEB] Search engine yükleniyor: {profile.key}")
        AppState.search_engines[profile.key] = load_search_engine(profile.key)
        return AppState.search_engines[profile.key]


def ensure_bm25_engine(profile_key):
    profile = get_profile(profile_key)
    cached = AppState.bm25_indices.get(profile.key)
    if cached is not None:
        return cached

    with AppState.lock:
        cached = AppState.bm25_indices.get(profile.key)
        if cached is not None:
            return cached

        engine = AppState.search_engines.get(profile.key)
        if engine is not None:
            _, _, mappings = engine
        else:
            mappings = load_mappings(profile.key)
        print(f"[WEB] BM25 index hazırlanıyor: {profile.key}")
        AppState.bm25_indices[profile.key] = BM25Index(mappings)
        return AppState.bm25_indices[profile.key]


def filter_results_by_source(raw_results, source_filter, top_k):
    results = []
    for score, item in raw_results:
        if source_filter != "all" and (item.get("source") or "").lower() != source_filter:
            continue
        results.append((score, item))
        if len(results) >= top_k:
            break
    return results


def execute_search_method(method, query, profile_key, source_filter, top_k, query_plan, enable_cross_encoder=False):
    if method == "bm25":
        bm25_index = ensure_bm25_engine(profile_key)
        candidate_k = len(bm25_index.doc_ids) if source_filter != "all" else top_k
        raw_results = bm25_search(
            query,
            bm25_index,
            top_k=candidate_k,
            query_plan=query_plan,
        )
    elif method == "hybrid":
        model, index, mappings = ensure_search_engine(profile_key)
        bm25_index = ensure_bm25_engine(profile_key)
        candidate_k = (
            len(mappings)
            if source_filter != "all"
            else max(top_k, min(DEFAULT_CANDIDATE_K, len(mappings)))
        )
        raw_results = hybrid_search(
            query,
            model,
            index,
            mappings,
            bm25_index,
            top_k=candidate_k if source_filter != "all" else top_k,
            semantic_weight=DEFAULT_SEMANTIC_WEIGHT,
            candidate_k=candidate_k,
            profile_key=profile_key,
            query_plan=query_plan,
            enable_cross_encoder=enable_cross_encoder,
        )
    else:
        model, index, mappings = ensure_search_engine(profile_key)
        candidate_k = index.ntotal if source_filter != "all" else top_k
        raw_results = semantic_search(
            query,
            model,
            index,
            mappings,
            top_k=candidate_k,
            profile_key=profile_key,
            query_plan=query_plan,
            enable_cross_encoder=enable_cross_encoder,
        )
    return filter_results_by_source(raw_results, source_filter, top_k)


def find_method_row(rows, method):
    for row in rows:
        if row.get("method") == method:
            return row
    return None


def find_profile_method_row(rows, profile_key, method):
    for row in rows:
        if row.get("profile") == profile_key and row.get("method") == method:
            return row
    return None


def is_profile_ready(profile_key):
    paths = get_profile_paths(profile_key)
    return paths["index"].exists() and paths["mappings"].exists()


def get_other_profile_key(profile_key, language="en"):
    preferred_candidates = []
    if profile_key == "minilm":
        preferred_candidates = ["multilingual" if language == "tr" else "e5_base", "minilm_ft", "multilingual"]
    elif profile_key == "e5_base":
        preferred_candidates = ["minilm", "minilm_ft", "multilingual"]
    elif profile_key == "minilm_ft":
        preferred_candidates = ["minilm", "e5_base", "multilingual"]
    elif profile_key == "multilingual":
        preferred_candidates = ["minilm" if language == "tr" else "e5_base", "minilm", "minilm_ft"]

    for candidate in preferred_candidates + list(PROFILE_ORDER):
        if candidate != profile_key and is_profile_ready(candidate):
            return candidate
    return profile_key


def build_rq1_payload(query_plan, profile_key, source_filter, top_k, method_results):
    rq1_data = (AppState.metadata or {}).get("rq1") or {}
    query_style = "sentence" if query_plan.get("is_sentence_query") else "keyword"
    style_rows = (rq1_data.get("by_query_style") or {}).get(query_style, [])
    recommended = max(
        style_rows,
        key=lambda row: row.get("mean_ndcg_at_k", 0.0),
        default=None,
    )
    guidance = []
    if recommended:
        guidance.append(
            f"Bu sorgu `{query_style}` tipinde gorunuyor. RQ1 benchmarkinda bu tipte en güçlü method `{METHOD_LABELS.get(recommended['method'], recommended['method'])}` "
            f"(nDCG@5={recommended['mean_ndcg_at_k']:.3f})."
        )

    semantic_row = find_method_row(style_rows, "semantic")
    bm25_row = find_method_row(style_rows, "bm25")
    if semantic_row and bm25_row and query_style == "sentence":
        guidance.append(
            f"Sentence query'lerde semantic retrieval, BM25'i geçiyor "
            f"(semantic nDCG@5={semantic_row['mean_ndcg_at_k']:.3f}, BM25={bm25_row['mean_ndcg_at_k']:.3f})."
        )
    elif semantic_row and bm25_row:
        guidance.append(
            f"Keyword query'lerde BM25 hala anlamlı bir baseline; ama genel ranking kalitesinde hybrid onde kalir."
        )

    if profile_key != DEFAULT_WEB_PROFILE:
        guidance.append(
            "Not: RQ1 benchmark metrikleri tezin ana İngilizce MiniLM profiline aittir."
        )

    snapshots = []
    for method in ("bm25", "semantic", "hybrid"):
        rows = method_results.get(method) or []
        top = rows[0] if rows else None
        top_result = None
        if top is not None:
            score, item = top
            top_result = {
                "score": score,
                "title": item.get("title"),
                "source": item.get("source"),
                "ref": item.get("ref"),
            }
        snapshots.append(
            {
                "method": method,
                "top_result": top_result,
                "benchmark_metrics": find_method_row(style_rows, method),
            }
        )

    return {
        "query_style": query_style,
        "query_style_label": "sentence query" if query_style == "sentence" else "keyword query",
        "recommended_method": recommended.get("method") if recommended else None,
        "guidance_text": " ".join(guidance),
        "source_filter": source_filter,
        "top_k": top_k,
        "snapshots": snapshots,
    }


def build_rq2_payload(profile_key, source_filter, top_k, method_results):
    if source_filter == "all":
        return {
            "enabled": False,
            "source_filter": source_filter,
            "top_k": top_k,
        }

    rq2_data = (AppState.metadata or {}).get("rq2") or {}
    direction = (
        "huggingface_to_kaggle"
        if source_filter == "kaggle"
        else "kaggle_to_huggingface"
    )
    direction_rows = (rq2_data.get("by_direction") or {}).get(direction, [])
    recommended = max(
        direction_rows,
        key=lambda row: row.get("mean_ndcg_at_k", 0.0),
        default=None,
    )

    guidance = [
        f"Hedef kaynak `{source_filter}` olduğu için bu canlı arama, RQ2 benchmarkindaki `{DIRECTION_LABELS.get(direction, direction)}` yönüne en yakın senaryo."
    ]
    if recommended:
        guidance.append(
            f"Bu yönde en güçlü method `{METHOD_LABELS.get(recommended['method'], recommended['method'])}` "
            f"(nDCG@5={recommended['mean_ndcg_at_k']:.3f}, Bridge@5={recommended['mean_hit_rate_at_k']:.3f})."
        )

    semantic_row = find_method_row(direction_rows, "semantic")
    bm25_row = find_method_row(direction_rows, "bm25")
    hybrid_row = find_method_row(direction_rows, "hybrid")
    if semantic_row and bm25_row:
        guidance.append(
            f"Semantic retrieval, BM25'e göre daha güçlü cross-source kopru kuruyor "
            f"(semantic Bridge@5={semantic_row['mean_hit_rate_at_k']:.3f}, BM25={bm25_row['mean_hit_rate_at_k']:.3f})."
        )
    if hybrid_row and semantic_row:
        guidance.append(
            f"Hybrid uygulamadaki en güçlü sistem olarak korunabilir; ancak RQ2'nin doğrudan cevabi semantic retrieval performansidir."
        )
    if profile_key != DEFAULT_WEB_PROFILE:
        guidance.append(
            "Not: RQ2 benchmark metrikleri tezin ana İngilizce MiniLM profiline aittir."
        )

    snapshots = []
    for method in ("bm25", "semantic", "hybrid"):
        rows = method_results.get(method) or []
        top = rows[0] if rows else None
        top_result = None
        if top is not None:
            score, item = top
            top_result = {
                "score": score,
                "title": item.get("title"),
                "source": item.get("source"),
                "ref": item.get("ref"),
            }
        snapshots.append(
            {
                "method": method,
                "top_result": top_result,
                "benchmark_metrics": find_method_row(direction_rows, method),
            }
        )

    return {
        "enabled": True,
        "direction": direction,
        "direction_label": DIRECTION_LABELS.get(direction, direction),
        "target_source": source_filter,
        "target_source_label": "Kaggle" if source_filter == "kaggle" else "Hugging Face",
        "recommended_method": recommended.get("method") if recommended else None,
        "guidance_text": " ".join(guidance),
        "top_k": top_k,
        "snapshots": snapshots,
    }


def build_rq3_payload(query, profile_key, selected_method, source_filter, top_k, query_plan, method_results, enable_cross_encoder=False):
    rq3_data = (AppState.metadata or {}).get("rq3") or {}
    language = detect_query_language(query)
    compare_method = selected_method if selected_method in {"semantic", "hybrid"} else "hybrid"
    primary_slice = "tr_subset" if language == "tr" else ("cross_source" if source_filter != "all" else "english_main")
    alternative_profile_key = get_other_profile_key(profile_key, language)

    primary_rows = (rq3_data.get("by_study_slice") or {}).get(primary_slice, [])
    language_rows = (rq3_data.get("by_language") or {}).get(language, [])
    current_primary = find_profile_method_row(primary_rows, profile_key, compare_method)
    alternative_primary = find_profile_method_row(primary_rows, alternative_profile_key, compare_method)
    current_language = find_profile_method_row(language_rows, profile_key, compare_method)
    alternative_language = find_profile_method_row(language_rows, alternative_profile_key, compare_method)
    benchmark_scope = (
        "paired"
        if profile_key in RQ3_BENCHMARK_PROFILES and alternative_profile_key in RQ3_BENCHMARK_PROFILES
        else "live_only"
    )

    guidance = []
    if compare_method != selected_method:
        guidance.append(
            "RQ3 benchmarki BM25'i kapsamadigi için model etkisi karşılaştırması hybrid katmanı üzerinden yapılıyor."
        )

    guidance.append(
        f"Algılanan sorgu dili `{LANGUAGE_LABELS.get(language, language)}`. Bu nedenle birincil karşılaştırma kesiti `{STUDY_SLICE_LABELS.get(primary_slice, primary_slice)}` olarak secildi."
    )
    if benchmark_scope != "paired":
        guidance.append(
            "Seçili profil çifti depolanmış RQ3 benchmark raporlarinda birlikte yer almiyor; bu panel bu durumda canlı retrieval davranışını ve top sonuç farklarini karşılaştırır."
        )

    recommended_profile = profile_key
    if current_primary and alternative_primary:
        current_score = current_primary.get("mean_ndcg_at_k", 0.0)
        alternative_score = alternative_primary.get("mean_ndcg_at_k", 0.0)
        recommended_profile = profile_key if current_score >= alternative_score else alternative_profile_key
        guidance.append(
            f"{METHOD_LABELS.get(compare_method, compare_method)} için bu kesitte `{get_profile(recommended_profile).label}` daha güçlü gorunuyor "
            f"(aktif profile nDCG@5={current_score:.3f}, alternatif nDCG@5={alternative_score:.3f})."
        )
    elif profile_key == "e5_base" or alternative_profile_key == "e5_base":
        guidance.append(
            "E5 Base, English retrieval için query:/passage: prefix kullanan canlı bir semantic profile olarak eklendi. Bu nedenle burada benchmark tablosundan çok, aynı sorgudaki canlı ranking degisimi izlenmelidir."
        )

    if current_language and alternative_language and language == "tr":
        guidance.append(
            f"Turkish sorgularda multilingual profile daha güçlü beklenir "
            f"(MiniLM nDCG@5={find_profile_method_row(language_rows, 'minilm', compare_method).get('mean_ndcg_at_k', 0.0):.3f}, "
            f"Multilingual nDCG@5={find_profile_method_row(language_rows, 'multilingual', compare_method).get('mean_ndcg_at_k', 0.0):.3f})."
        )
    elif current_language and alternative_language:
        guidance.append(
            f"English ağırlıklı sorgularda MiniLM ana profile olarak daha güçlü kalir "
            f"(MiniLM nDCG@5={find_profile_method_row(language_rows, 'minilm', compare_method).get('mean_ndcg_at_k', 0.0):.3f}, "
            f"Multilingual nDCG@5={find_profile_method_row(language_rows, 'multilingual', compare_method).get('mean_ndcg_at_k', 0.0):.3f})."
        )
    elif language == "en" and (profile_key == "e5_base" or alternative_profile_key == "e5_base"):
        guidance.append(
            "English retrieval odakli sorgularda E5 Base, title yerine description/task/modality ağırlıklı structured semantic_text ile daha hedefli bir eşleşme davranışı verebilir."
        )

    selected_rows = method_results.get(compare_method) or []
    alternative_rows = selected_rows if alternative_profile_key == profile_key else execute_search_method(
        compare_method,
        query,
        alternative_profile_key,
        source_filter,
        top_k,
        query_plan,
        enable_cross_encoder=enable_cross_encoder,
    )

    snapshots = []
    for candidate_profile_key, rows, is_selected in (
        (profile_key, selected_rows, True),
        (alternative_profile_key, alternative_rows, False),
    ):
        top = rows[0] if rows else None
        top_result = None
        if top is not None:
            score, item = top
            top_result = {
                "score": score,
                "title": item.get("title"),
                "source": item.get("source"),
                "ref": item.get("ref"),
            }
        profile = get_profile(candidate_profile_key)
        snapshots.append(
            {
                "profile": candidate_profile_key,
                "profile_label": profile.label,
                "profile_description": profile.description,
                "is_selected": is_selected,
                "top_result": top_result,
                "primary_metrics": find_profile_method_row(primary_rows, candidate_profile_key, compare_method),
                "language_metrics": find_profile_method_row(language_rows, candidate_profile_key, compare_method),
            }
        )

    return {
        "enabled": True,
        "selected_profile": profile_key,
        "alternative_profile": alternative_profile_key,
        "language": language,
        "language_label": LANGUAGE_LABELS.get(language, language),
        "primary_slice": primary_slice,
        "primary_slice_label": STUDY_SLICE_LABELS.get(primary_slice, primary_slice),
        "compare_method": compare_method,
        "benchmark_scope": benchmark_scope,
        "recommended_profile": recommended_profile,
        "recommended_profile_label": get_profile(recommended_profile).label,
        "guidance_text": " ".join(guidance),
        "top_k": top_k,
        "snapshots": snapshots,
    }


def build_rq4_snapshot(method, rows, rq4_data):
    top = rows[0] if rows else None
    top_result = None
    description_style = None
    length_bucket = None
    term_bucket = None
    length_metrics = None
    style_metrics = None
    term_metrics = None

    if top is not None:
        _, item = top
        description = item.get("description") or item.get("text") or ""
        keywords = item.get("keywords") or []
        word_count = int(item.get("description_len_words") or len(description.split()))
        term_count = estimate_content_term_count(description)
        threshold_block = rq4_data.get("thresholds") or {}
        length_thresholds = threshold_block.get("length_word_thresholds") or {}
        term_thresholds = threshold_block.get("term_thresholds") or {}
        length_bucket = infer_length_bucket(word_count, length_thresholds)
        term_bucket = infer_term_bucket(term_count, term_thresholds)
        description_style = infer_description_style(description, keywords, word_count)
        length_metrics = find_method_row((rq4_data.get("by_length") or {}).get(length_bucket, []), method)
        style_metrics = find_method_row((rq4_data.get("by_style") or {}).get(description_style, []), method)
        term_metrics = find_method_row((rq4_data.get("by_term") or {}).get(term_bucket, []), method)
        top_result = {
            "title": item.get("title"),
            "source": item.get("source"),
            "ref": item.get("ref"),
            "word_count": word_count,
            "term_count": term_count,
        }

    return {
        "method": method,
        "top_result": top_result,
        "description_style": description_style,
        "description_style_label": DESCRIPTION_STYLE_LABELS.get(description_style, description_style),
        "length_bucket": length_bucket,
        "length_bucket_label": LENGTH_BUCKET_LABELS.get(length_bucket, length_bucket),
        "term_bucket": term_bucket,
        "term_bucket_label": TERM_BUCKET_LABELS.get(term_bucket, term_bucket),
        "length_metrics": length_metrics,
        "style_metrics": style_metrics,
        "term_metrics": term_metrics,
    }


def build_rq4_payload(query, profile_key, selected_method, method_results):
    rq4_data = (AppState.metadata or {}).get("rq4") or {}
    language = detect_query_language(query)
    compare_method = selected_method if selected_method in {"semantic", "hybrid"} else "hybrid"

    guidance = []
    if compare_method != selected_method:
        guidance.append(
            "RQ4 benchmarki belge kalite etkisini semantic ve hybrid katmanları üzerinden okudugu için BM25 seciminde hybrid fallback kullanılıyor."
        )
    if language != "en":
        guidance.append(
            "RQ4 benchmarki English query'ler ile üretilmiştir; Turkish sorgularda gösterilen yorum belge kalite sezgisidir, doğrudan benchmark esdegeri değildir."
        )
    if profile_key != DEFAULT_WEB_PROFILE:
        guidance.append(
            "RQ4 bucket metrikleri tezin ana MiniLM profile'ina aittir; yine de canlı panel seçili profile'in getirdigi top sonucu kalite sinyali olarak yorumlar."
        )

    snapshots = []
    for method in ("semantic", "hybrid"):
        snapshot = build_rq4_snapshot(method, method_results.get(method) or [], rq4_data)
        snapshots.append(snapshot)

    active_snapshot = next((snapshot for snapshot in snapshots if snapshot.get("method") == compare_method), None)
    if active_snapshot and active_snapshot.get("top_result"):
        guidance.append(
            f"Aktif top sonuç `{active_snapshot['description_style_label']}` / `{active_snapshot['length_bucket_label']}` / `{active_snapshot['term_bucket_label']}` sinyali veriyor."
        )
        if active_snapshot.get("length_bucket") == "short" or active_snapshot.get("term_bucket") == "term_sparse":
            guidance.append(
                "Kısa ya da terimce zayıf aciklamalarda hybrid daha güvenli bir fallback olarak tutulmalı."
            )
        elif active_snapshot.get("description_style") == "narrative" or active_snapshot.get("term_bucket") == "term_rich":
            guidance.append(
                "Anlatımsal ve terimce zengin açıklamalar semantic retrieval için daha sağlam sinyal taşıyor."
            )

    return {
        "enabled": True,
        "language": language,
        "language_label": LANGUAGE_LABELS.get(language, language),
        "compare_method": compare_method,
        "compare_method_label": METHOD_LABELS.get(compare_method, compare_method),
        "guidance_text": " ".join(guidance),
        "snapshots": snapshots,
    }


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "SemanticDatasetSearch/3.0"

    def log_message(self, fmt, *args):
        print(f"[WEB] {self.address_string()} - {fmt % args}")

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self):
        body = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_html()
            return
        if path == "/api/metadata":
            self.send_json(AppState.metadata)
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/search":
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "Invalid Content-Length"}, HTTPStatus.BAD_REQUEST)
            return

        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            self.send_json({"error": "Invalid request body"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "Invalid JSON"}, HTTPStatus.BAD_REQUEST)
            return

        query = str(payload.get("query", "")).strip()
        if not query:
            self.send_json({"error": "Sorgu bos olamaz."}, HTTPStatus.BAD_REQUEST)
            return

        try:
            profile = get_profile(payload.get("profile", DEFAULT_WEB_PROFILE))
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        method = str(payload.get("method", "semantic")).strip().lower()
        if method not in {"semantic", "bm25", "hybrid"}:
            self.send_json({"error": "Unknown search method"}, HTTPStatus.BAD_REQUEST)
            return

        source_filter = str(payload.get("source", "all")).strip().lower()
        if source_filter not in {"all", "kaggle", "huggingface"}:
            self.send_json({"error": "Unknown source filter"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            top_k = int(payload.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 20))
        enable_cross_encoder = bool(payload.get("enable_cross_encoder", False))

        try:
            query_plan = build_query_plan(query)
            candidate_methods = ("bm25", "semantic", "hybrid")
            with ThreadPoolExecutor(max_workers=len(candidate_methods)) as pool:
                futures = {
                    candidate_method: pool.submit(
                        execute_search_method,
                        candidate_method,
                        query,
                        profile.key,
                        source_filter,
                        top_k,
                        query_plan,
                        enable_cross_encoder=enable_cross_encoder,
                    )
                    for candidate_method in candidate_methods
                }
                method_results = {name: future.result() for name, future in futures.items()}
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        filtered_results = method_results[method]
        results = []
        for score, item in filtered_results:
            quality_payload = build_quality_flag_payload(item)
            results.append(
                {
                    "score": score,
                    "source": item.get("source"),
                    "ref": item.get("ref"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "description": item.get("description"),
                    "text": item.get("text"),
                    "semantic_text": item.get("semantic_text"),
                    "semantic_summary": item.get("semantic_summary"),
                    "semantic_quality_note": item.get("semantic_quality_note") or "",
                    "keywords": item.get("keywords") or [],
                    "metadata_terms": item.get("metadata_terms") or [],
                    "language_hint": item.get("language_hint") or "",
                    "inferred_domains": item.get("inferred_domains") or [],
                    "inferred_use_cases": item.get("inferred_use_cases") or [],
                    "inferred_modalities": item.get("inferred_modalities") or [],
                    "license": item.get("license") or "",
                    "quality_flags": item.get("quality_flags") or [],
                    "description_len_chars": item.get("description_len_chars"),
                    "description_len_words": item.get("description_len_words"),
                    "semantic_score": item.get("semantic_score"),
                    "bm25_score": item.get("bm25_score"),
                    "semantic_weight": item.get("semantic_weight"),
                    "bm25_weight": item.get("bm25_weight"),
                    "semantic_variant_hits": item.get("semantic_variant_hits"),
                    "semantic_aspect_count": item.get("semantic_aspect_count"),
                    "semantic_aspect_hits": item.get("semantic_aspect_hits"),
                    "semantic_aspect_labels": item.get("semantic_aspect_labels") or [],
                    "semantic_aspect_scores": item.get("semantic_aspect_scores") or {},
                    "aspect_coverage": item.get("aspect_coverage"),
                    "aspect_min_score": item.get("aspect_min_score"),
                    "aspect_avg_score": item.get("aspect_avg_score"),
                    "semantic_rerank_score": item.get("semantic_rerank_score"),
                    "heuristic_rerank_score": item.get("heuristic_rerank_score"),
                    "cross_encoder_score": item.get("cross_encoder_score"),
                    "cross_encoder_score_norm": item.get("cross_encoder_score_norm"),
                    "cross_encoder_model": item.get("cross_encoder_model"),
                    "cross_encoder_enabled": item.get("cross_encoder_enabled"),
                    "cross_encoder_error": item.get("cross_encoder_error"),
                    "weak_match": bool(item.get("weak_match")),
                    "query_confidence": item.get("query_confidence"),
                    "query_is_ood": bool(item.get("query_is_ood")),
                    "lexical_anchor_signal": item.get("lexical_anchor_signal") or {},
                    "profile_key": profile.key,
                    "profile_label": profile.label,
                    **quality_payload,
                }
            )

        self.send_json(
            {
                "query": query,
                "query_plan": query_plan,
                "method": method,
                "enable_cross_encoder": enable_cross_encoder,
                "profile": {"key": profile.key, "label": profile.label},
                "source": source_filter,
                "top_k": top_k,
                "rq1": build_rq1_payload(query_plan, profile.key, source_filter, top_k, method_results),
                "rq2": build_rq2_payload(profile.key, source_filter, top_k, method_results),
                "rq3": build_rq3_payload(
                    query,
                    profile.key,
                    method,
                    source_filter,
                    top_k,
                    query_plan,
                    method_results,
                    enable_cross_encoder=enable_cross_encoder,
                ),
                "rq4": build_rq4_payload(query, profile.key, method, method_results),
                "results": results,
            }
        )


def main():
    configure_output()
    AppState.metadata = load_app_metadata()

    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    print(f"[WEB] UI hazır: http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\n[WEB] Kapatılıyor...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
