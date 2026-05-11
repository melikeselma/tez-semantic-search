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
    re.compile(r"^(?:about|for|with)\s+(.+)", re.IGNORECASE),
    re.compile(r"\bveri(?:\s+seti|leri)?\s+(?:hakkinda|icin|ile ilgili)\s+(.+)", re.IGNORECASE),
    re.compile(r"^(?:hakkinda|icin|ile ilgili)\s+(.+)", re.IGNORECASE),
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
    r"\b(veri setleri|veri seti|datasets?|datasetler|dataseti|veriler|verisi|veri|data)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a",
    "about",
    "above",
    "affect",
    "affects",
    "all",
    "am",
    "analyse",
    "analyze",
    "analyzing",
    "an",
    "and",
    "any",
    "ara",
    "are",
    "ariyorum",
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
    "finding",
    "for",
    "from",
    "gibi",
    "goster",
    "hakkinda",
    "how",
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
    "research",
    "researching",
    "search",
    "searching",
    "show",
    "something",
    "style",
    "study",
    "studying",
    "taken",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "to",
    "use",
    "used",
    "using",
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
    "ariyorum",
    "analiz",
    "analizi",
    "afet",
    "balik",
    "bana",
    "deniz",
    "deprem",
    "dolandiricilik",
    "dolandiriciligi",
    "duygu",
    "firtina",
    "finansal",
    "goster",
    "hava",
    "icin",
    "iklim",
    "islem",
    "karti",
    "kredi",
    "kuraklik",
    "okyanus",
    "sonrasi",
    "supheli",
    "sel",
    "su",
    "tahmin",
    "tespit",
    "tespiti",
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
    # These rules add retrieval-focused concepts so semantic query embeddings
    # reflect both the domain and the task described in natural language.
    "house price": {
        "terms": ["house price", "housing price", "real estate", "property valuation", "sale price"],
        "task_terms": ["prediction", "regression", "valuation"],
    },
    "house prices": {
        "terms": ["house price", "housing price", "real estate", "property valuation", "sale price"],
        "task_terms": ["prediction", "regression", "valuation"],
    },
    "housing price": {
        "terms": ["house price", "housing price", "real estate", "property valuation", "sale price"],
        "task_terms": ["prediction", "regression", "valuation"],
    },
    "housing prices": {
        "terms": ["house price", "housing price", "real estate", "property valuation", "sale price"],
        "task_terms": ["prediction", "regression", "valuation"],
    },
    "real estate": {
        "terms": ["real estate", "property valuation", "house price", "sale price"],
        "task_terms": ["prediction", "regression", "valuation"],
    },
    "property valuation": {
        "terms": ["property valuation", "real estate", "house price", "sale price"],
        "task_terms": ["prediction", "regression", "valuation"],
    },
    "sale price": {
        "terms": ["sale price", "house price", "property valuation", "real estate"],
        "task_terms": ["prediction", "regression", "valuation"],
    },
    "weather affects crop production": {
        "domain": "weather_climate",
        "terms": ["crop production", "crop yield", "agriculture", "farming", "rainfall", "temperature", "climate impact"],
        "task_terms": ["prediction", "impact analysis", "forecasting"],
    },
    "crop production": {
        "domain": "weather_climate",
        "terms": ["crop production", "crop yield", "agriculture", "farming", "rainfall", "temperature"],
        "task_terms": ["prediction", "impact analysis", "forecasting"],
    },
    "crop yield": {
        "domain": "weather_climate",
        "terms": ["crop yield", "crop production", "agriculture", "farming", "rainfall", "temperature"],
        "task_terms": ["prediction", "impact analysis", "forecasting"],
    },
    "agriculture climate": {
        "domain": "weather_climate",
        "terms": ["agriculture", "crop production", "crop yield", "rainfall", "temperature", "climate impact"],
        "task_terms": ["prediction", "impact analysis", "forecasting"],
    },
    "climate impact": {
        "domain": "weather_climate",
        "terms": ["climate impact", "crop yield", "crop production", "agriculture", "rainfall", "temperature"],
        "task_terms": ["prediction", "impact analysis", "forecasting"],
    },
    "diabetes prediction": {
        "domain": "health_medical",
        "terms": ["diabetes", "glucose", "blood sugar", "patient records", "medical records", "clinical indicators"],
        "task_terms": ["prediction", "classification", "risk modeling"],
    },
    "patient records": {
        "domain": "health_medical",
        "terms": ["patient records", "medical records", "clinical indicators", "clinical data", "health records"],
        "task_terms": ["prediction", "classification", "risk modeling"],
    },
    "medical records": {
        "domain": "health_medical",
        "terms": ["medical records", "patient records", "clinical indicators", "clinical data", "health records"],
        "task_terms": ["prediction", "classification", "risk modeling"],
    },
    "blood sugar": {
        "domain": "health_medical",
        "terms": ["blood sugar", "glucose", "diabetes", "clinical indicators", "patient records"],
        "task_terms": ["prediction", "classification"],
    },
    "clinical indicators": {
        "domain": "health_medical",
        "terms": ["clinical indicators", "patient records", "medical records", "diabetes", "glucose"],
        "task_terms": ["prediction", "classification", "risk modeling"],
    },
    "financial fraud": {
        "domain": "finance_markets",
        "terms": ["financial fraud", "transaction fraud", "credit card fraud", "banking fraud", "suspicious transaction"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    "credit card fraud": {
        "domain": "finance_markets",
        "terms": ["credit card fraud", "financial fraud", "transaction fraud", "suspicious transaction", "banking fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    "suspicious transaction": {
        "domain": "finance_markets",
        "terms": ["suspicious transaction", "transaction fraud", "financial fraud", "banking fraud", "credit card fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    "banking fraud": {
        "domain": "finance_markets",
        "terms": ["banking fraud", "financial fraud", "transaction fraud", "suspicious transaction", "credit card fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    "transaction fraud": {
        "domain": "finance_markets",
        "terms": ["transaction fraud", "financial fraud", "credit card fraud", "suspicious transaction", "banking fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    "finansal dolandiricilik": {
        "domain": "finance_markets",
        "terms": ["financial fraud", "transaction fraud", "credit card fraud", "suspicious transaction", "banking fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    "supheli islem": {
        "domain": "finance_markets",
        "terms": ["suspicious transaction", "transaction fraud", "financial fraud", "banking fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    "kredi karti dolandiriciligi": {
        "domain": "finance_markets",
        "terms": ["credit card fraud", "financial fraud", "transaction fraud", "suspicious transaction", "banking fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    "kredi karti dolandiricilik": {
        "domain": "finance_markets",
        "terms": ["credit card fraud", "financial fraud", "transaction fraud", "suspicious transaction", "banking fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    "satellite imagery": {
        "domain": "earth_observation",
        "terms": ["satellite imagery", "remote sensing", "aerial imagery", "land use", "land cover", "urban change", "urban growth"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    "aerial imagery": {
        "domain": "earth_observation",
        "terms": ["aerial imagery", "satellite imagery", "remote sensing", "urban change", "land use", "land cover"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    "images taken from above": {
        "domain": "earth_observation",
        "terms": ["aerial imagery", "satellite imagery", "remote sensing", "urban change", "urban growth", "land use", "land cover"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    "remote sensing": {
        "domain": "earth_observation",
        "terms": ["remote sensing", "satellite imagery", "aerial imagery", "land use", "land cover", "urban change"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    "land use": {
        "domain": "earth_observation",
        "terms": ["land use", "land cover", "remote sensing", "satellite imagery", "urban change"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    "land cover": {
        "domain": "earth_observation",
        "terms": ["land cover", "land use", "remote sensing", "satellite imagery", "urban change"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    "urban change": {
        "domain": "earth_observation",
        "terms": ["urban change", "urban growth", "satellite imagery", "remote sensing", "land use", "land cover"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    "urban growth": {
        "domain": "earth_observation",
        "terms": ["urban growth", "urban change", "satellite imagery", "remote sensing", "land use", "land cover"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    "harmful language": {
        "domain": "nlp_text",
        "terms": ["harmful language", "toxic comments", "hate speech", "abusive language", "offensive language"],
        "task_terms": ["classification", "moderation"],
    },
    "hate speech": {
        "domain": "nlp_text",
        "terms": ["hate speech", "toxic comments", "abusive language", "offensive language", "harmful language"],
        "task_terms": ["classification", "moderation"],
    },
    "toxic comments": {
        "domain": "nlp_text",
        "terms": ["toxic comments", "hate speech", "abusive language", "offensive language", "harmful language"],
        "task_terms": ["classification", "moderation"],
    },
    "abusive language": {
        "domain": "nlp_text",
        "terms": ["abusive language", "toxic comments", "hate speech", "offensive language", "harmful language"],
        "task_terms": ["classification", "moderation"],
    },
    "offensive language": {
        "domain": "nlp_text",
        "terms": ["offensive language", "abusive language", "toxic comments", "hate speech", "harmful language"],
        "task_terms": ["classification", "moderation"],
    },
    "time series": {
        "terms": ["time series", "historical values", "past measurements", "future values", "sensor measurements"],
        "task_terms": ["forecasting", "prediction"],
    },
    "future values": {
        "terms": ["future values", "historical values", "time series", "past measurements", "sensor measurements"],
        "task_terms": ["forecasting", "prediction"],
    },
    "past measurements": {
        "terms": ["past measurements", "historical values", "time series", "future values", "sensor measurements"],
        "task_terms": ["forecasting", "prediction"],
    },
    "sensor measurements": {
        "terms": ["sensor measurements", "time series", "historical values", "past measurements", "future values"],
        "task_terms": ["forecasting", "prediction"],
    },
    "historical values": {
        "terms": ["historical values", "time series", "past measurements", "future values", "sensor measurements"],
        "task_terms": ["forecasting", "prediction"],
    },
    "social media posts": {
        "domain": "nlp_text",
        "terms": ["social media posts", "tweets", "short online messages", "comments", "reviews"],
        "task_terms": ["classification", "sentiment analysis"],
    },
    "short online messages": {
        "domain": "nlp_text",
        "terms": ["short online messages", "tweets", "social media posts", "comments", "reviews"],
        "task_terms": ["classification", "sentiment analysis"],
    },
    "online messages": {
        "domain": "nlp_text",
        "terms": ["short online messages", "tweets", "social media posts", "comments", "reviews"],
        "task_terms": ["classification", "sentiment analysis"],
    },
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
    "aerial": {"domain": "earth_observation", "terms": ["aerial imagery", "satellite imagery", "remote sensing"]},
    "agriculture": {"domain": "weather_climate", "terms": ["agriculture", "crop production", "crop yield", "farming"]},
    "banking": {"domain": "finance_markets", "terms": ["banking fraud", "financial fraud", "transaction fraud"]},
    "balik": {"domain": "ocean_climate", "terms": ["fishery", "aquatic", "marine"]},
    "climate": {"domain": "weather_climate", "terms": ["climate", "weather", "meteorology"]},
    "crop": {"domain": "weather_climate", "terms": ["crop production", "crop yield", "agriculture", "farming"]},
    "credit": {"domain": "finance_markets", "terms": ["credit card fraud", "financial fraud", "transaction fraud"]},
    "deepfake": {"domain": "media_authenticity", "terms": ["deepfake", "forgery detection", "audio fake"]},
    "deniz": {"domain": "ocean_climate", "terms": ["marine", "ocean", "sea surface", "aquatic"]},
    "deprem": {"domain": "earth_science", "terms": ["earthquake", "seismic", "geology", "disaster response"]},
    "diabetes": {"domain": "health_medical", "terms": ["diabetes", "glucose", "blood sugar", "patient records"]},
    "dolandiricilik": {"domain": "finance_markets", "terms": ["financial fraud", "transaction fraud", "banking fraud"]},
    "drought": {"domain": "climate_variability", "terms": ["drought", "climate", "precipitation", "temperature"]},
    "duygu": {"domain": "reviews_sentiment", "terms": ["sentiment", "emotion", "affect", "opinion"]},
    "earthquake": {"domain": "earth_science", "terms": ["earthquake", "seismic", "usgs", "geology"]},
    "ecology": {"domain": "ocean_climate", "terms": ["ecology", "ecosystem", "biodiversity"]},
    "estate": {"terms": ["real estate", "property valuation", "house price"]},
    "farming": {"domain": "weather_climate", "terms": ["farming", "agriculture", "crop yield", "crop production"]},
    "financial": {"domain": "finance_markets", "terms": ["financial fraud", "transaction fraud", "banking fraud"]},
    "firtina": {"domain": "weather_climate", "terms": ["storm", "severe weather", "forecast", "meteorology"]},
    "fishery": {"domain": "ocean_climate", "terms": ["fishery", "marine", "ocean", "aquatic"]},
    "flood": {"domain": "hydrology", "terms": ["flood", "runoff", "river", "water level", "hydrology"]},
    "forecast": {"domain": "weather_climate", "terms": ["forecast", "meteorology", "weather archive"]},
    "forecasting": {"terms": ["forecasting", "time series", "historical values", "future values"]},
    "fraud": {"domain": "finance_markets", "terms": ["financial fraud", "transaction fraud", "credit card fraud", "banking fraud"]},
    "glucose": {"domain": "health_medical", "terms": ["glucose", "blood sugar", "diabetes", "clinical indicators"]},
    "hava": {"domain": "weather_climate", "terms": ["weather", "meteorology", "climate"]},
    "hate": {"domain": "nlp_text", "terms": ["hate speech", "toxic comments", "abusive language"]},
    "health": {"domain": "health", "terms": ["health", "medical", "clinical"]},
    "house": {"terms": ["house price", "housing price", "real estate", "property valuation"]},
    "housing": {"terms": ["housing price", "house price", "real estate", "property valuation"]},
    "iklim": {"domain": "weather_climate", "terms": ["climate", "weather", "meteorology"]},
    "imagery": {"domain": "earth_observation", "terms": ["satellite imagery", "aerial imagery", "remote sensing"]},
    "islem": {"domain": "finance_markets", "terms": ["suspicious transaction", "transaction fraud", "financial fraud"]},
    "kuraklik": {"domain": "climate_variability", "terms": ["drought", "climate", "precipitation", "temperature"]},
    "land": {"domain": "earth_observation", "terms": ["land use", "land cover", "remote sensing", "satellite imagery"]},
    "marine": {"domain": "ocean_climate", "terms": ["marine", "ocean", "aquatic", "sea surface"]},
    "medical": {"domain": "health", "terms": ["medical", "clinical", "health"]},
    "meteorology": {"domain": "weather_climate", "terms": ["meteorology", "weather", "forecast"]},
    "ocean": {"domain": "ocean_climate", "terms": ["ocean", "marine", "buoy", "sea surface", "aquatic"]},
    "offensive": {"domain": "nlp_text", "terms": ["offensive language", "abusive language", "toxic comments"]},
    "okyanus": {"domain": "ocean_climate", "terms": ["ocean", "marine", "buoy", "sea surface", "aquatic"]},
    "opinion": {"domain": "social_discourse", "terms": ["opinion", "discussion", "comments", "public sentiment"]},
    "pollution": {"domain": "environment", "terms": ["pollution", "environment", "air quality"]},
    "precipitation": {"domain": "weather_climate", "terms": ["precipitation", "rainfall", "weather", "climate"]},
    "property": {"terms": ["property valuation", "real estate", "house price", "sale price"]},
    "records": {"domain": "health_medical", "terms": ["patient records", "medical records", "clinical indicators"]},
    "rain": {"domain": "weather_climate", "terms": ["rainfall", "precipitation", "weather", "meteorology"]},
    "rainfall": {"domain": "weather_climate", "terms": ["rainfall", "precipitation", "weather", "climate"]},
    "rainy": {"domain": "weather_climate", "terms": ["rainfall", "precipitation", "weather", "meteorology"]},
    "remote": {"domain": "earth_observation", "terms": ["remote sensing", "satellite imagery", "aerial imagery"]},
    "review": {"domain": "reviews_sentiment", "terms": ["reviews", "sentiment", "ratings", "opinions"]},
    "river": {"domain": "hydrology", "terms": ["river", "water level", "runoff", "discharge"]},
    "runoff": {"domain": "hydrology", "terms": ["runoff", "discharge", "flood", "river", "hydrology"]},
    "satellite": {"domain": "earth_observation", "terms": ["satellite imagery", "remote sensing", "land cover", "land use"]},
    "sensor": {"terms": ["sensor measurements", "time series", "historical values", "past measurements"]},
    "sea": {"domain": "ocean_climate", "terms": ["sea surface", "ocean", "marine", "aquatic"]},
    "sel": {"domain": "hydrology", "terms": ["flood", "runoff", "river", "water level", "hydrology"]},
    "sentiment": {"domain": "reviews_sentiment", "terms": ["sentiment", "reviews", "ratings", "classification"]},
    "sonrasi": {"domain": "earth_science", "terms": ["post disaster", "response", "recovery"]},
    "suspicious": {"domain": "finance_markets", "terms": ["suspicious transaction", "transaction fraud", "financial fraud"]},
    "speech": {"domain": "audio_speech", "terms": ["speech", "audio", "recognition", "acoustic"]},
    "stock": {"domain": "finance", "terms": ["stock", "market", "financial", "price history"]},
    "storm": {"domain": "weather_climate", "terms": ["storm", "severe weather", "forecast", "meteorology"]},
    "su": {"domain": "hydrology", "terms": ["water", "hydrology", "aquatic"]},
    "supheli": {"domain": "finance_markets", "terms": ["suspicious transaction", "transaction fraud", "financial fraud"]},
    "tahmin": {"terms": ["prediction", "forecast", "estimate"]},
    "toxic": {"domain": "nlp_text", "terms": ["toxic comments", "hate speech", "abusive language", "offensive language"]},
    "transaction": {"domain": "finance_markets", "terms": ["transaction fraud", "suspicious transaction", "financial fraud"]},
    "tweet": {"domain": "nlp_text", "terms": ["tweets", "social media posts", "short online messages"]},
    "tweets": {"domain": "nlp_text", "terms": ["tweets", "social media posts", "short online messages"]},
    "urban": {"domain": "earth_observation", "terms": ["urban change", "urban growth", "satellite imagery", "remote sensing"]},
    "valuation": {"terms": ["property valuation", "house price", "real estate", "sale price"]},
    "water": {"domain": "hydrology", "terms": ["water", "river", "runoff", "hydrology"]},
    "weather": {"domain": "weather_climate", "terms": ["weather", "climate", "meteorology", "forecast"]},
    "yardim": {"domain": "earth_science", "terms": ["relief", "aid", "humanitarian", "response", "crisis"]},
    "yagmur": {"domain": "weather_climate", "terms": ["rainfall", "precipitation", "weather", "meteorology"]},
    "yagmurlu": {"domain": "weather_climate", "terms": ["rainfall", "precipitation", "weather", "meteorology"]},
    "yield": {"domain": "weather_climate", "terms": ["crop yield", "crop production", "agriculture", "farming"]},
    "yorum": {"domain": "reviews_sentiment", "terms": ["reviews", "comments", "opinions", "sentiment"]},
    "yosun": {"domain": "ocean_climate", "terms": ["algae", "aquatic", "ecology", "marine"]},
    # Botany / flowers (TR + EN forms)
    "botanik": {"domain": "botany", "terms": ["botany", "botanical", "plant", "flora", "plant science"]},
    "botany": {"domain": "botany", "terms": ["botany", "botanical", "plant", "flora", "plant science"]},
    "botanical": {"domain": "botany", "terms": ["botany", "botanical", "plant", "flora", "plant species"]},
    "cicek": {"domain": "botany", "terms": ["flower", "floral", "flower classification", "flower recognition", "iris", "oxford flowers"]},
    "cicekler": {"domain": "botany", "terms": ["flowers", "floral", "flower classification", "flower recognition", "oxford flowers"]},
    "ciceklerle": {"domain": "botany", "terms": ["flowers", "flower classification", "flower recognition"]},
    "flower": {"domain": "botany", "terms": ["flower", "floral", "flower classification", "flower recognition", "iris", "oxford flowers"]},
    "flowers": {"domain": "botany", "terms": ["flowers", "floral", "flower classification", "flower recognition", "oxford flowers"]},
    "floral": {"domain": "botany", "terms": ["floral", "flower", "flower classification"]},
    "iris": {"domain": "botany", "terms": ["iris", "iris species", "flower classification"]},
    "orchid": {"domain": "botany", "terms": ["orchid", "flower", "flower classification"]},
    "bitki": {"domain": "botany", "terms": ["plant", "flora", "botany", "plant disease"]},
    "bitkiler": {"domain": "botany", "terms": ["plants", "flora", "botany", "plant species"]},
    "plant": {"domain": "botany", "terms": ["plant", "flora", "botany", "plant species", "plant disease"]},
    "plants": {"domain": "botany", "terms": ["plants", "flora", "botany", "plant species"]},
    "flora": {"domain": "botany", "terms": ["flora", "plant", "botany", "plant species"]},
    # Titanic (canonical ML beginner dataset; TR spelling differs: 'titanik')
    "titanik": {"domain": "tabular_ml", "terms": ["titanic", "titanic dataset", "titanic survival", "passenger survival"]},
    "titanic": {"domain": "tabular_ml", "terms": ["titanic", "titanic dataset", "titanic survival", "passenger survival"]},
    "hayatta": {"domain": "tabular_ml", "terms": ["survival", "survived", "survivor"]},
    "kalanlar": {"domain": "tabular_ml", "terms": ["survivors", "survival", "survived"]},
    "yolcu": {"domain": "tabular_ml", "terms": ["passenger", "passengers"]},
    "yolcular": {"domain": "tabular_ml", "terms": ["passengers", "passenger"]},
}
COMBO_RULES = [
    {
        "tokens": {"house", "price"},
        "terms": ["house price", "housing price", "real estate", "property valuation", "sale price"],
        "task_terms": ["prediction", "regression", "valuation"],
    },
    {
        "tokens": {"housing", "price"},
        "terms": ["house price", "housing price", "real estate", "property valuation", "sale price"],
        "task_terms": ["prediction", "regression", "valuation"],
    },
    {
        "tokens": {"real", "estate"},
        "terms": ["real estate", "property valuation", "house price", "sale price"],
        "task_terms": ["prediction", "regression", "valuation"],
    },
    {
        "tokens": {"property", "valuation"},
        "terms": ["property valuation", "real estate", "house price", "sale price"],
        "task_terms": ["prediction", "regression", "valuation"],
    },
    {
        "tokens": {"sale", "price"},
        "terms": ["sale price", "house price", "property valuation", "real estate"],
        "task_terms": ["prediction", "regression", "valuation"],
    },
    {
        "tokens": {"weather", "crop"},
        "domain": "weather_climate",
        "terms": ["crop production", "crop yield", "agriculture", "farming", "rainfall", "temperature", "climate impact"],
        "task_terms": ["prediction", "impact analysis", "forecasting"],
    },
    {
        "tokens": {"crop", "production"},
        "domain": "weather_climate",
        "terms": ["crop production", "crop yield", "agriculture", "farming", "rainfall", "temperature"],
        "task_terms": ["prediction", "impact analysis", "forecasting"],
    },
    {
        "tokens": {"crop", "yield"},
        "domain": "weather_climate",
        "terms": ["crop yield", "crop production", "agriculture", "farming", "rainfall", "temperature"],
        "task_terms": ["prediction", "impact analysis", "forecasting"],
    },
    {
        "tokens": {"agriculture", "climate"},
        "domain": "weather_climate",
        "terms": ["agriculture", "crop production", "crop yield", "rainfall", "temperature", "climate impact"],
        "task_terms": ["prediction", "impact analysis", "forecasting"],
    },
    {
        "tokens": {"farming", "rainfall"},
        "domain": "weather_climate",
        "terms": ["farming", "agriculture", "crop production", "crop yield", "rainfall", "temperature"],
        "task_terms": ["prediction", "impact analysis", "forecasting"],
    },
    {
        "tokens": {"farming", "temperature"},
        "domain": "weather_climate",
        "terms": ["farming", "agriculture", "crop production", "crop yield", "rainfall", "temperature"],
        "task_terms": ["prediction", "impact analysis", "forecasting"],
    },
    {
        "tokens": {"diabetes", "prediction"},
        "domain": "health_medical",
        "terms": ["diabetes", "glucose", "blood sugar", "patient records", "medical records", "clinical indicators"],
        "task_terms": ["prediction", "classification", "risk modeling"],
    },
    {
        "tokens": {"diabetes", "patient"},
        "domain": "health_medical",
        "terms": ["diabetes", "patient records", "medical records", "clinical indicators", "glucose"],
        "task_terms": ["prediction", "classification", "risk modeling"],
    },
    {
        "tokens": {"patient", "records"},
        "domain": "health_medical",
        "terms": ["patient records", "medical records", "clinical indicators", "clinical data"],
        "task_terms": ["prediction", "classification", "risk modeling"],
    },
    {
        "tokens": {"medical", "records"},
        "domain": "health_medical",
        "terms": ["medical records", "patient records", "clinical indicators", "clinical data"],
        "task_terms": ["prediction", "classification", "risk modeling"],
    },
    {
        "tokens": {"blood", "sugar"},
        "domain": "health_medical",
        "terms": ["blood sugar", "glucose", "diabetes", "clinical indicators"],
        "task_terms": ["prediction", "classification"],
    },
    {
        "tokens": {"financial", "fraud"},
        "domain": "finance_markets",
        "terms": ["financial fraud", "transaction fraud", "credit card fraud", "banking fraud", "suspicious transaction"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    {
        "tokens": {"credit", "card", "fraud"},
        "domain": "finance_markets",
        "terms": ["credit card fraud", "financial fraud", "transaction fraud", "suspicious transaction", "banking fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    {
        "tokens": {"suspicious", "transaction"},
        "domain": "finance_markets",
        "terms": ["suspicious transaction", "transaction fraud", "financial fraud", "banking fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    {
        "tokens": {"banking", "fraud"},
        "domain": "finance_markets",
        "terms": ["banking fraud", "financial fraud", "transaction fraud", "credit card fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    {
        "tokens": {"transaction", "fraud"},
        "domain": "finance_markets",
        "terms": ["transaction fraud", "financial fraud", "credit card fraud", "suspicious transaction"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    {
        "tokens": {"finansal", "dolandiricilik"},
        "domain": "finance_markets",
        "terms": ["financial fraud", "transaction fraud", "credit card fraud", "banking fraud", "suspicious transaction"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    {
        "tokens": {"supheli", "islem"},
        "domain": "finance_markets",
        "terms": ["suspicious transaction", "transaction fraud", "financial fraud", "banking fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    {
        "tokens": {"kredi", "karti", "dolandiriciligi"},
        "domain": "finance_markets",
        "terms": ["credit card fraud", "financial fraud", "transaction fraud", "suspicious transaction", "banking fraud"],
        "task_terms": ["fraud detection", "anomaly detection"],
    },
    {
        "tokens": {"satellite", "imagery"},
        "domain": "earth_observation",
        "terms": ["satellite imagery", "remote sensing", "aerial imagery", "land use", "land cover", "urban change"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    {
        "tokens": {"aerial", "imagery"},
        "domain": "earth_observation",
        "terms": ["aerial imagery", "satellite imagery", "remote sensing", "urban change", "land use", "land cover"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    {
        "tokens": {"remote", "sensing"},
        "domain": "earth_observation",
        "terms": ["remote sensing", "satellite imagery", "aerial imagery", "land use", "land cover", "urban change"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    {
        "tokens": {"land", "use"},
        "domain": "earth_observation",
        "terms": ["land use", "land cover", "remote sensing", "satellite imagery", "urban change"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    {
        "tokens": {"land", "cover"},
        "domain": "earth_observation",
        "terms": ["land cover", "land use", "remote sensing", "satellite imagery", "urban change"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    {
        "tokens": {"urban", "change"},
        "domain": "earth_observation",
        "terms": ["urban change", "urban growth", "satellite imagery", "remote sensing", "land use", "land cover"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    {
        "tokens": {"urban", "growth"},
        "domain": "earth_observation",
        "terms": ["urban growth", "urban change", "satellite imagery", "remote sensing", "land use", "land cover"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    {
        "tokens": {"cities", "change", "images"},
        "domain": "earth_observation",
        "terms": ["urban change", "urban growth", "satellite imagery", "aerial imagery", "remote sensing", "land use", "land cover"],
        "task_terms": ["change detection", "monitoring", "segmentation"],
    },
    {
        "tokens": {"harmful", "language"},
        "domain": "nlp_text",
        "terms": ["harmful language", "toxic comments", "hate speech", "abusive language", "offensive language"],
        "task_terms": ["classification", "moderation"],
    },
    {
        "tokens": {"hate", "speech"},
        "domain": "nlp_text",
        "terms": ["hate speech", "toxic comments", "abusive language", "offensive language", "harmful language"],
        "task_terms": ["classification", "moderation"],
    },
    {
        "tokens": {"toxic", "comments"},
        "domain": "nlp_text",
        "terms": ["toxic comments", "hate speech", "abusive language", "offensive language", "harmful language"],
        "task_terms": ["classification", "moderation"],
    },
    {
        "tokens": {"abusive", "language"},
        "domain": "nlp_text",
        "terms": ["abusive language", "toxic comments", "hate speech", "offensive language", "harmful language"],
        "task_terms": ["classification", "moderation"],
    },
    {
        "tokens": {"offensive", "language"},
        "domain": "nlp_text",
        "terms": ["offensive language", "abusive language", "toxic comments", "hate speech", "harmful language"],
        "task_terms": ["classification", "moderation"],
    },
    {
        "tokens": {"time", "series"},
        "terms": ["time series", "historical values", "future values", "past measurements", "sensor measurements"],
        "task_terms": ["forecasting", "prediction"],
    },
    {
        "tokens": {"sensor", "measurements"},
        "terms": ["sensor measurements", "time series", "historical values", "past measurements", "future values"],
        "task_terms": ["forecasting", "prediction"],
    },
    {
        "tokens": {"historical", "values"},
        "terms": ["historical values", "time series", "future values", "past measurements", "sensor measurements"],
        "task_terms": ["forecasting", "prediction"],
    },
    {
        "tokens": {"future", "values"},
        "terms": ["future values", "time series", "historical values", "past measurements", "sensor measurements"],
        "task_terms": ["forecasting", "prediction"],
    },
    {
        "tokens": {"social", "media", "posts"},
        "domain": "nlp_text",
        "terms": ["social media posts", "tweets", "short online messages", "comments", "reviews"],
        "task_terms": ["classification", "sentiment analysis"],
    },
    {
        "tokens": {"short", "online", "messages"},
        "domain": "nlp_text",
        "terms": ["short online messages", "tweets", "social media posts", "comments", "reviews"],
        "task_terms": ["classification", "sentiment analysis"],
    },
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


IRREGULAR_PLURAL_KEEP = {
    # Words whose surface form is plural-looking but should never be stemmed:
    # plural form == singular form, OR the naive de-plural produces nonsense.
    "diabetes",
    "species",
    "series",
    "news",
    "analysis",
    "basis",
    "thesis",
    "crisis",
    "axis",
    "physics",
    "ethics",
    "statistics",
    "mathematics",
    "economics",
    "metrics",
    "metadata",
    "data",
    "media",
    "schemas",
    "items",
    "this",
    "his",
    "as",
    "us",
    "is",
    "was",
    "has",
    "does",
    "goes",
    "lens",
    "bus",
    "address",
    "process",
    "access",
    "class",
    "loss",
    "miss",
    "boss",
    "less",
    "kiss",
    "mess",
    "press",
    "stress",
    "guess",
    "dress",
    "express",
}

# Suffix replacements that produce real English roots.
ENGLISH_PLURAL_RULES = {
    # spelled "ies" -> "y" (e.g. "industries" -> "industry"), but keep "movies"->"movy"? skip.
    "ies": "y",
}


def token_variants(token: str) -> list[str]:
    variants = [token]

    for suffix in TURKISH_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            variants.append(token[: -len(suffix)])
            break

    if token in IRREGULAR_PLURAL_KEEP:
        return unique_preserve_order(variants)

    if token.endswith("ies") and len(token) >= 5 and token[-4] not in {"a", "e", "i", "o", "u"}:
        # "industries" -> "industry"; skip "movies"/"series" which would yield "movy"/"sery".
        variants.append(token[:-3] + "y")
    elif token.endswith("ing") and len(token) >= 7:
        variants.append(token[:-3])
    elif token.endswith("ed") and len(token) >= 6:
        variants.append(token[:-2])
    elif (
        token.endswith("s")
        and len(token) >= 5
        and token not in IRREGULAR_PLURAL_KEEP
        and not token.endswith(("ss", "us", "is", "ous", "sis", "ies"))
    ):
        variants.append(token[:-1])

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
    body = re.sub(
        r"^(?:to\s+)?(?:find|study|research|search|analyze|analyse|explore|examine|investigate|predict|detect|classify)\s+",
        "",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(r"^how\s+", "", body, flags=re.IGNORECASE)

    for pattern in PHRASE_PATTERNS:
        match = pattern.search(body)
        if match:
            # Preserve content tokens that appeared before "dataset for X" so
            # queries like "mushroom dataset for classification" do not lose
            # the actual topic ("mushroom") to the trailing fragment.
            prefix = body[: match.start()].strip()
            prefix_tokens = [
                tok
                for tok in tokenize(prefix)
                if tok not in STOPWORDS
                and tok not in {"dataset", "datasets", "data", "veri", "veriler", "verisi", "veri seti"}
            ]
            suffix = match.group(1).strip()
            if prefix_tokens:
                body = " ".join(prefix_tokens + [suffix]).strip()
            else:
                body = suffix
            break

    for pattern in TAIL_PATTERNS:
        body = pattern.sub(" ", body).strip()

    body = DATASET_TERMS_RE.sub(" ", body).strip()
    # Strip dangling prepositions/articles that survive verb removal so the
    # encoded intent does not carry "on", "a", "of" as semantic anchors.
    body = re.sub(
        r"^(?:about|for|with|on|of|in|into|over|regarding|around|the|a|an)\s+",
        "",
        body,
        flags=re.IGNORECASE,
    )
    # Repeat once: "a of fingerprint" -> "of fingerprint" -> "fingerprint".
    body = re.sub(
        r"^(?:about|for|with|on|of|in|into|over|regarding|around|the|a|an)\s+",
        "",
        body,
        flags=re.IGNORECASE,
    )
    body = MULTISPACE_RE.sub(" ", body).strip()
    return body


def extract_focus_terms(text: str) -> list[str]:
    terms = []
    for token in tokenize(text):
        for variant in token_variants(token):
            if (len(variant) <= 2 and variant not in SHORT_KEEP_TOKENS) or variant in STOPWORDS:
                continue
            terms.append(variant)
    return unique_preserve_order(terms)


def collect_concept_matches(normalized_query: str, focus_terms: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    matched_terms = []
    matched_phrases = []
    task_terms = []
    domain_counts = Counter()
    token_set = set(focus_terms)

    for phrase, payload in PHRASE_RULES.items():
        if phrase in normalized_query:
            matched_phrases.extend(payload.get("terms", []))
            matched_terms.extend(payload.get("terms", []))
            task_terms.extend(payload.get("task_terms", []))
            if payload.get("domain"):
                domain_counts[payload["domain"]] += 2

    for rule in COMBO_RULES:
        if rule["tokens"].issubset(token_set):
            matched_phrases.extend(rule.get("terms", []))
            matched_terms.extend(rule.get("terms", []))
            task_terms.extend(rule.get("task_terms", []))
            if rule.get("domain"):
                domain_counts[rule["domain"]] += 2

    for token in focus_terms:
        payload = TOKEN_RULES.get(token)
        if not payload:
            continue
        matched_terms.extend(payload.get("terms", []))
        if payload.get("domain"):
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
    return (
        unique_preserve_order(matched_phrases),
        concept_terms,
        primary_domains,
        unique_preserve_order(task_terms),
    )


SEMANTIC_ASPECT_GROUPS = [
    {
        "key": "residential_property",
        "label": "residential_property",
        "kind": "concept",
        "match_terms": ["house price", "housing price", "real estate", "house", "housing", "residential property"],
        "terms": ["house", "housing", "real estate", "residential property"],
    },
    {
        "key": "property_valuation",
        "label": "property_valuation",
        "kind": "concept",
        "match_terms": ["house price", "housing price", "property valuation", "sale price", "valuation"],
        "terms": ["house price", "housing price", "property valuation", "sale price"],
    },
    {
        "key": "weather_climate",
        "label": "weather_climate",
        "kind": "concept",
        "match_terms": ["weather", "climate", "rainfall", "temperature", "precipitation", "meteorology"],
        "terms": ["weather", "climate", "rainfall", "temperature", "meteorology"],
    },
    {
        "key": "agriculture_crop",
        "label": "agriculture_crop",
        "kind": "concept",
        "match_terms": ["crop production", "crop yield", "agriculture", "farming", "crop", "yield"],
        "terms": ["crop production", "crop yield", "agriculture", "farming"],
    },
    {
        "key": "health_diabetes",
        "label": "health_diabetes",
        "kind": "concept",
        "match_terms": ["diabetes", "glucose", "blood sugar", "clinical indicators"],
        "terms": ["diabetes", "glucose", "blood sugar", "clinical indicators"],
    },
    {
        "key": "patient_records",
        "label": "patient_records",
        "kind": "concept",
        "match_terms": ["patient records", "medical records", "clinical data", "health records", "patient", "records"],
        "terms": ["patient records", "medical records", "clinical data", "health records"],
    },
    {
        "key": "finance_fraud",
        "label": "finance_fraud",
        "kind": "concept",
        "match_terms": ["financial fraud", "transaction fraud", "credit card fraud", "banking fraud", "finansal dolandiricilik"],
        "terms": ["financial fraud", "transaction fraud", "credit card fraud", "banking fraud"],
    },
    {
        "key": "transaction_risk",
        "label": "transaction_risk",
        "kind": "concept",
        "match_terms": ["suspicious transaction", "transaction", "credit card", "banking", "supheli islem", "kredi karti"],
        "terms": ["suspicious transaction", "transaction records", "credit card activity", "banking transactions"],
    },
    {
        "key": "earth_imagery",
        "label": "earth_imagery",
        "kind": "concept",
        "match_terms": ["satellite imagery", "aerial imagery", "remote sensing", "images taken from above", "overhead images"],
        "terms": ["satellite imagery", "aerial imagery", "remote sensing", "overhead images"],
    },
    {
        "key": "urban_change",
        "label": "urban_change",
        "kind": "concept",
        "match_terms": ["urban change", "urban growth", "city change", "cities change", "land use", "land cover"],
        "terms": ["urban change", "urban growth", "land use", "land cover", "city change"],
    },
    {
        "key": "toxic_language",
        "label": "toxic_language",
        "kind": "concept",
        "match_terms": ["harmful language", "hate speech", "toxic comments", "abusive language", "offensive language"],
        "terms": ["harmful language", "hate speech", "toxic comments", "abusive language", "offensive language"],
    },
    {
        "key": "social_media_text",
        "label": "social_media_text",
        "kind": "concept",
        "match_terms": ["social media posts", "tweets", "short online messages", "comments", "reviews"],
        "terms": ["social media posts", "tweets", "short online messages", "comments", "reviews"],
    },
    {
        "key": "emotion_sentiment",
        "label": "emotion_sentiment",
        "kind": "concept",
        "match_terms": ["emotion", "emotions", "sentiment", "opinions", "duygu analizi"],
        "terms": ["emotion", "sentiment", "opinions", "classification"],
    },
    {
        "key": "time_series_history",
        "label": "time_series_history",
        "kind": "concept",
        "match_terms": ["time series", "historical values", "past measurements", "sensor measurements", "future values"],
        "terms": ["time series", "historical values", "past measurements", "sensor measurements", "future values"],
    },
    {
        "key": "predictive_modeling",
        "label": "predictive_modeling",
        "kind": "task",
        "match_terms": ["prediction", "forecasting", "regression", "valuation", "risk modeling", "impact analysis"],
        "terms": ["prediction", "forecasting", "regression", "risk modeling", "impact analysis"],
    },
    {
        "key": "detection_classification",
        "label": "detection_classification",
        "kind": "task",
        "match_terms": ["fraud detection", "anomaly detection", "classification", "moderation"],
        "terms": ["fraud detection", "anomaly detection", "classification", "moderation"],
    },
    {
        "key": "change_monitoring",
        "label": "change_monitoring",
        "kind": "task",
        "match_terms": ["change detection", "monitoring", "segmentation"],
        "terms": ["change detection", "monitoring", "segmentation"],
    },
]


def aspect_group_matches(group: dict, evidence_text: str, evidence_tokens: set[str]) -> bool:
    for term in group.get("match_terms", []):
        normalized_term = normalize_text(term)
        if not normalized_term:
            continue
        if " " in normalized_term:
            if normalized_term in evidence_text:
                return True
            continue
        if normalized_term in evidence_tokens:
            return True
    return False


def build_semantic_aspects(
    intent_body: str,
    raw_focus_terms: list[str],
    concept_phrases: list[str],
    concept_terms: list[str],
    domains: list[str],
    task_terms: list[str],
) -> list[dict]:
    evidence_chunks = [
        intent_body,
        " ".join(raw_focus_terms),
        " ".join(concept_phrases),
        " ".join(concept_terms),
        " ".join(domain.replace("_", " ") for domain in domains),
        " ".join(task_terms),
    ]
    evidence_text = normalize_text(" ".join(chunk for chunk in evidence_chunks if chunk))
    evidence_tokens = set(tokenize(evidence_text))
    aspects = []

    for group in SEMANTIC_ASPECT_GROUPS:
        if not aspect_group_matches(group, evidence_text, evidence_tokens):
            continue

        aspect_terms = unique_preserve_order(group.get("terms", []))
        if group.get("kind") == "task":
            aspect_terms = unique_preserve_order(task_terms + aspect_terms)
        elif concept_phrases:
            matched_concepts = [
                phrase
                for phrase in concept_phrases
                if any(normalize_text(match_term) in normalize_text(phrase) for match_term in group.get("match_terms", []))
            ]
            aspect_terms = unique_preserve_order(matched_concepts + aspect_terms)

        aspect_text = " ".join(aspect_terms[:5]).strip()
        if not aspect_text:
            continue
        aspects.append(
            {
                "key": group["key"],
                "label": group["label"],
                "text": aspect_text,
                "kind": group["kind"],
            }
        )

    if len(aspects) < 2:
        fallback_aspects = []
        if domains:
            fallback_aspects.append(
                {
                    "key": "domain_focus",
                    "label": "domain_focus",
                    "text": " ".join(domain.replace("_", " ") for domain in domains[:2]),
                    "kind": "fallback",
                }
            )
        if task_terms:
            fallback_aspects.append(
                {
                    "key": "task_focus",
                    "label": "task_focus",
                    "text": " ".join(task_terms[:3]),
                    "kind": "fallback",
                }
            )
        if concept_phrases:
            fallback_aspects.append(
                {
                    "key": "concept_focus",
                    "label": "concept_focus",
                    "text": " ".join(concept_phrases[:4]),
                    "kind": "fallback",
                }
            )
        for aspect in fallback_aspects:
            if aspect["text"].strip():
                aspects.append(aspect)

    deduped = []
    seen = set()
    for aspect in aspects:
        key = (aspect["key"], aspect["text"])
        if key in seen:
            continue
        deduped.append(aspect)
        seen.add(key)

    return deduped[:4]


def build_semantic_variants(
    query: str,
    intent_body: str,
    raw_focus_terms: list[str],
    concept_phrases: list[str],
    concept_terms: list[str],
    domains: list[str],
    task_terms: list[str],
    detected_language: str,
) -> list[dict]:
    # These variants are intentionally structured around domain, task, and
    # multi-aspect intent so query embeddings better match description-driven
    # dataset discovery instead of collapsing to a single keyword cluster.
    raw_focus_query = " ".join(
        unique_preserve_order(
            token
            for token in tokenize(intent_body or query)
            if token not in STOPWORDS
        )[:8]
    )
    concept_phrase_query = " ".join(concept_phrases[:4])
    concept_query = " ".join(concept_terms[:10])
    domain_query = " ".join(domain.replace("_", " ") for domain in domains[:2])
    task_query = " ".join(unique_preserve_order(token for task in task_terms for token in tokenize(task))[:6])
    combined_query = " ".join(
        part
        for part in [
            intent_body,
            concept_phrase_query,
            task_query,
        ]
        if part
    ).strip()
    has_projection = bool(concept_terms)
    original_weight = 0.85 if detected_language == "tr" and has_projection else 1.0
    intent_weight = 1.05 if has_projection else 1.0
    # When the original query is verbose (long paragraph), the dense embedding
    # gets diluted by filler words. Down-weight the raw query and lift the
    # intent / concept variants so the topic-distilled views drive ranking.
    original_token_count = len(tokenize(query or ""))
    if original_token_count >= 15:
        original_weight *= 0.55
        intent_weight = max(intent_weight, 1.35)
    concept_weight = 1.45 if detected_language == "tr" else 1.25
    task_weight = 1.35 if task_query else 1.0
    domain_weight = 1.35 if detected_language == "tr" else 1.15
    bridge_weight = 1.25 if detected_language == "tr" and has_projection else 1.05
    combined_weight = 1.5 if detected_language == "tr" and has_projection else 1.3
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
    add(intent_body, intent_weight)
    add(raw_focus_query, 1.0)
    add(concept_phrase_query, concept_weight)
    add(concept_query, concept_weight)
    add(task_query, task_weight)
    add(combined_query, combined_weight)
    add(f"{raw_focus_query} {concept_query}".strip(), bridge_weight)
    add(f"{concept_phrase_query} {task_query}".strip(), max(bridge_weight, task_weight))
    add(f"{intent_body} {task_query}".strip(), max(intent_weight, task_weight))
    add(f"dataset for {raw_focus_query}" if raw_focus_query else "", 1.0)
    add(f"dataset for {task_query}" if task_query else "", task_weight)
    add(f"dataset about {concept_phrase_query or concept_query}" if (concept_phrase_query or concept_query) else "", concept_weight)
    add(
        f"{domain_query} dataset {task_query} {concept_phrase_query or concept_query}".strip()
        if domain_query or task_query or concept_query or concept_phrase_query else "",
        domain_weight,
    )
    return variants


def build_query_plan(query: str) -> dict:
    normalized = normalize_text(query)
    detected_language = detect_language(query, normalized)
    intent_body = derive_intent_body(query)
    raw_focus_terms = unique_preserve_order(extract_focus_terms(intent_body or normalized))
    concept_phrases, concept_terms, domains, task_terms = collect_concept_matches(normalized, raw_focus_terms)
    domain_terms = unique_preserve_order(
        token
        for domain in domains
        for token in tokenize(domain.replace("_", " "))
        if token not in STOPWORDS
    )
    task_tokens = unique_preserve_order(
        token
        for task in task_terms
        for token in tokenize(task)
        if token not in STOPWORDS
    )
    focus_terms = unique_preserve_order(raw_focus_terms + concept_terms + task_tokens)
    lexical_query = " ".join(focus_terms) if focus_terms else normalized
    concise_intent = " ".join((concept_terms or raw_focus_terms or task_tokens)[:8]) or normalized
    semantic_variants = build_semantic_variants(
        query,
        intent_body,
        raw_focus_terms,
        concept_phrases,
        concept_terms,
        domains,
        task_terms,
        detected_language,
    )
    # Aspect groups let semantic retrieval separately encode the major parts
    # of a dataset-discovery query so results that cover multiple aspects can
    # outrank records that only match one dominant term.
    semantic_aspects = build_semantic_aspects(
        intent_body,
        raw_focus_terms,
        concept_phrases,
        concept_terms,
        domains,
        task_terms,
    )
    semantic_queries = [item["text"] for item in semantic_variants]

    return {
        "original_query": query.strip(),
        "normalized_query": normalized,
        "detected_language": detected_language,
        "intent_body": intent_body,
        "raw_focus_terms": raw_focus_terms,
        "focus_terms": focus_terms,
        "concept_phrases": concept_phrases,
        "concept_terms": concept_terms,
        "domains": domains,
        "domain_terms": domain_terms,
        "task_terms": task_terms,
        "lexical_query": lexical_query,
        "concise_intent": concise_intent,
        "semantic_aspects": semantic_aspects,
        "semantic_variants": semantic_variants,
        "semantic_queries": semantic_queries,
        "has_multi_aspect_intent": len(semantic_aspects) >= 2,
        "is_sentence_query": len(tokenize(query)) >= 6,
    }
