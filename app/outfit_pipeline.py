"""
Outfit Completion Pipeline — Fashion Florence Edition
=======================================================
Stages:
  0. User uploads image + optional text prompt
  1. Fashion Florence (fine-tuned Florence-2, 0.77B) → structured attribute JSON
     - category, color, material, style — purpose-trained on fashion, 94.6% accuracy
     - FashionCLIP zero-shot fallback when color is "unknown"
  2. Profile builder → merge vision output + user text into OutfitProfile
  3a. sentence-transformers (multilingual MiniLM) → embed query text   [local]
  3b. HSL color harmony engine → compatible color families             [rule-based]
  4. Qdrant in-memory → HNSW search with payload filters
  5. Multi-dimension re-ranker → weighted score
  6. Ollama (mistral) → styling explanation (why this completes the outfit)
  6b. Florence-2 captioning → natural-language description of EACH result's
      product image (<MORE_DETAILED_CAPTION> task) — this is the "image
      description" requested for each result
  7. Result cards: image, name, price, URL, image_description, styling_explanation

Models used (all free / local):
  - anushreeberlia/fashion-florence  → attribute extraction (Stage 1)
  - microsoft/Florence-2-base-ft     → image captioning for results (Stage 6b)
  - patrickjohncyh/fashion-clip      → zero-shot color fallback (Stage 1)
  - paraphrase-multilingual-MiniLM   → text embeddings (Stage 3a)
  - mistral (via Ollama)             → styling explanation text (Stage 6)
"""

import os
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter,
    FieldCondition, MatchAny
)

try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_PATH           = os.environ.get("CSV_PATH", "/app/data/modanisa_products.csv")
QDRANT_PATH         = os.environ.get("QDRANT_PATH", "/app/data/qdrant_storage")
COLLECTION_NAME      = "modanisa_products"
EMBED_MODEL_NAME     = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DIM           = 384
TOP_K_RETRIEVE       = 30
TOP_K_FINAL          = 5

FASHION_FLORENCE_ID  = "anushreeberlia/fashion-florence"   # attribute extraction
CAPTION_MODEL_ID      = "microsoft/Florence-2-base-ft"      # generic captioning (image description)
TEXT_MODEL            = os.environ.get("OLLAMA_TEXT_MODEL", "mistral")
OLLAMA_HOST           = os.environ.get("OLLAMA_HOST", "http://ollama:11434")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ─────────────────────────────────────────────
# COLOR HARMONY ENGINE (Stage 3b — zero cost, deterministic)
# ─────────────────────────────────────────────

COLOR_HUE_MAP = {
    "black": None, "white": None, "cream": None, "vanilla": None,
    "beige": None, "ecru": None, "off-white": None, "ivory": None,
    "gray": None, "grey": None, "silver": None, "stone": None, "mink": None,
    "navy": 220, "navy blue": 220,
    "blue": 210, "light blue": 200, "sky blue": 195,
    "red": 0, "cherry": 355, "burgundy": 340, "wine": 345, "maroon": 350,
    "pink": 330, "rose": 340, "blush": 340, "dusty rose": 345,
    "purple": 270, "violet": 280, "lilac": 290, "lavender": 285,
    "green": 120, "olive": 80, "sage": 140, "mint": 150, "emerald": 140,
    "yellow": 50, "lemon yellow": 55, "mustard": 45, "gold": 45,
    "orange": 25, "peach": 20, "terracotta": 15, "rust": 10,
    "brown": 25, "camel": 35, "tan": 30, "chocolate": 25,
    "teal": 180, "turquoise": 175, "aqua": 185,
    "fuchsia": 310, "magenta": 300,
}

NEUTRAL_COLORS = {
    "black", "white", "cream", "vanilla", "beige", "ecru",
    "off-white", "ivory", "gray", "grey", "silver", "stone",
    "mink", "nude", "camel", "tan"
}

SKIN_TONE_COMPATIBLE = {
    "fair":    ["blue", "purple", "green", "navy", "pink", "teal"],
    "light":   ["blue", "purple", "green", "navy", "burgundy", "rose"],
    "medium":  ["orange", "rust", "olive", "brown", "gold", "purple"],
    "olive":   ["rust", "orange", "olive", "teal", "burgundy", "green"],
    "tan":     ["orange", "gold", "red", "olive", "brown", "white"],
    "dark":    ["orange", "gold", "red", "white", "cream", "yellow"],
    "unknown": []
}

BODY_SHAPE_NOTES = {
    "hourglass": "fitted waist, structured",
    "pear":      "dark bottom, flowy top, A-line",
    "apple":     "flowy, empire waist, V-neck",
    "rectangle": "belted, defined waist, layered",
    "plus":      "dark solid, vertical lines, wrap",
    "unknown":   ""
}


def color_to_hue(color_name: str) -> Optional[float]:
    c = color_name.lower().strip()
    for key, hue in COLOR_HUE_MAP.items():
        if key in c:
            return hue
    return None


def harmony_families(anchor_color: str) -> list:
    anchor_lower = anchor_color.lower()
    for n in NEUTRAL_COLORS:
        if n in anchor_lower:
            return ["blue", "red", "green", "purple", "orange", "pink",
                    "teal", "yellow", "brown", "navy", "black", "white",
                    "beige", "cream", "gray"]
    hue = color_to_hue(anchor_color)
    if hue is None:
        return []
    comp_hue = (hue + 180) % 360
    compatible = []
    for h_name, h_val in COLOR_HUE_MAP.items():
        if h_val is None:
            continue
        dist      = min(abs(h_val - hue), 360 - abs(h_val - hue))
        dist_comp = min(abs(h_val - comp_hue), 360 - abs(h_val - comp_hue))
        if dist <= 40 or dist_comp <= 40:
            compatible.append(h_name)
    compatible += list(NEUTRAL_COLORS)
    return list(set(compatible))


def skin_tone_colors(skin_tone: str) -> list:
    return SKIN_TONE_COMPATIBLE.get(skin_tone.lower(), [])


# ─────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────

@dataclass
class OutfitProfile:
    anchor_color:     str  = "unknown"
    anchor_type:      str  = "unknown"
    anchor_style:     str  = "casual"
    anchor_material:  str  = "unknown"
    target_category:  str  = ""
    body_shape:       str  = "unknown"
    skin_tone:        str  = "unknown"
    occasion:         str  = "daily"
    user_request:     str  = ""
    harmony_families: list = field(default_factory=list)


@dataclass
class ProductResult:
    product_id:        int
    name:               str
    category:           str
    color:               str
    price:               str
    image_url:           str
    product_url:         str
    score:               float
    image_description:   str = ""   # caption generated from the product image
    styling_explanation: str = ""   # why it completes the outfit


# ─────────────────────────────────────────────
# MODEL LOADER (singleton pattern, loaded once)
# ─────────────────────────────────────────────

class ModelRegistry:
    """Lazily loads and caches all vision/embedding models."""
    _fashion_florence = None
    _fashion_florence_proc = None
    _caption_model = None
    _caption_proc = None
    _embed_model = None

    @classmethod
    def fashion_florence(cls):
        if cls._fashion_florence is None:
            print(f"  Loading Fashion Florence ({FASHION_FLORENCE_ID})...")
            cls._fashion_florence_proc = AutoProcessor.from_pretrained(
                FASHION_FLORENCE_ID, trust_remote_code=True
            )
            cls._fashion_florence = AutoModelForCausalLM.from_pretrained(
                FASHION_FLORENCE_ID, trust_remote_code=True, torch_dtype=TORCH_DTYPE
            ).to(DEVICE).eval()
            print("  Fashion Florence ready.")
        return cls._fashion_florence, cls._fashion_florence_proc

    @classmethod
    def caption_model(cls):
        """Generic Florence-2 for image captioning (Stage 6b — image description)."""
        if cls._caption_model is None:
            print(f"  Loading caption model ({CAPTION_MODEL_ID})...")
            cls._caption_proc = AutoProcessor.from_pretrained(
                CAPTION_MODEL_ID, trust_remote_code=True
            )
            cls._caption_model = AutoModelForCausalLM.from_pretrained(
                CAPTION_MODEL_ID, trust_remote_code=True, torch_dtype=TORCH_DTYPE
            ).to(DEVICE).eval()
            print("  Caption model ready.")
        return cls._caption_model, cls._caption_proc

    @classmethod
    def embed_model(cls):
        if cls._embed_model is None:
            print(f"  Loading embedding model ({EMBED_MODEL_NAME})...")
            cls._embed_model = SentenceTransformer(EMBED_MODEL_NAME)
            print("  Embedding model ready.")
        return cls._embed_model


# ─────────────────────────────────────────────
# STAGE 1 — FASHION FLORENCE ATTRIBUTE EXTRACTION
# ─────────────────────────────────────────────

def _florence_generate(model, processor, image: Image.Image, prompt: str) -> str:
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, TORCH_DTYPE)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=512,
        num_beams=3,
        do_sample=False,
    )
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=prompt, image_size=(image.width, image.height)
    )
    return parsed.get(prompt, text)


def extract_fashion_attributes(image_path: str) -> dict:
    """
    Stage 1: Run Fashion Florence on the garment image.
    Returns category, color, material, style as a dict.
    Falls back to "unknown" fields if the model output can't be parsed.
    """
    model, processor = ModelRegistry.fashion_florence()
    image = Image.open(image_path).convert("RGB")

    # Fashion Florence is fine-tuned with a task-prefix prompt for structured tagging
    raw = _florence_generate(model, processor, image, "<FASHION_ATTRIBUTES>")

    # Output is typically JSON-like; extract and parse
    raw_str = raw if isinstance(raw, str) else json.dumps(raw)
    m = re.search(r"\{.*\}", raw_str, re.DOTALL)
    json_str = m.group(0) if m else raw_str

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        data = _loose_parse(raw_str)

    return {
        "category": data.get("category", "unknown"),
        "color":    data.get("color", "unknown"),
        "material": data.get("material", "unknown"),
        "style":    data.get("style", "casual"),
    }


def _loose_parse(raw: str) -> dict:
    result = {}
    for key in ["category", "color", "material", "style"]:
        m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', raw)
        if m:
            result[key] = m.group(1)
    return result


def fashionclip_color_fallback(image_path: str) -> str:
    """
    Zero-shot color classification fallback when Fashion Florence returns 'unknown'.
    Uses FashionCLIP for zero-shot matching against a candidate color list.
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        return "unknown"

    candidates = list(set(
        k for k in COLOR_HUE_MAP.keys() if " " not in k or k in NEUTRAL_COLORS
    ))

    model_id = "patrickjohncyh/fashion-clip"
    if not hasattr(fashionclip_color_fallback, "_model"):
        fashionclip_color_fallback._model = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
        fashionclip_color_fallback._proc = CLIPProcessor.from_pretrained(model_id)

    model = fashionclip_color_fallback._model
    proc  = fashionclip_color_fallback._proc

    image = Image.open(image_path).convert("RGB")
    texts = [f"a {c} colored garment" for c in candidates]
    inputs = proc(text=texts, images=image, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    best_idx = probs.argmax().item()
    return candidates[best_idx]


# ─────────────────────────────────────────────
# STAGE 2 — PROFILE BUILDER
# ─────────────────────────────────────────────

TARGET_CATEGORY_RULES = {
    # If anchor is X, default completion target is Y (overridden by user text)
    "pants":   "blouse",
    "trousers":"blouse",
    "skirt":   "top",
    "blouse":  "pants",
    "top":     "skirt",
    "dress":   "outerwear",
    "abaya":   "hijab",
}


def build_profile(attrs: dict, image_path: str, user_text: str = "") -> OutfitProfile:
    """Stage 2: Merge Fashion Florence attributes + user text into OutfitProfile."""
    color = attrs.get("color", "unknown")
    if color == "unknown":
        print("  [Stage 1 fallback] Color unknown — running FashionCLIP zero-shot...")
        color = fashionclip_color_fallback(image_path)

    anchor_type = attrs.get("category", "unknown")
    target = user_text.strip() if user_text.strip() else TARGET_CATEGORY_RULES.get(
        anchor_type.lower(), "top"
    )

    profile = OutfitProfile(
        anchor_color    = color,
        anchor_type     = anchor_type,
        anchor_style    = attrs.get("style", "casual"),
        anchor_material = attrs.get("material", "unknown"),
        target_category = target,
        user_request    = user_text,
    )
    profile.harmony_families = harmony_families(profile.anchor_color)

    print(f"  → {profile.anchor_color} {profile.anchor_type} ({profile.anchor_material}) | "
          f"style={profile.anchor_style} | target={profile.target_category}")
    return profile


# ─────────────────────────────────────────────
# STAGE 3a — EMBEDDING
# ─────────────────────────────────────────────

def build_product_text(row: pd.Series) -> str:
    parts = [
        str(row.get("name", "")),
        str(row.get("category", "")).replace("-", " "),
        str(row.get("colors", "")),
        str(row.get("fabrics", "")),
    ]
    return " | ".join(p for p in parts if p and p != "nan")


def build_query_text(profile: OutfitProfile) -> str:
    parts = [
        profile.target_category,
        profile.anchor_style,
        profile.anchor_color,
        profile.user_request,
    ]
    return " ".join(p for p in parts if p)


# ─────────────────────────────────────────────
# OFFLINE INDEXER
# ─────────────────────────────────────────────

def build_index(csv_path: str, qdrant: QdrantClient):
    embed_model = ModelRegistry.embed_model()
    print(f"Loading catalog from {csv_path}...")
    df = pd.read_csv(csv_path).fillna("")
    print(f"  {len(df)} products loaded.")

    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )

    BATCH = 256
    total, points = len(df), []
    for i, (_, row) in enumerate(df.iterrows()):
        text   = build_product_text(row)
        vector = embed_model.encode(text, normalize_embeddings=True).tolist()
        payload = {
            "product_id":  int(row["id"]),
            "name":        str(row["name"]),
            "category":    str(row["category"]),
            "color":       str(row["colors"]).lower(),
            "price":       str(row["price"]),
            "image_url":   str(row["image_url"]),
            "product_url": str(row["product_url"]),
            "is_new":      bool(row.get("is_new", False)),
        }
        points.append(PointStruct(id=i, vector=vector, payload=payload))
        if len(points) == BATCH or i == total - 1:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
            print(f"  Indexed {i+1}/{total} ({(i+1)/total*100:.0f}%)", end="\r")
    print(f"\nIndex built: {total} products.")


# ─────────────────────────────────────────────
# STAGE 4 — QDRANT SEARCH
# ─────────────────────────────────────────────

CATEGORY_MAP = {
    "hijab": ["hijab"], "scarf": ["hijab"],
    "blouse": ["modest-clothing", "flash-deals-12"],
    "top": ["modest-clothing", "flash-deals-12"],
    "shirt": ["modest-clothing", "flash-deals-12"],
    "tunic": ["modest-clothing", "flash-deals-12"],
    "dress": ["modest-clothing", "dresses-evening-gowns-z1-en-en"],
    "abaya": ["modest-clothing", "outerwear"],
    "pants": ["modest-clothing", "flash-deals-12"],
    "skirt": ["modest-clothing", "flash-deals-12"],
    "suit": ["modest-clothing", "outerwear"],
    "coat": ["outerwear"], "jacket": ["outerwear"],
    "cardigan": ["modest-clothing", "outerwear"],
    "swimwear": ["sea"], "evening": ["dresses-evening-gowns-z1-en-en"],
    "plus size": ["plus-size-clothing"],
    "shoes": ["shoes-bag"], "bag": ["shoes-bag"],
}


def map_to_categories(target: str) -> list:
    t = target.lower()
    for key, cats in CATEGORY_MAP.items():
        if key in t:
            return cats
    return ["modest-clothing", "flash-deals-12", "outerwear"]


def search_qdrant(profile: OutfitProfile, query_vector: list, qdrant: QdrantClient) -> list:
    target_cats = map_to_categories(profile.target_category)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=TOP_K_RETRIEVE,
        query_filter=Filter(must=[
            FieldCondition(key="category", match=MatchAny(any=target_cats))
        ]),
        with_payload=True,
    )
    return [{**r.payload, "_qdrant_score": r.score} for r in results]


# ─────────────────────────────────────────────
# STAGE 5 — RE-RANKER
# ─────────────────────────────────────────────

def score_product(product: dict, profile: OutfitProfile) -> float:
    prod_color = product.get("color", "").lower()
    prod_name  = product.get("name", "").lower()

    s_semantic = float(product.get("_qdrant_score", 0.5))

    s_color = 0.8
    if profile.harmony_families:
        s_color = 0.0
        for fam in profile.harmony_families:
            if fam in prod_color or fam in prod_name:
                s_color = 1.0
                break
        for n in NEUTRAL_COLORS:
            if n in profile.anchor_color.lower():
                s_color = 1.0
                break

    s_fresh = 0.6 + (0.4 if product.get("is_new") else 0.0)

    return round(0.5 * s_semantic + 0.35 * s_color + 0.15 * s_fresh, 4)


def rerank(candidates: list, profile: OutfitProfile) -> list:
    for p in candidates:
        p["_final_score"] = score_product(p, profile)
    return sorted(candidates, key=lambda x: x["_final_score"], reverse=True)[:TOP_K_FINAL]


# ─────────────────────────────────────────────
# STAGE 6 — STYLING EXPLANATION  (Ollama / mistral)
# ─────────────────────────────────────────────

EXPLANATION_SYSTEM = """You are a friendly fashion stylist for a modest fashion platform.
Given an anchor garment and a suggested completion item, explain in 2-3 sentences why they work together.
Be specific: color harmony, style match, occasion suitability.
Write in English, or mix Arabic/English if the user's request was in Arabic. Be warm and practical."""


def explain_styling(profile: OutfitProfile, product: dict) -> str:
    if not OLLAMA_AVAILABLE:
        return f"This {product.get('color','')} {product.get('name','')} pairs well with your {profile.anchor_color} {profile.anchor_type}."
    try:
        client = _ollama.Client(host=OLLAMA_HOST)
        prompt = (
            f"Anchor item: {profile.anchor_color} {profile.anchor_type} ({profile.anchor_style} style)\n"
            f"Suggested: {product.get('name','')} — color: {product.get('color','')}\n"
            f"User request: {profile.user_request or 'complete the outfit'}\n\n"
            f"Explain in 2-3 sentences why this is a great outfit completion."
        )
        response = client.chat(model=TEXT_MODEL, messages=[
            {"role": "system", "content": EXPLANATION_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        return response["message"]["content"].strip()
    except Exception as e:
        return f"(Explanation unavailable — Ollama error: {e})"


# ─────────────────────────────────────────────
# STAGE 6b — IMAGE DESCRIPTION  (Florence-2 captioning)
# This is the per-result "image description" requested by the user.
# Generated from each recommended PRODUCT's image, not the anchor.
# ─────────────────────────────────────────────

def describe_product_image(image_url: str) -> str:
    """
    Stage 6b: Download the product image and generate a natural-language
    description using Florence-2's <MORE_DETAILED_CAPTION> task.
    """
    import requests
    from io import BytesIO

    try:
        resp = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        return f"(Could not load image: {e})"

    model, processor = ModelRegistry.caption_model()
    caption = _florence_generate(model, processor, image, "<MORE_DETAILED_CAPTION>")
    return caption if isinstance(caption, str) else str(caption)


# ─────────────────────────────────────────────
# MAIN PIPELINE CLASS
# ─────────────────────────────────────────────

class OutfitPipeline:
    """
    Outfit completion pipeline using Fashion Florence for attribute extraction.

    Usage:
        pipeline = OutfitPipeline()
        pipeline.build_index()
        results, profile = pipeline.complete_outfit("pants.jpg", "I want a blouse")
    """

    def __init__(self):
        Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        self.qdrant = QdrantClient(path=QDRANT_PATH)   # persistent storage
        print(f"  Qdrant ready (persistent at {QDRANT_PATH}).")
        print(f"  Device: {DEVICE}")

    def build_index(self, csv_path: str = CSV_PATH):
        build_index(csv_path, self.qdrant)

    def complete_outfit(
        self,
        image_path: str,
        user_text: str = "",
        explain: bool = True,
        describe_images: bool = True,
    ):
        print(f"\n{'='*60}")
        print(f"Query: {image_path!r}  |  '{user_text}'")

        # Stage 1: Fashion Florence attribute extraction
        print("  [Stage 1] Extracting attributes with Fashion Florence...")
        attrs = extract_fashion_attributes(image_path)

        # Stage 2: Build profile
        profile = build_profile(attrs, image_path, user_text)

        # Stage 3a: Embed query
        print("  [Stage 3a] Embedding query...")
        embed_model = ModelRegistry.embed_model()
        query_text   = build_query_text(profile)
        query_vector = embed_model.encode(query_text, normalize_embeddings=True).tolist()
        print(f"  Query text: '{query_text}'")

        # Stage 4: Qdrant search
        print(f"  [Stage 4] Searching Qdrant (top {TOP_K_RETRIEVE})...")
        candidates = search_qdrant(profile, query_vector, self.qdrant)
        print(f"  Got {len(candidates)} candidates.")
        if not candidates:
            return [], profile

        # Stage 5: Re-rank
        print("  [Stage 5] Re-ranking...")
        top = rerank(candidates, profile)

        # Stage 6 + 6b: Explanation + image description per result
        results = []
        for i, p in enumerate(top):
            result = ProductResult(
                product_id  = p.get("product_id", 0),
                name        = p.get("name", ""),
                category    = p.get("category", ""),
                color       = p.get("color", ""),
                price       = p.get("price", ""),
                image_url   = p.get("image_url", ""),
                product_url = p.get("product_url", ""),
                score       = p["_final_score"],
            )
            if explain:
                print(f"  [Stage 6] Explaining result {i+1}/{len(top)}...")
                result.styling_explanation = explain_styling(profile, p)
            if describe_images:
                print(f"  [Stage 6b] Describing image {i+1}/{len(top)}...")
                result.image_description = describe_product_image(p.get("image_url", ""))
            results.append(result)

        return results, profile


# ─────────────────────────────────────────────
# DISPLAY + CLI
# ─────────────────────────────────────────────

def print_results(results, profile: OutfitProfile):
    print(f"\n{'='*60}")
    print(f"TOP {len(results)} OUTFIT COMPLETIONS")
    print(f"Anchor: {profile.anchor_color} {profile.anchor_type} ({profile.anchor_material})")
    print(f"Searching for: {profile.target_category}")
    print("="*60)
    for i, r in enumerate(results, 1):
        print(f"\n#{i}  [{r.score:.3f}]  {r.name}")
        print(f"  Color       : {r.color}")
        print(f"  Price       : {r.price}")
        print(f"  Image URL   : {r.image_url}")
        print(f"  Product URL : {r.product_url}")
        if r.image_description:
            print(f"  Image desc. : {r.image_description}")
        if r.styling_explanation:
            print(f"  Why it works: {r.styling_explanation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outfit Completion Pipeline (Fashion Florence)")
    parser.add_argument("--index", action="store_true", help="Build Qdrant index from CSV")
    parser.add_argument("--image", type=str, default="", help="Path to garment image")
    parser.add_argument("--text",  type=str, default="", help="User text request")
    args = parser.parse_args()

    pipeline = OutfitPipeline()

    if args.index:
        pipeline.build_index(CSV_PATH)

    if args.image:
        results, profile = pipeline.complete_outfit(
            image_path=args.image, user_text=args.text,
            explain=True, describe_images=True,
        )
        print_results(results, profile)
    elif not args.index:
        print("Usage:")
        print("  python outfit_pipeline.py --index")
        print("  python outfit_pipeline.py --image x.jpg --text 'I want a blouse'")
