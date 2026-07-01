# Outfit Completion Pipeline — Fashion Florence Edition

AI outfit completion using Fashion Florence (fine-tuned Florence-2) for fashion attribute
extraction, FashionCLIP as a color fallback, sentence-transformers for search embeddings,
Qdrant for vector search, and Ollama/mistral for styling explanations. Each recommended
product also gets an automatically generated **image description** (Florence-2 caption)
alongside the styling explanation.

## Pipeline overview

```
Image + text
   ↓
Stage 1   Fashion Florence → category, color, material, style (JSON)
          (FashionCLIP zero-shot fallback if color = "unknown")
   ↓
Stage 2   Profile builder → merges attributes + user text
   ↓
Stage 3a  sentence-transformers → embed query              [local]
Stage 3b  HSL color harmony engine → compatible colors      [rule-based]
   ↓
Stage 4   Qdrant → vector search with category filters
   ↓
Stage 5   Re-ranker → weighted score (semantic + color + freshness)
   ↓
Stage 6   Ollama/mistral → styling explanation per result
Stage 6b  Florence-2 captioning → image description per result
   ↓
Result cards: image, name, price, URL, image_description, styling_explanation
```

---

## Option A — Docker (recommended)

Everything runs in containers: the app (Fashion Florence, FashionCLIP, embeddings, Qdrant)
and Ollama (mistral) as a separate service.

### 1. Put your data in place

```bash
mkdir -p data
cp /path/to/modanisa_products.csv data/
```

### 2. Build and start

```bash
docker compose up -d --build
```

This starts two containers:
- `outfit_ollama` — serves the mistral model
- `outfit_app` — runs the pipeline (built from `Dockerfile`)

### 3. Pull the mistral model into the Ollama container (one-time)

```bash
docker exec outfit_ollama ollama pull mistral
```

### 4. Build the product index (one-time, ~10-20 min on CPU)

```bash
docker exec -it outfit_app python outfit_pipeline.py --index
```

Qdrant storage persists to `./data/qdrant_storage` on your host, so you only need to do
this once — it survives container restarts.

### 5. Run a query

```bash
# Copy a test image into data/ first
cp my_pants.jpg data/

docker exec -it outfit_app python outfit_pipeline.py \
    --image /app/data/my_pants.jpg \
    --text "I want a blouse for work"
```

### 6. Stop everything

```bash
docker compose down          # stop containers, keep volumes (models, qdrant data)
docker compose down -v       # stop and wipe everything including downloaded models
```

### GPU support (optional, much faster)

If you have an NVIDIA GPU, install `nvidia-container-toolkit`, then uncomment the
`deploy.resources` block in `docker-compose.yml` for both the `app` and `ollama` services.

---

## Option B — Local virtual environment (no Docker)

Use this for development/debugging directly on your machine.

### 1. Create the venv

```bash
chmod +x setup_venv.sh
./setup_venv.sh
source venv/bin/activate
```

### 2. Install and run Ollama separately

```bash
# Install from https://ollama.com/download
ollama serve &
ollama pull mistral
```

### 3. Set environment variables

```bash
export OLLAMA_HOST=http://localhost:11434
export CSV_PATH=./data/modanisa_products.csv
export QDRANT_PATH=./data/qdrant_storage
```

### 4. Build index and query

```bash
python app/outfit_pipeline.py --index
python app/outfit_pipeline.py --image data/my_pants.jpg --text "I want a blouse"
```

---

## Models used (all free)

| Stage | Model | Size | Purpose |
|-------|-------|------|---------|
| 1 | `anushreeberlia/fashion-florence` | 0.77B | Structured attribute extraction (category/color/material/style) |
| 1 (fallback) | `patrickjohncyh/fashion-clip` | ~150M | Zero-shot color classification when Fashion Florence is unsure |
| 3a | `paraphrase-multilingual-MiniLM-L12-v2` | ~120MB | Text embeddings for search (Arabic + English) |
| 6 | `mistral` (via Ollama) | ~4.1GB | Styling explanation generation |
| 6b | `microsoft/Florence-2-base-ft` | ~460MB | Image captioning — generates the "image description" for each result |

All models download automatically on first run and are cached in the `hf_cache` Docker
volume (or `~/.cache/huggingface` for the venv setup), so subsequent runs are fast.

## Project structure

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup_venv.sh
├── .env.example
├── .dockerignore
├── app/
│   └── outfit_pipeline.py
├── data/                  ← put your CSV and test images here
│   └── modanisa_products.csv
└── outputs/                ← generated results land here
```

## Notes

- Qdrant runs embedded (no separate server) with persistent on-disk storage at
  `QDRANT_PATH`, so the index survives restarts.
- First run downloads ~1GB of model weights total — subsequent runs are instant
  thanks to the cache volume.
- To swap the explanation model, change `OLLAMA_TEXT_MODEL` in `.env` or
  `docker-compose.yml` (e.g. `llama3.1`, `gemma2`).
# Outfit-Completion
# outfit_completion_docker
# outfit_completion_docker
