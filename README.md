# VisualFind — Multimodal Product Search

A production-quality portfolio project demonstrating **multimodal AI search** using OpenAI's CLIP model, FAISS vector indexing, and Streamlit. Search a product catalogue using text, images, or both at the same time.

---

## What It Does

Traditional e-commerce search matches keywords. **VisualFind** maps products and queries into a shared *semantic embedding space* — so searching for *"something cosy to wear in winter"* can surface a merino wool sweater even if those exact words don't appear in the product listing. Upload a photo of a chair you like and find visually similar ones. Combine both for pinpoint precision.

### Three Search Modes

| Mode | Input | How it works |
|------|-------|-------------|
| **Text** | Natural language string | CLIP text encoder → 512-d vector → FAISS nearest-neighbour |
| **Image** | Uploaded photo (JPG/PNG) | CLIP image encoder → 512-d vector → FAISS nearest-neighbour |
| **Hybrid** | Text + Image | Weighted average of both embeddings → FAISS search |

---

## Architecture

```
VisualFind
├── data/
│   ├── build_dataset.py     # Generates synthetic product catalogue (CSV)
│   └── products.csv         # 100+ products across 5 categories (generated)
│
├── embeddings/
│   ├── build_embeddings.py  # CLIP-encodes all products → FAISS index
│   ├── faiss_index.bin      # FAISS IndexFlatIP (inner product) (generated)
│   ├── text_embeddings.npy  # (N, 512) float32 embedding matrix (generated)
│   └── product_ids.json     # Ordered product ID list (generated)
│
├── utils/
│   └── search_engine.py     # ProductSearchEngine class — core retrieval logic
│
├── app.py                   # Streamlit UI
├── requirements.txt
└── README.md
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Multimodal embeddings | [OpenAI CLIP ViT-B/32](https://openai.com/research/clip) via `open-clip-torch` |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) `IndexFlatIP` |
| UI | [Streamlit](https://streamlit.io) |
| Data | Pandas, NumPy |
| Visualisation | Plotly |

---

## Quickstart

### 1. Clone & install dependencies

```bash
git clone https://github.com/yourname/visualfind.git
cd visualfind
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** First run downloads CLIP weights (~350 MB). Requires ~2 GB RAM minimum; GPU optional but recommended for embedding generation.

### 2. Build the product catalogue

```bash
python data/build_dataset.py
```

Generates `data/products.csv` with ~100 products across 5 categories (Electronics, Furniture, Clothing, Kitchen, Sports) with realistic names, descriptions, prices, ratings, and Unsplash image URLs.

### 3. Generate CLIP embeddings

```bash
python embeddings/build_embeddings.py
```

- Downloads CLIP ViT-B/32 weights (once)
- Encodes every product's text description into a 512-dimensional vector
- Builds a FAISS `IndexFlatIP` (exact inner-product search on L2-normalised vectors = cosine similarity)
- Saves `faiss_index.bin`, `text_embeddings.npy`, `product_ids.json`

### 4. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

Alternatively, use the **setup wizard** built into the app — it has buttons to run steps 2 and 3 directly from the UI.

---

## 💡 How the Search Works (Deep Dive)

### CLIP Embedding Space

CLIP (Contrastive Language–Image Pretraining) is trained on 400 million image-text pairs to place semantically related images and text **near each other** in the same 512-dimensional vector space. This means:

- *"a person wearing a red jacket"* (text) is close to actual red jacket photos (images)
- Products are encoded as text: `"{name}. {description}. Tags: {tags}."`

### Indexing

Each product text embedding is stored in a **FAISS `IndexFlatIP`** (flat, inner product). Because all vectors are L2-normalised before insertion, inner product == cosine similarity. This gives exact results (no approximation), fast enough for catalogues up to ~1M items on CPU.

### Query Time

```
User query
    ↓
CLIP text encoder  OR  CLIP image encoder  OR  weighted average of both
    ↓
512-d query vector (L2-normalised)
    ↓
FAISS.search(query_vector, top_k * 5)  → candidate product IDs + scores
    ↓
Post-filter (category, price, rating, stock)
    ↓
Top-K results rendered as product cards
```

### Hybrid Fusion

When both text and image are provided, the query embedding is a weighted combination:

```
q_hybrid = α * q_text + (1 - α) * q_image
q_hybrid = q_hybrid / ‖q_hybrid‖    # re-normalise
```

The slider in the sidebar controls α (0 = pure image, 1 = pure text).

---

## Key Files Explained

### `data/build_dataset.py`
Defines a dictionary of product templates (name, description, tags, price range, Unsplash image query) across 5 categories. Generates 2–4 randomised variants per template for a realistic-sized catalogue.

### `embeddings/build_embeddings.py`
Uses `open_clip` to load the ViT-B/32 CLIP model. Encodes all product texts in batches of 64, L2-normalises the outputs, and inserts them into a FAISS index. Runs in ~60s on CPU, ~5s on GPU.

### `utils/search_engine.py`
The `ProductSearchEngine` class is lazy-initialised (Streamlit's `@st.cache_resource` ensures it loads once). Public method `.search()` accepts any combination of text query and PIL Image, performs FAISS lookup, and applies post-filters before returning a ranked DataFrame.

### `app.py`
Streamlit UI with:
- Three tabs (Text / Image / Hybrid)
- Sidebar filters (category, price, rating, stock, grid columns, hybrid weight)
- Responsive CSS card grid with similarity scores
- Landing dashboard showing catalogue stats and bar chart
- First-run setup wizard with one-click setup buttons

---

## 🔧 Extending the Project

| Idea | How |
|------|-----|
| Real product images | Replace Unsplash URLs with real downloaded images; encode them with the CLIP image encoder instead of text |
| Approximate search | Swap `IndexFlatIP` for `IndexIVFFlat` or `IndexHNSWFlat` for million-scale catalogues |
| Re-ranking | Add a cross-encoder re-ranker (e.g. BLIP-2) on the top-50 candidates |
| Personalisation | Store user click history and blend a user preference vector into the query |
| Multi-language | Switch to `xlm-roberta-large-ViT-H-14` CLIP variant for multilingual queries |
| Vector DB | Replace FAISS with Pinecone, Weaviate or Qdrant for production deployments |

---

## Screenshots

```

---

## 📄 License

MIT

---

*Built as a portfolio project demonstrating multimodal ML engineering with CLIP, FAISS, and Streamlit.*
