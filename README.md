# SOC - Simple Outfit Compatibility

A modular, step-by-step project for building an outfit compatibility and recommendation system, starting from a simple baseline and evolving toward explainable and learning-based models.

---

## 🎯 Project Goal

Given an outfit (image-based input), recommend the **top 5 compatible items** from a local catalog. Future updates will include explainable recommendations and then upgrading to evidence base recom and explanation.

### Current Version

- Uses image embeddings (CLIP)
- Retrieves visually compatible items
- Supports category-based filtering (e.g., shoes, bag, jacket)
- Introduces a structured pipeline for explainable recommendations (LLM-ready)
- Visualization and comparison supporting LLM explanations.

---

## 🧠 What We Built (So Far)

### 🔹 Core Idea

This is a **retrieval-based baseline system**, not yet a fully trained compatibility model.

### Pipeline

```
Query Image
   ↓
CLIP Encoder
   ↓
Embedding Vector
   ↓
Cosine Similarity
   ↓
Top-K Retrieval (Filtered by Category)
   ↓
Structured Explanation Payload (LLM-ready)
```

---

## ⚙️ Methodology

### 1. Image Representation

We use a pretrained **CLIP model (ViT-B/32)** to convert images into embeddings.

- Each image → vector
- Similar items → closer in embedding space

---

### 2. Retrieval (Baseline Recommender)

- Encode all catalog images once
- Encode query image
- Compute **cosine similarity**
- Rank items
- Return **top 5 results**

#### Optional

- Filter by `target_category` (e.g., only shoes)

---

### 3. Explanation Layer (NEW 🚀)

We added a structured **explanation payload builder**:

- Converts recommendations into structured data
- Prepares inputs for an LLM (next step)

### Example Output

```json
{
  "query": {
    "image_path": "...",
    "target_category": "shoes"
  },
  "recommendations": [
    {
      "rank": 1,
      "item_id": 1,
      "category": "shoes",
      "score": 0.83,
      "color": "white",
      "style": "casual"
    }
  ]
}
```

### This Enables

- Explainable recommendations  
- LLM integration (next phase)

---

## 📁 Project Structure

```
SOC/
│
├── notebooks/
│   ├── 01_environment_check.ipynb
│   ├── 02_build_catalog_embeddings.ipynb
│   ├── 03_recommend_top5.ipynb
│
├── src/soc/
│   ├── config.py
│   │
│   ├── data/
│   │   ├── catalog.py
│   │   └── loaders.py
│   │
│   ├── models/
│   │   ├── clip_encoder.py
│   │   └── retriever.py
│   │
│   ├── inference/
│   │   └── recommend.py
│   │
│   ├── utils/
│   │   └── visualize.py
│   │
│   ├── explain/                # NEW
│   │   └── formatter.py
│
├── data/
│   ├── catalog_images/
│   ├── query_images/
│   └── metadata/catalog.csv
│
├── outputs/
│   ├── embeddings/
│   ├── predictions/
│   └── figures/
│
├── pyproject.toml
└── README.md
```

---

## 📊 Dataset

### Current Setup (Simple)

- Local custom dataset
- Small catalog of clothing items
- Metadata stored in CSV

### Required Fields

- `item_id`
- `image_path`
- `category`

### Optional Fields

- `color`
- `style`

---

### ⚠️ Important Note on Images

For best results:

- **Catalog images** → preferably clean product images (minimal background)
- **Query image** → can be a full outfit

**Why?**  
CLIP captures everything in the image, including background, which can affect similarity.

---

## 🚀 How to Run

### 1. Install Environment

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

### 2. Build Catalog Embeddings

Open:

```
notebooks/02_build_catalog_embeddings.ipynb
```

This will:

- Encode all catalog images  
- Save embeddings  

---

### 3. Run Recommendation

Open:

```
notebooks/03_recommend_top5.ipynb
```

Example:

```python
results = recommend_from_image(
    query_image_path="data/query_images/test_outfit.jpg",
    top_k=5,
    target_category="shoes"
)
```

---

### 4. Build Explanation Payload

```python
from soc.explain.formatter import build_explanation_payload

payload = build_explanation_payload(
    query_image_path=query_image,
    target_category="shoes",
    recommendations_df=results,
)
```

---

## 🧩 What This Model Actually Does

### ✔️ It DOES

- Find visually similar or style-aligned items  
- Provide fast local recommendations  
- Support explainability pipeline  

### ❌ It DOES NOT (Yet)

- Learn true outfit compatibility  
- Understand fashion rules explicitly  
- Reason about combinations (top + pants + shoes)  

---

## 🧠 Compatibility vs Similarity

**Important distinction:**

- **Similarity** → items look alike  
- **Compatibility** → items go well together  

Current system = **similarity-based retrieval**  
Future system = **compatibility learning**

---

## 🔮 Next Steps

### 🔹 Short-term

- Add LLM-based explanation generation  
- Improve prompt grounding  
- Add visualization of results  

---

### 🔹 Mid-term

- Add rule-based signals:
  - Color harmony  
  - Style matching  
  - Category constraints  

---

### 🔹 Long-term (Research Direction)

Move toward real compatibility models:

- Type-Aware Embeddings  
- Outfit-level models (Bi-LSTM)  
- Transformer-based models (OutfitTransformer)  
- Complementary item retrieval models  

---

## 📦 Future Dataset Upgrade

We plan to integrate:

- Polyvore dataset  
- Real outfit compatibility benchmarks  
- Training pipelines for learned models  

---

## 🧑‍💻 Tech Stack

- Python  
- PyTorch  
- OpenCLIP  
- Pandas / NumPy  
- Scikit-learn  
- Jupyter (VS Code)  
- uv (package manager)  

---

## 🧪 Current Status

- ✅ Working baseline  
- ✅ Retrieval pipeline  
- ✅ Embedding system  
- ✅ Category filtering  
- ✅ Explanation payload (LLM-ready)  

- 🚧 LLM explanation (next step)  
- 🚧 Improved compatibility modeling  

---

## 💡 Philosophy of This Project

**Build simple → correct → extensible**

Instead of jumping to complex models:

- Start with a working baseline  
- Understand each component  
- Improve step by step  

---

## 👤 Author

**Ali Jamali**  
MPhil Artificial Intelligence, University of Salford  

---

## ⭐ If you find this useful

Give it a star and follow the project as it evolves into a full explainable fashion recommendation system.