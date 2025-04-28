# === File: build_faiss_index.py ===
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === Config ===
CHUNK_FOLDER = "processed_chunks"
OUTPUT_INDEX = "faiss_index/vector.index"
OUTPUT_METADATA = "faiss_index/metadata.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

# === Load Embedder ===
embed_model = SentenceTransformer(EMBED_MODEL)

# === Create Output Folder ===
os.makedirs("faiss_index", exist_ok=True)

# === Prepare Storage ===
all_embeddings = []
metadata = []
dimension = embed_model.get_sentence_embedding_dimension()

# === Iterate over Chunks ===
print("📚 Indexing chunks...")
for filename in tqdm(os.listdir(CHUNK_FOLDER)):
    if not filename.endswith(".jsonl"):
        continue
    
    file_path = os.path.join(CHUNK_FOLDER, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"].strip()
            if not text:
                continue

            embedding = embed_model.encode(text).astype("float32")
            all_embeddings.append(embedding)

            metadata.append({
                "chunk_id": data["id"],
                "source_file": filename,
                "text_preview": text[:150]
            })

# === Build FAISS Index ===
index = faiss.IndexFlatL2(dimension)
index.add(np.array(all_embeddings))
faiss.write_index(index, OUTPUT_INDEX)

# === Save Metadata ===
with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved index to {OUTPUT_INDEX} with {len(all_embeddings)} vectors.")
