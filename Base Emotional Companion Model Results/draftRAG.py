# === File: rag_therapist_module.py (Hybrid RAG + Conversational AI) ===
import os
import json
import re
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from collections import defaultdict
from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Constants ===
embedding_model = "text-embedding-3-small"
llm_model = "gpt-4o"
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
SHOW_RETRIEVED_CHUNKS = True

# === Setup ===
load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Load Index and Metadata ===
faiss_index = faiss.read_index("faiss_index/vector.index")
with open("faiss_index/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === Load Corpus ===
chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
corpus_by_id = defaultdict(str)
bm25_docs = []
for item in metadata:
    file_path = f"data/{item['source_file']}"
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["id"] == item["chunk_id"]:
                chunks = chunk_splitter.split_text(data["text"])
                for sub_chunk in chunks:
                    bm25_docs.append(sub_chunk)
                break
bm25 = BM25Okapi([doc.split() for doc in bm25_docs])

# === Memory ===
conversation_memory: List[Tuple[str, str]] = []
psycho_profile = {"working_hypotheses": [], "defenses_observed": [], "topics": [], "client_needs": []}

# === Embed ===
def embed_query(text: str) -> List[float]:
    return openai.embeddings.create(input=text, model=embedding_model).data[0].embedding

# === Reranking ===
def semantic_rerank(query: str, candidates: List[str]) -> List[str]:
    scores = reranker_model.predict([(query, c) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked]

# === Retrieve ===
def retrieve_context(query: str, k=5) -> List[str]:
    dense_vec = np.array(embed_query(query)).astype("float32")
    _, dense_ids = faiss_index.search(np.array([dense_vec]), k)
    dense_chunks = [metadata[i] for i in dense_ids[0] if i < len(metadata)]

    sparse_scores = bm25.get_scores(query.split())
    top_sparse = sorted([(i, s) for i, s in enumerate(sparse_scores)], key=lambda x: x[1], reverse=True)[:k * 2]
    sparse_chunks = [bm25_docs[i] for i, _ in top_sparse]

    combined = list({text: None for text in sparse_chunks}.keys())
    reranked = semantic_rerank(query, combined)

    if SHOW_RETRIEVED_CHUNKS:
        print("\n--- Retrieved Chunks ---")
        for i, chunk in enumerate(reranked[:k]):
            print(f"Chunk {i+1}:\n{chunk[:300]}\n")

    return reranked[:k]

# === Response Filtering ===
def apply_safety_filter(text: str) -> str:
    risky = [r"you should definitely", r"this will cure", r"you must", r"clearly you are", r"you are diagnosed with"]
    for r in risky:
        if re.search(r, text, re.IGNORECASE):
            return "(Filtered response due to overconfident language)"
    return text

# === Psychoanalysis ===
def update_profile():
    transcript = "\n".join([f"User: {u}\nAI: {a}" for u, a in conversation_memory])
    prompt = f"""
You are a psychoanalyst. Extract:
- Working Hypotheses
- Defenses Observed
- Key Topics
- Client's likely therapeutic needs

Transcript:
{transcript}
Return JSON with keys: working_hypotheses, defenses_observed, topics, client_needs.
"""
    response = openai.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "Analyze therapy session."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        result = json.loads(response.choices[0].message.content.strip())
        for k in psycho_profile:
            for i in result.get(k, []):
                if i not in psycho_profile[k]:
                    psycho_profile[k].append(i)
    except:
        pass

# === Conversation Logic ===
def generate_prompt(user_msg: str, context: List[str]) -> str:
    context_text = "\n\n".join(context)
    memory_text = "\n".join([f"User: {u}\nAI: {a}" for u, a in conversation_memory[-5:]])
    profile = json.dumps(psycho_profile, indent=2)
    return f"""
You are a compassionate AI therapist. Respond helpfully but briefly if the user's input is short.

Psycho Profile:
{profile}

Recent Memory:
{memory_text}

Context:
{context_text}

User:
{user_msg}
"""

def run_conversation(user_msg: str):
    context = retrieve_context(user_msg)
    update_profile()
    prompt = generate_prompt(user_msg, context)
    short = len(user_msg.strip().split()) <= 4
    model = llm_model

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an empathetic therapist."},
            {"role": "user", "content": prompt}
        ]
    )
    reply = response.choices[0].message.content.strip()
    reply = apply_safety_filter(reply)

    if short:
        reply = reply.split(". ")[0] + "."  # Clip to first sentence if user input is short

    conversation_memory.append((user_msg, reply))
    return reply

# === CLI Entry ===
if __name__ == "__main__":
    print("🤖 AI Companion: Hello, I'm here to listen. What's on your mind?")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("👋 Goodbye. Take care.")
                break
            response = run_conversation(user_input)
            print(f"AI Companion: {response}\n")
        except KeyboardInterrupt:
            print("\n👋 Session ended.")
            break