import json
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from tqdm import tqdm

# === MODIFY THESE ===
VANILLA_JSONL_PATH = "vanilla_gpt_output.jsonl"
EMO_COMP_JSONL_PATH = "emo_comp_output.jsonl"
VANILLA_TXT_OUTPUT = "vanilla_eval_results.txt"
EMO_TXT_OUTPUT = "emo_comp_eval_results.txt"
MAX_ENTRIES = 44

def load_jsonl(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
            if len(entries) >= MAX_ENTRIES:
                break
    return entries

def evaluate_entries(entries, output_path, classifier_model, classifier_tokenizer,
                     similarity_model, emotion_pipe, device):
    results = []

    for i, entry in enumerate(tqdm(entries, desc=f"Evaluating {output_path}")):
        user_input = entry.get("input") or entry.get("user_input")
        model_output = entry.get("response") or entry.get("model_output")

        if not user_input or not model_output:
            continue

        # BERT Confidence
        inputs = classifier_tokenizer(model_output, return_tensors="pt", truncation=True, padding=True)
        if device >= 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            logits = classifier_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        confidence_score = float(np.max(probs))

        # Semantic Similarity
        emb1 = similarity_model.encode(user_input, convert_to_tensor=True)
        emb2 = similarity_model.encode(model_output, convert_to_tensor=True)
        similarity_score = float(util.pytorch_cos_sim(emb1, emb2)[0][0])

        # Empathy Score
        emotions = emotion_pipe(model_output)[0]
        non_neutral = [e for e in emotions if e["label"].lower() != "neutral"]
        empathy_score = max([e["score"] for e in non_neutral]) if non_neutral else 0.0

        results.append({
            "index": i + 1,
            "user_input": user_input,
            "model_output": model_output,
            "bert_score": confidence_score,
            "semantic_similarity": similarity_score,
            "empathy_score": empathy_score
        })

    # Calculate averages
    avg_bert = np.mean([r["bert_score"] for r in results])
    avg_sim = np.mean([r["semantic_similarity"] for r in results])
    avg_emo = np.mean([r["empathy_score"] for r in results])

    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== SUMMARY STATISTICS ===\n")
        f.write(f"Average BERT Confidence     : {avg_bert:.3f}\n")
        f.write(f"Average Semantic Similarity : {avg_sim:.3f}\n")
        f.write(f"Average Empathy Score       : {avg_emo:.3f}\n")
        f.write("=" * 70 + "\n\n")

        for r in results:
            f.write(f"[{r['index']}]\n")
            f.write(f"User Input:\n{r['user_input']}\n\n")
            f.write(f"Model Output:\n{r['model_output']}\n\n")
            f.write(f"BERT Confidence       : {r['bert_score']:.3f}\n")
            f.write(f"Semantic Similarity   : {r['semantic_similarity']:.3f}\n")
            f.write(f"Empathy Score         : {r['empathy_score']:.3f}\n")
            f.write("=" * 70 + "\n\n")

    print(f"✅ Saved results to {output_path}")

def main():
    device = 0 if torch.cuda.is_available() else -1

    # Load models
    print("Loading models...")
    classifier_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    classifier_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    classifier_model.eval()
    if device >= 0:
        classifier_model.cuda()

    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=device)

    # Load and evaluate
    vanilla_entries = load_jsonl(VANILLA_JSONL_PATH)
    emo_entries = load_jsonl(EMO_COMP_JSONL_PATH)

    evaluate_entries(vanilla_entries, VANILLA_TXT_OUTPUT,
                     classifier_model, classifier_tokenizer,
                     similarity_model, emotion_pipe, device)

    evaluate_entries(emo_entries, EMO_TXT_OUTPUT,
                     classifier_model, classifier_tokenizer,
                     similarity_model, emotion_pipe, device)

if __name__ == "__main__":
    main()
