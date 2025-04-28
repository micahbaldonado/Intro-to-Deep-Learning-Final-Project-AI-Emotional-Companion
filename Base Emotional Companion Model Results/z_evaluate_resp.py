import json
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from tqdm import tqdm

# === MODIFY THESE ===
INPUT_JSONL_PATH = "sample_eval.jsonl"
OUTPUT_TXT_PATH = "evaluation_results.txt"


def evaluate(jsonl_path, output_txt_path):
    # Load models
    device = 0 if torch.cuda.is_available() else -1

    print("Loading models...")
    classifier_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    classifier_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    classifier_model.eval()
    if device >= 0:
        classifier_model.cuda()

    emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=device)
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load data
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []
    for i, entry in enumerate(tqdm(data, desc="Evaluating")):
        user_input = entry["user_input"]
        model_output = entry["model_output"]

        # --- BERT Confidence Score ---
        inputs = classifier_tokenizer(model_output, return_tensors="pt", truncation=True, padding=True)
        if device >= 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = classifier_model(**inputs).logits
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        avg_bert_score = float(np.max(probs))

        # --- Semantic Similarity ---
        emb1 = similarity_model.encode(user_input, convert_to_tensor=True)
        emb2 = similarity_model.encode(model_output, convert_to_tensor=True)
        similarity_score = float(util.pytorch_cos_sim(emb1, emb2)[0][0])

        # --- Empathy Score ---
        emotions = emotion_pipe(model_output)[0]
        non_neutral = [e for e in emotions if e["label"].lower() != "neutral"]
        empathy_score = max([e["score"] for e in non_neutral]) if non_neutral else 0.0

        results.append({
            "user_input": user_input,
            "model_output": model_output,
            "bert_score": avg_bert_score,
            "semantic_similarity": similarity_score,
            "empathy_score": empathy_score
        })

    # Compute Averages
    avg_bert = np.mean([r["bert_score"] for r in results])
    avg_sim = np.mean([r["semantic_similarity"] for r in results])
    avg_emo = np.mean([r["empathy_score"] for r in results])

    # Write to file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Average BERT Score         : {avg_bert:.3f}\n")
        f.write(f"Average Semantic Similarity: {avg_sim:.3f}\n")
        f.write(f"Average Empathy Score      : {avg_emo:.3f}\n")
        f.write("=" * 60 + "\n\n")

        for idx, r in enumerate(results, 1):
            f.write(f"[{idx}]\n")
            f.write(f"User Input   : {r['user_input']}\n")
            f.write(f"Model Output : {r['model_output']}\n")
            f.write(f"BERT Score          : {r['bert_score']:.3f}\n")
            f.write(f"Semantic Similarity : {r['semantic_similarity']:.3f}\n")
            f.write(f"Empathy Score       : {r['empathy_score']:.3f}\n\n")

    print(f"\n✅ Evaluation complete. Results saved to: {output_txt_path}")


if __name__ == "__main__":
    evaluate(INPUT_JSONL_PATH, OUTPUT_TXT_PATH)
