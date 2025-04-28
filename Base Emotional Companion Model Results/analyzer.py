# # === File: analyzer.py ===
# import os
# import json
# import numpy as np
# import faiss
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# from typing import List, Dict

# # Load environment variables
# load_dotenv()

# # Initialize models
# emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Load FAISS index and metadata
# faiss_index = faiss.read_index("faiss_index/vector.index")
# with open("faiss_index/metadata.json", "r", encoding="utf-8") as f:
#     metadata = json.load(f)

# # Initialize psychoanalytic profile
# psycho_profile = {
#     "psychoanalysis": {},
#     "about_user": {}
# }

# # === Emotion Analysis ===
# def analyze_emotions(user_input: str) -> Dict[str, float]:
#     transformer_output = emotion_classifier(user_input)
#     emotions = {item['label']: round(item['score'], 4) for item in transformer_output[0] if item['score'] > 0.01}

#     with open("emotions.txt", "w", encoding="utf-8") as f:
#         json.dump(emotions, f, indent=2)

#     return emotions

# # === RAG Retrieval ===
# def retrieve_context(query: str, k: int = 5) -> List[str]:
#     query_vec = embedding_model.encode([query])[0].astype("float32")
#     if query_vec.shape[0] != faiss_index.d:
#         raise ValueError(f"Query vector dimension ({query_vec.shape[0]}) does not match FAISS index dimension ({faiss_index.d})")

#     _, indices = faiss_index.search(np.array([query_vec]), k)
#     retrieved = [metadata[i].get("text", "") for i in indices[0] if i < len(metadata)]

#     with open("rag_output.txt", "w", encoding="utf-8") as f:
#         for i, chunk in enumerate(retrieved):
#             f.write(f"Chunk {i + 1}:\n{chunk}\n\n")

#     return retrieved

# # === Update Psychoanalytic Profile ===
# def update_psycho_profile(user_input: str, context_chunks: List[str]) -> None:
#     prompt = f"""
# You are a psychoanalyst AI. Based on the user input and context, provide:
# 1. Good/Bad thinking patterns
# 2. Axioms or core beliefs
# 3. Cognitive distortions
# 4. Defense mechanisms
# 5. Maladaptive patterns
# 6. Inferred beliefs / self-schema
# 7. Emotional regulation patterns

# Input:
# {user_input}

# Context:
# {''.join(context_chunks[:3])}
# """

#     # Simulate psychoanalytic inference
#     psychoanalysis_output = {
#         "thinking_patterns": {"User demonstrates proactive optimism.": {"short_term": 0.7, "long_term": 0.6}},
#         "axioms": {"Believes effort leads to success.": {"short_term": 0.8, "long_term": 0.7}},
#         "cognitive_distortions": {"Possible minimization of potential risks.": {"short_term": 0.3, "long_term": 0.2}},
#         "defense_mechanisms": {"Might intellectualize emotions when excited.": {"short_term": 0.4, "long_term": 0.3}},
#         "maladaptive_patterns": {"Mild dependency on external validation for emotional uplift.": {"short_term": 0.35, "long_term": 0.25}},
#         "inferred_beliefs": {"Views emotional connection as part of personal success.": {"short_term": 0.5, "long_term": 0.4}},
#         "emotional_regulation": {"Utilizes positive feedback loops to maintain mood.": {"short_term": 0.6, "long_term": 0.5}}
#     }

#     psycho_profile["psychoanalysis"] = psychoanalysis_output

#     with open("psychoanalysis.txt", "w", encoding="utf-8") as f:
#         json.dump(psycho_profile["psychoanalysis"], f, indent=2)

#         # Simulate about_user inference (you can replace with a real model later)
#     about_user_output = {
#         "certain": {
#             "The user values joy and friendship.": True,
#             "The user feels confident in their abilities.": True
#         },
#         "unsure": {
#             "The user may have recently formed new social connections.": {
#                 "short_term": 0.7,
#                 "long_term": 0.5
#             },
#             "The user may derive motivation from social energy.": {
#                 "short_term": 0.6,
#                 "long_term": 0.4
#             }
#         }
#     }

#     psycho_profile["about_user"] = about_user_output

#     with open("about_user.txt", "w", encoding="utf-8") as f:
#         json.dump(psycho_profile["about_user"], f, indent=2)


# # === Expose profile for external use ===
# def get_psycho_profile() -> Dict[str, Dict]:
#     return psycho_profile

# === File: analyzer.py ===
import json
from transformers import pipeline
from typing import Dict, Any
from dotenv import load_dotenv
import os
import openai

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

def clamp_probability(value, min_val=0.01, max_val=0.95):
    return max(min_val, min(value, max_val))

def update_about_user_memory(text: str, emotions: Dict[str, float]) -> Dict[str, Any]:
    prompt = f"""
You are a clinical AI therapist. Based on the user's input and emotions, what can be inferred about the user?
Group into:
- Certain
- Unsure with soft probability estimates

Use format:
Certain:
- fact

Unsure:
- inference: short=0.5, long=0.3

Input:
"{text}"

Emotions:
{json.dumps(emotions)}
"""
    about_user = {"certain": {}, "unsure": {}}
    try:
        result = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful clinical psychoanalyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=400
        )
        lines = result.choices[0].message.content.strip().split("\n")
        current = None
        for line in lines:
            if line.lower().startswith("certain"):
                current = "certain"
            elif line.lower().startswith("unsure"):
                current = "unsure"
            elif line.startswith("- ") and current == "certain":
                about_user["certain"][line[2:].strip()] = True
            elif ": short=" in line and current == "unsure":
                try:
                    trait, rest = line[2:].split(": short=")
                    short_val, long_val = rest.split(", long=")
                    about_user["unsure"][trait.strip()] = {
                        "short_term": clamp_probability(float(short_val)),
                        "long_term": clamp_probability(float(long_val))
                    }
                except:
                    continue
    except Exception as e:
        about_user["error"] = str(e)
    return about_user

def update_psychoanalysis_memory(text: str, emotions: Dict[str, float]) -> Dict[str, Any]:
    prompt = f"""
You are a CBT-informed AI therapist. Analyze the user input and generate insights for each of these categories:

- Good or Bad Thinking Patterns
- Cognitive Distortions
- Defense Mechanisms
- Maladaptive Patterns
- Inferred Beliefs / Self-Schema
- Emotional Regulation Patterns
- Axioms / Core Beliefs

If confidence is low, still include insights but assign lower probability estimates (e.g., short=0.1, long=0.05). Provide up to 2 insights per category if applicable.

Format:
[Category]:
- Description: short=0.4, long=0.2
- Another possible insight: short=0.1, long=0.05

Input:
\"{text}\"
Emotions:
{json.dumps(emotions)}
"""
    expected_categories = [
        "Good or Bad Thinking Patterns",
        "Cognitive Distortions",
        "Defense Mechanisms",
        "Maladaptive Patterns",
        "Inferred Beliefs / Self-Schema",
        "Emotional Regulation Patterns",
        "Axioms / Core Beliefs"
    ]

    structured_analysis = {cat: [] for cat in expected_categories}

    try:
        result = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a structured, conservative CBT therapist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=700
        )
        current_category = None
        lines = result.choices[0].message.content.strip().split("\n")
        for line in lines:
            if line.strip() == "":
                continue
            if line.endswith(":"):
                current_category = line[:-1].strip()
                if current_category not in structured_analysis:
                    structured_analysis[current_category] = []
            elif ": short=" in line and current_category:
                try:
                    desc, rest = line.strip("- ").split(": short=")
                    short_val, long_val = rest.split(", long=")
                    structured_analysis[current_category].append({
                        "description": desc.strip(),
                        "short_term": clamp_probability(float(short_val)),
                        "long_term": clamp_probability(float(long_val))
                    })
                except:
                    continue
    except Exception as e:
        structured_analysis["error"] = str(e)

    for category in expected_categories:
        if not structured_analysis.get(category):
            structured_analysis[category] = [{
                "description": "No clear insight identified in this category.",
                "short_term": 0.05,
                "long_term": 0.01
            }]

    return structured_analysis

# def analyze_user_input(user_text: str) -> Dict[str, Any]:
#     transformer_output = classifier(user_text)[0]
#     emotions = {item['label']: round(item['score'], 4) for item in transformer_output}
#     about_user = update_about_user_memory(user_text, emotions)
#     psycho = update_psychoanalysis_memory(user_text, emotions)
#     return {
#         "psychoanalysis": psycho,
#         "emotions": emotions,
#         "about_user": about_user
#     }

def analyze_user_input(user_text: str) -> Dict[str, Any]:
    transformer_output = classifier(user_text)[0]
    emotions = {item['label']: round(item['score'], 4) for item in transformer_output}
    about_user = update_about_user_memory(user_text, emotions)
    psycho = update_psychoanalysis_memory(user_text, emotions)
    return {
        "psychoanalysis": psycho,
        "emotions": emotions,
        "about_user": about_user
    }
