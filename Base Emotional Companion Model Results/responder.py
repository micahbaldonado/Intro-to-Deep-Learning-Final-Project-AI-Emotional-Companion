# === File: responder.py ===
import openai
from dotenv import load_dotenv
import os
import json
from typing import Dict, Any

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# def generate_response(
#     user_text: str,
#     psychoanalysis: Dict[str, Any],
#     emotions: Dict[str, float],
#     about_user: Dict[str, Any],
#     short_term_memory: str,
#     long_term_memory: str,
#     global_memory: Dict[str, Any],
#     model: str = "gpt-4"
# ) -> str:
def generate_response(
    user_text: str,
    psychoanalysis: Dict[str, Any],
    emotions: Dict[str, float],
    about_user: Dict[str, Any],
    model: str = "gpt-4"
) -> str:
    """
    Generate a therapeutic and introspective response using GPT-4.
    """
    insight_lines = []
    for category, entries in psychoanalysis.items():
        if isinstance(entries, list):
            for insight in entries:
                line = f"[{category}] {insight['description']} (short={insight['short_term']}, long={insight['long_term']})"
                insight_lines.append(line)

    prompt = f"""
    You are a warm, emotionally intelligent AI therapist and companion.
    You do not lecture but instead help the user gently reflect, identify patterns, and move toward insight.
    Always respect uncertainty, and invite exploration. You may gently ask questions when high-confidence issues arise.

    User Message:
    "{user_text}"

    Emotions:
    {json.dumps(emotions, indent=2)}

    Psychoanalytic Observations:
    {chr(10).join(insight_lines) or 'No observations available.'}

    About the User:
    {json.dumps(about_user, indent=2)}

    """
    # Respond with warm, grounded tone. Reflect the user's emotional content, and highlight soft insights only if relevant.


    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a non-judgmental and thoughtful AI therapist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content
