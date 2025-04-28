# === File: vanilla_gpt_runner.py ===
import openai
import os
import json
import re
from dotenv import load_dotenv
from typing import List

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(user_text: str, model: str = "gpt-4") -> str:
    """
    Generates a simple, introspective ChatGPT response to raw user input.
    No additional context (emotions, psychoanalysis, etc.) is used.
    """
    prompt = f"""
You are a warm, emotionally intelligent AI therapist and companion.
You do not lecture but instead help the user gently reflect, identify patterns, and move toward insight.
Always respect uncertainty, and invite exploration. You may gently ask questions when high-confidence issues arise.

User Message:
\"\"\"{user_text}\"\"\"
"""

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a non-judgmental and thoughtful AI therapist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

def load_inputs_from_txt(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by [1], [2], etc.
    entries = re.split(r"\[\d+\]", text)
    return [entry.strip() for entry in entries if entry.strip()]

def main():
    input_path = "test_user_inputs.txt"
    output_path = "vanilla_gpt_output.jsonl"

    user_inputs = load_inputs_from_txt(input_path)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for idx, user_text in enumerate(user_inputs, start=1):
            print(f"Processing input #{idx}...")
            try:
                response = generate_response(user_text)
                result = {"input": user_text, "response": response}
                outfile.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"Error on input #{idx}: {e}")

    print(f"\n✅ All inputs processed. Output saved to: {output_path}")

if __name__ == "__main__":
    main()
