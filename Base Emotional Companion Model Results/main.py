# === File: main.py ===
import os
import json
import nltk
from analyzer import analyze_user_input
from responder import generate_response

nltk.download("vader_lexicon", quiet=True)

PSYCHO_PATH = "psychoanalysis.txt"
ABOUT_PATH = "about-user.txt"
EMOTIONS_PATH = "emotions.txt"

# Initialize logs
with open(PSYCHO_PATH, "w") as f:
    f.write("=== Session Psychoanalysis Log ===\n\n")

with open(ABOUT_PATH, "w") as f:
    f.write("=== About User Memory Log ===\n\n")

with open(EMOTIONS_PATH, "w") as f:
    f.write("=== Emotions Log ===\n\n")

print("Welcome to your emotional companion. Type 'exit' to end the session.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("\nSession ended. Insights saved to psychoanalysis.txt, about-user.txt, and emotions.txt")
        break

    analysis = analyze_user_input(user_input)

    # Save everything at once
    with open(PSYCHO_PATH, "a") as pf, open(ABOUT_PATH, "a") as af, open(EMOTIONS_PATH, "a") as ef:
        psycho_data = analysis.get("psychoanalysis", {})
        if psycho_data:
            for category, insights in psycho_data.items():
                if isinstance(insights, list) and insights:
                    pf.write(f"[{category}]\n")
                    for insight in insights:
                        desc = insight.get("description", "")
                        short = insight.get("short_term", 0)
                        long = insight.get("long_term", 0)
                        pf.write(f"- {desc} (short={short}, long={long})\n")
                    pf.write("\n")

        af.write(json.dumps(analysis["about_user"], indent=2) + "\n\n")
        ef.write(json.dumps(analysis["emotions"], indent=2) + "\n\n")

    response = generate_response(
        user_text=user_input,
        psychoanalysis=analysis["psychoanalysis"],
        emotions=analysis["emotions"],
        about_user=analysis["about_user"]
    )

    print("\nCompanion:", response, "\n")
