import os
import json
from analyzer import analyze_user_input
from responder import generate_response

# File paths
INPUT_FILE = "test_user_inputs.txt"
OUTPUT_FILE = "emo_comp_output.jsonl"
PSYCHO_DIR = "psychoanalysis"
EMO_DIR = "emotions"
ABOUT_DIR = "about_user"

# Ensure output directories exist
os.makedirs(PSYCHO_DIR, exist_ok=True)
os.makedirs(EMO_DIR, exist_ok=True)
os.makedirs(ABOUT_DIR, exist_ok=True)

# Load user inputs as paragraph blocks (split by double newlines)
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read()
    user_inputs = [block.strip() for block in content.split("\n\n") if block.strip()]

# Process and save outputs
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for idx, user_text in enumerate(user_inputs, start=1):
        print(f"Processing input #{idx}...")
        try:
            analysis = analyze_user_input(user_text)
            response = generate_response(
                user_text=user_text,
                psychoanalysis=analysis["psychoanalysis"],
                emotions=analysis["emotions"],
                about_user=analysis["about_user"]
            )

            # Write response to JSONL
            json.dump({"input": user_text, "response": response}, out_f)
            out_f.write("\n")
            out_f.flush()

            # Save individual analysis files
            with open(os.path.join(PSYCHO_DIR, f"psychoanalysis_{idx}.txt"), "w", encoding="utf-8") as pf:
                for category, insights in analysis["psychoanalysis"].items():
                    pf.write(f"[{category}]\n")
                    for insight in insights:
                        desc = insight.get("description", "")
                        short = insight.get("short_term", 0)
                        long = insight.get("long_term", 0)
                        pf.write(f"- {desc} (short={short}, long={long})\n")
                    pf.write("\n")

            with open(os.path.join(EMO_DIR, f"emotions_{idx}.txt"), "w", encoding="utf-8") as ef:
                json.dump(analysis["emotions"], ef, indent=2)

            with open(os.path.join(ABOUT_DIR, f"about_user_{idx}.txt"), "w", encoding="utf-8") as af:
                json.dump(analysis["about_user"], af, indent=2)

        except Exception as e:
            print(f"❌ Error processing input #{idx}: {e}")

print(f"\n✅ All inputs processed. Output saved to {OUTPUT_FILE}")
print(f"📂 Detailed logs saved to: {PSYCHO_DIR}/, {EMO_DIR}/, and {ABOUT_DIR}/")
