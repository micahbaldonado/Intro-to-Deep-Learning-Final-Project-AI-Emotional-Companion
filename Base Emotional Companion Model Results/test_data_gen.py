# === File: test_data_gen.py ===
from datasets import load_dataset
import random

# Load the mental_health subset of the reddit dataset
dataset = load_dataset("reddit", "mental_health", split="train")

# Filter out long posts from the 'content' field (where the full post text is stored)
long_posts = [
    entry["content"].strip()
    for entry in dataset
    if isinstance(entry.get("content"), str) and len(entry["content"].split()) > 25
]

# Shuffle and pick 50 samples
random.seed(42)
selected = random.sample(long_posts, 50)

# Save to a text file
with open("test_user_inputs.txt", "w", encoding="utf-8") as f:
    for i, post in enumerate(selected, 1):
        f.write(f"[{i}] {post}\n\n")

print("✅ Saved 50 user-like posts to test_user_inputs.txt")
