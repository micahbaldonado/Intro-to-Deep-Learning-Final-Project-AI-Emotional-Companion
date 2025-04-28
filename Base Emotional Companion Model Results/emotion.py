# === IMPORTS ===
import text2emotion as te                          # Text2Emotion
from nrclex import NRCLex                          # NRCLex
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # VADER
from transformers import pipeline                  # Transformer (GoEmotions-like model)
import nltk

# === NLTK SETUP ===
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# === TRANSFORMER SETUP ===
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# === ANALYSIS FUNCTIONS ===
def analyze_with_text2emotion(text):
    return te.get_emotion(text)

def analyze_with_nrclex(text):
    nrc = NRCLex(text)
    return nrc.raw_emotion_scores, nrc.top_emotions

def analyze_with_vader(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

def analyze_with_transformer(text):
    return classifier(text)

# === MAIN INTERFACE ===
if __name__ == "__main__":
    user_input = input("How are you feeling today?\n> ")

    print("\n🔵 Text2Emotion (5-emotion breakdown):")
    t2e_result = analyze_with_text2emotion(user_input)
    for emotion, score in t2e_result.items():
        print(f"{emotion:8}: {score:.2f}")

    print("\n🟢 NRCLex (Raw emotion scores):")
    nrc_raw, nrc_top = analyze_with_nrclex(user_input)
    for emotion, score in nrc_raw.items():
        print(f"{emotion:8}: {score}")

    print("\n🟢 NRCLex (Top Emotions):")
    for emotion, score in nrc_top:
        print(f"{emotion:8}: {score:.2f}")

    print("\n🟠 VADER Sentiment (Polarity Scores):")
    vader_result = analyze_with_vader(user_input)
    for key, value in vader_result.items():
        print(f"{key:8}: {value:.2f}")

    print("\n🧠 Transformer-Based Emotion Classification (GoEmotions-like):")
    transformer_output = analyze_with_transformer(user_input)
    for item in transformer_output[0]:
        print(f"{item['label']:12}: {item['score']:.4f}")
