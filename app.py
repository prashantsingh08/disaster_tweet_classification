
import re
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model_store" / "best_disaster_tweet_model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_features(user_text: str) -> pd.DataFrame:
    cleaned = clean_text(user_text)
    features = pd.DataFrame([{
        "clean_text": cleaned,
        "text_length_chars": len(str(user_text)),
        "word_count": len(cleaned.split()),
        "has_hashtag": int("#" in str(user_text)),
        "has_mention": int("@" in str(user_text)),
        "has_url": int(("http" in str(user_text).lower()) or ("www" in str(user_text).lower()))
    }])
    return features

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    tweet_text = ""

    if request.method == "POST":
        tweet_text = request.form.get("tweet_text", "").strip()
        if tweet_text:
            features = build_features(tweet_text)
            pred = int(model.predict(features)[0])
            prediction = "Disaster Tweet" if pred == 1 else "Non-Disaster Tweet"

            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba(features)[0].max())
            elif hasattr(model, "decision_function"):
                score = float(model.decision_function(features)[0])
                confidence = 1 / (1 + pow(2.71828, -abs(score)))
            else:
                confidence = None

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        tweet_text=tweet_text
    )

if __name__ == "__main__":
    app.run(debug=True)

