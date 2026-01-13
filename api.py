import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import json
import joblib
import google.generativeai as genai
import nltk
from nltk.corpus import stopwords

# ---------------- CONFIG & SETUP ----------------

# 1. Setup Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCgzsQztpQoE7FKVKRF2VZvqXHjrRxCnjg") 
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

# 2. Setup Local ML Model
# ⚠️ CRITICAL: Ensure 'model.pkl' and 'vectorizer.pkl' are in this folder!
print("Loading local models...")
try:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("Local models loaded successfully.")
    LOCAL_MODEL_AVAILABLE = True
except Exception as e:
    print(f"Error loading local models: {e}")
    print("API will run in Gemini-only mode.")
    LOCAL_MODEL_AVAILABLE = False

# ---------------- APP ----------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsRequest(BaseModel):
    text: str

# ---------------- HELPER FUNCTIONS ----------------

def clean_text(text):
    """Standard cleaning for the local model"""
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

def analyze_with_gemini(text):
    """Queries Gemini for the primary verdict"""
    prompt = f"""
You are a professional fact-checking AI.

Return ONLY valid JSON. No markdown. No extra text.

Format:
{{
  "verdict": "Real" | "Fake" | "Uncertain",
  "confidence": number between 0 and 100,
  "explanation": "short clear explanation"
}}

Claim:
{text}
"""
    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print("Gemini error:", e)
        return {"verdict": "Uncertain", "confidence": 0, "explanation": "Gemini error."}

# ---------------- ENDPOINT ----------------

@app.post("/predict")
def predict_news(request: NewsRequest):
    # 1. Run Gemini (Primary)
    gemini_data = analyze_with_gemini(request.text)

    # 2. Run Local Model (Reference Only)
    ml_label = "N/A"
    ml_conf = 0

    if LOCAL_MODEL_AVAILABLE:
        try:
            cleaned = clean_text(request.text)
            vec_text = vectorizer.transform([cleaned])
            
            # Predict
            pred = model.predict(vec_text)[0]
            prob = model.predict_proba(vec_text)[0].max()
            
            ml_label = "Fake" if pred == 1 else "Real"
            ml_conf = round(prob * 100, 2)
        except Exception as e:
            print(f"Local prediction error: {e}")

    # 3. Combine & Return
    return {
        # Primary (Gemini)
        "final_verdict": gemini_data.get("verdict", "Uncertain"),
        "confidence": gemini_data.get("confidence", 0),
        "explanation": gemini_data.get("explanation", "No explanation."),
        "verdict_type": "Gemini AI",
        
        # Secondary (Local Reference)
        "ml_result": ml_label,
        "ml_confidence": ml_conf
    }