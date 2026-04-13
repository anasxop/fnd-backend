import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import google.generativeai as genai

# ---------------- CONFIG & SETUP ----------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

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
    gemini_data = analyze_with_gemini(request.text)

    return {
        "final_verdict": gemini_data.get("verdict", "Uncertain"),
        "confidence": gemini_data.get("confidence", 0),
        "explanation": gemini_data.get("explanation", "No explanation."),
        "verdict_type": "Gemini AI",
    }
