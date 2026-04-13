import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# ---------------- CONFIG ----------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

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

# ---------------- GEMINI ANALYSIS ----------------

def analyze_with_gemini(text: str) -> dict:
    prompt = f"""
You are a professional fact-checking AI. Analyze the news claim or article below and determine whether it is Real or Fake.

Return ONLY valid JSON. No markdown. No extra text. No code fences.

Format:
{{
  "label": "Real" or "Fake" or "Uncertain",
  "confidence": <integer between 0 and 100>,
  "explanation": "<clear, concise explanation in 2-3 sentences>"
}}

News to analyze:
{text}
"""
    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print("Gemini error:", e)
        return {
            "label": "Uncertain",
            "confidence": 0,
            "explanation": "Unable to analyze at this time. Please try again."
        }

# ---------------- ENDPOINT ----------------

@app.post("/predict")
def predict_news(request: NewsRequest):
    result = analyze_with_gemini(request.text)
    return {
        "label": result.get("label", "Uncertain"),
        "confidence": result.get("confidence", 0),
        "explanation": result.get("explanation", "No explanation available.")
    }