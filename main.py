from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os, logging
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="SOUL7OS Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    risk_score: float = 0.0

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        risky_keywords = ["legal advice", "medical diagnosis", "financial advice"]
        risk_score = sum(0.3 for k in risky_keywords if k in request.message.lower())
        risk_score = min(risk_score, 1.0)

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Be compliant and safe."},
                {"role": "user", "content": request.message},
            ],
            max_tokens=500,
            temperature=0.2,
        )
        response_text = completion.choices[0].message.content or ""
        if risk_score > 0.6:
            response_text = (
                "⚠️ Disclaimer: I am an AI and cannot provide professional advice.\n\n" + response_text
            )

        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            timestamp=datetime.utcnow().isoformat(),
            risk_score=risk_score,
        )
    except Exception as e:
        logging.exception("Chat error")
        raise HTTPException(status_code=500, detail="Internal server error")
