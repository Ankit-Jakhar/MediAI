from fastapi import FastAPI
from pydantic import BaseModel
from core_chatbot import ask_bot

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    if any(k in query.question.lower() for k in ["symptom", "pain", "treatment", "doctor", "fever", "disease", "medicine"]):
        response = ask_bot(query.question)
    else:
        response = "I only answer medical-related questions."
    return {"response": response}