from fastapi import FastAPI, Body
from pydantic import BaseModel
from transformers import pipeline
import requests
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

# ------- API key related --------
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_HEADERS = {
    "Authorization": f"Bearer {os.environ['SARVAM_API_KEY']}",  
    "Content-Type": "application/json"
}

# ------- Transformers Model --------
intent_classifier = pipeline(
    "text-classification",
    model="./intent_model",
    tokenizer="./intent_model",
    device=0
)

# ------- Knowledge/Data Base --------
with open("knowledge_base.json", "r") as f:
    kb = json.load(f)

# ------- FastAPI App --------
app = FastAPI(title="BoB Chatbot API")

# ------- Request Models --------
class QueryRequest(BaseModel):
    query: str

class ChatRequest(BaseModel):
    query: str
    messages: list = []  # optional memory of previous messages


# ------- Endpoints --------
@app.post("/find-intent")
def find_intent(req: QueryRequest):
    """Classify user query into an intent."""
    intent = intent_classifier(req.query)[0]['label']
    return {"query": req.query, "intent": intent}


@app.post("/chat-query")
def chat_query(req: ChatRequest):
    """Answer user query using Sarvam API and knowledge base."""
    # Step 1: Find intent
    intent = intent_classifier(req.query)[0]['label']

    # Step 2: Get database entries
    if intent not in kb:
        return {"reply": "Sorry, I don’t have information about that in my database."}

    database = "\n".join(kb[intent])

    # Step 3: Build system prompt
    sys_prompt = f"""
    You are an experienced customer service manager at a private banking company named "Bank of Baroda" a.k.a "BoB".
    Answer the user's query politely using the given database atmost 3 sentences.
    If the answer is not in database, politely reject to answer the query. You can carry bare minimum conversation.

    <database>
    {database}
    </database>
    
    # YOU MUST AND SHOULD ANSWER THE USER'S QUERY FROM THE GIVEN DATABASE ONLY!
    """

    # Step 4: Prepare messages
    messages = req.messages.copy()
    messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": req.query})

    # Step 5: API Call
    data = {
        "messages": messages,
        "model": "sarvam-m",
        "max_tokens": 300
    }

    response = requests.post(SARVAM_URL, headers=SARVAM_HEADERS, json=data)

    if response.status_code == 200:
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        clean_reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
        messages.append({"role": "assistant", "content": clean_reply})
        return {"reply": clean_reply, "messages": messages}
    else:
        return {"error": response.status_code, "details": response.json()}
