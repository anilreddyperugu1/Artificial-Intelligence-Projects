from fastapi import FastAPI
from chatbot_ollama import ask_question

#Creating the app
app = FastAPI()

@app.get('/')
def home():
    return ('message": "portfolio AI Assistant is running')


@app.get('/chat')
def chat(question: str):
    answer = ask_question(question)
    return {"answer": answer}