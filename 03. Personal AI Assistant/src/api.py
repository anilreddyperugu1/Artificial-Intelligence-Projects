from fastapi import FastAPI
from src.chatbot import ask_question
from fastapi.middleware.cors import CORSMiddleware

#Creating the app
app = FastAPI()

@app.get('/')
def home():
    return ('message": "portfolio AI Assistant is running')


@app.get('/chat')
def chat(question: str):
    answer = ask_question(question)
    return {"answer": answer}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://anilreddyperugu1.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)