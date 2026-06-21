import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv() #loading .env
# print(os.getenv("GOOGLE_API_KEY"))

current_diectory = os.path.dirname(os.path.abspath(__file__))
persist_db_dir = os.path.join(current_diectory, "chroma_db")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(persist_directory=persist_db_dir, embedding_function=embedding_model)

prompt = input('Hey there! how may I help you?: ')


def ask_question(query):
    print("Question received:", query)

    print("Starting retrieval...")
    matching_docs = vector_store.similarity_search(query, k=5) #Retriving the chunks
    print("Retrieval complete")

    #Convert chunks into one context string
    context = "\n\n".join(
    [
        f"Metadata: {doc.metadata}\nContent: {doc.page_content}"
        for doc in matching_docs
    ])
    # print(context)

    prompt = f""" 
    Context: {context}

    Question:{query}

    Answer:
    """

    response = llm.invoke(prompt)
    return response.content

answer = ask_question(prompt)
print(answer)
