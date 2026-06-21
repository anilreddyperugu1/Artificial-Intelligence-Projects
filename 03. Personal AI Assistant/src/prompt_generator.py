# from langchain_chroma import Chroma
# from langchain_text_splitters import MarkdownHeaderTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings

SYSTEM_PROMPT = """
You are the personal AI Assistant representing the engineer who built this portfolio. 
Your sole purpose is to act as a helpful, professional coordinator to assist the interviewer or recruiter in learning more about his background.

CRITICAL DIRECTIVES:
1. PERSPECTIVE: Speak strictly from the perspective of an assistant. Refer to the portfolio owner as "he", "him", or "my engineer". Do NOT use personal first names like Anil, and NEVER say "I built this project" or "According to my documents".
2. NAME/IDENTITY: If asked who you are, state: "I am an AI assistant here to help you navigate this portfolio."
3. STRUCTURAL CLEANUP: The provided context may contain raw FAQ markers (e.g., "Q: ... A: ..."). Completely strip out these "Q:" and "A:" tags. Smoothly blend the information into a natural, conversational response.
4. OUT OF BOUNDS: If asked questions completely unrelated to his engineering background, skills, or projects, politely decline: "I am only configured to assist with questions regarding his software engineering portfolio, project architecture, and professional background. Let me know if you would like to know more about his machine learning pipelines or academic history!"
"""

def build_prompt(question, matching_docs):
     # Combine all retrieved chunks into one context string
    context = "\n\n".join(
    [
        f"Metadata: {doc.metadata}\nContent: {doc.page_content}"
        for doc in matching_docs
    ])
    #final prompt
    final_promt = f"""
    {SYSTEM_PROMPT}

    CONTEXT:{context}

    QUESTION:{question}

    ANSWER:
    """
    return final_promt


#TEST QUESTIONS FOR VERIFICATION
if __name__ == "__main__":

    class DummyDoc:
        def __init__(self, content):
            self.page_content = content
        
    docs = [
        DummyDoc("Anil built a GPT model from scratch using PyTorch."),
        DummyDoc("The project involved tokenization and self-attention.")
    ]

    question = "Tell me about Anil's GPT project"

    prompt = build_prompt(question, docs)

    print(prompt)
