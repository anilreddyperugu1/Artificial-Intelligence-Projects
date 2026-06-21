import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, "..", "Knowledge_base", "Knowledge_base.md")
persist_db_dir = os.path.join(current_directory, "chroma_db")

with open(file_path, "r", encoding="utf-8") as f: #Reading the file
    markdown_content = f.read()


headers_to_split = [ #Markdown header hierarchies to split on
    ('#', "Header_1"),
    ('##', "Header_2"),
    ('###', "Header_3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split,strip_headers=False) #Intializing the splitter

splitted_chunks = splitter.split_text(markdown_content)

# print(f'splitted data: {len(splitted_chunks)}')
# print(splitted_chunks[0].page_content)
# print(splitted_chunks[0].metadata)
# print(max(len(chunk.page_content) for chunk in splitted_chunks))

print("Initializing local HuggingFace embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2") #Loading the model

# Generating embeddings and persist them into the Chroma vector store
vector_store = Chroma.from_documents(
    documents=splitted_chunks,
    embedding=embedding_model,
    persist_directory=persist_db_dir
)

print(f'SUCCESS: saved emebeddings')

# Executing a quick verification test query
# query = "What did they do at BITS Pilani?"
# print(f"\n🔍 Testing Semantic Search Validation for query: '{query}'")

# Performing a vector similarity check and retriving the doc
# matching_docs = vector_store.similarity_search(query, k=5)

# if matching_docs:
#     print("\nTop Matching Content Retreived:")
#     print(matching_docs[0].page_content)
#     print("\nSource Metadata Context:")
#     print(matching_docs[0].metadata)
