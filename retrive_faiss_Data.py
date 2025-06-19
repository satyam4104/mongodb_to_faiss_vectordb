from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Initialize Ollama embeddings (must match the model used when creating the FAISS index)
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
print("hi1")

# Load the FAISS index from the local directory
vector_store = FAISS.load_local("./faiss_db", embeddings, allow_dangerous_deserialization=True)
print("hi2")

# Retrieve all documents from the FAISS index
docs = vector_store.docstore._dict  # Access the internal document store
print("hi3")

# Print the content of each document
for doc_id, doc in docs.items():
    print(f"Document ID: {doc_id}")
    print(f"Content: {doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 50)