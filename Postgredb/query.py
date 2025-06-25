from pymongo import  MongoClient
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import uuid
import psycopg2
from sqlalchemy import create_engine
import json

# Step 1: Initialize Ollama embeddings (for retriever)
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
print("Ollama embeddings initialized")

# Step 2: Connect to PostgreSQL vector store
CONNECTION_STRING = "postgresql+psycopg2://postgres:1331@localhost:5432/vector_db"
COLLECTION_NAME = "resume_vectors"

vector_store = PGVector(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    use_jsonb=True
)

# Step 6: Set up the LLM (Ollama with Llama3.1)
llm = OllamaLLM(model="llama3.2:1b", base_url="http://127.0.0.1:11434", timeout=60)
print("hi9")

# Step 7: Define the RAG prompt template
prompt_template = """
You are a helpful assistant. Use the following context to answer the user's question accurately and concisely.

Context: {context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
print("hi10")

# Step 8: Create the RetrievalQA chain
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
print("hi11")

# Step 9: Function to query the RAG system
def query_rag(question):
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "source_documents": [
            {
                "content": doc.page_content,
                "metadata": json.loads(doc.metadata["full_document"])  # Parse JSON string back to dict
            } for doc in result["source_documents"]
        ]
    }

print("hi12")

# Example usage
if __name__ == "__main__":
    question = "What are the main skills of satyam?"
    response = query_rag(question)
    print("Answer:", response["answer"])
    print("\nSource Documents:")
    for doc in response["source_documents"]:
        print("Content:", doc["content"][:50] + "..." if len(doc["content"]) > 50 else doc["content"])
        print("Metadata:", doc["metadata"])