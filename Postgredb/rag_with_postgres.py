import pymongo
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

# Step 1: Connect to MongoDB (running locally)
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["Resume_data"]
collection = db["users"]
print("hi")

print(collection)

# Step 2: Custom MongoDB Loader
def load_mongo_documents():
    cursor = collection.find()
    documents = []
    for doc in cursor:
        # Extract all fields for page_content
        content_parts = []
        for key, value in doc.items():
            if key == '_id':
                continue
            if isinstance(value, list):
                formatted_value = ", ".join(str(v) for v in value if v)
            elif isinstance(value, dict):
                formatted_value = json.dumps(value)
            elif value is None:
                formatted_value = ""
            else:
                formatted_value = str(value)
            if formatted_value.strip():
                content_parts.append(f"{key}: {formatted_value}")
        content = "\n".join(content_parts) if content_parts else ""

        # Prepare metadata with the full document
        doc_metadata = {k: v for k, v in doc.items() if k != '_id'}
        for key, value in doc_metadata.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                doc_metadata[key] = str(value)
        metadata = {
            "source": str(doc.get("_id", str(uuid.uuid4()))),
            "full_document": json.dumps(doc_metadata)
        }

        if content.strip():
            documents.append(Document(page_content=content, metadata=metadata))
    
    if not documents:
        raise ValueError("No valid documents to process")
    return documents

# Load documents
documents = load_mongo_documents()
print("hi2")

print(documents)

# Step 3: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print("hi3")

print(chunks)

# Step 4: Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
print("hi4")

print(embeddings)

# Step 5: Connect to PostgreSQL and store vectors using PGVector
CONNECTION_STRING = "postgresql+psycopg2://postgres:1331@localhost:5432/vector_db"
COLLECTION_NAME = "resume_vectors"

print("hi5")

# Create the PostgreSQL table with pgvector extension if not exists
def initialize_postgres():
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="vector_db",
        user="postgres",
        password="1331"
    )
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    cursor.close()
    conn.close()

print("hi6")

initialize_postgres()

print("hi7")

# Store chunks in PostgreSQL with pgvector
vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    use_jsonb=True
)
print("hi8")

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
    question = "What are the main skills listed in the documents?"
    response = query_rag(question)
    print("Answer:", response["answer"])
    print("\nSource Documents:")
    for doc in response["source_documents"]:
        print("Content:", doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"])
        print("Metadata:", doc["metadata"])