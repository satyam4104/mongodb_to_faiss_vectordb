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

# Step 1: Connect to MongoDB (running locally)
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["Resume_data"]
collection = db["users"]

# Step 2: Custom MongoDB Loader
def load_mongo_documents():
    cursor = collection.find()
    documents = []
    for doc in cursor:
        skill = doc.get("skill", "")
        if isinstance(skill, list):
            content = ", ".join(str(s) for s in skill if s)
        elif not isinstance(skill, str):
            content = str(skill) if skill is not None else ""
        else:
            content = skill
        metadata = {"source": str(doc.get("_id", str(uuid.uuid4())))}
        if content.strip():
            documents.append(Document(page_content=content, metadata=metadata))
    if not documents:
        raise ValueError("No valid documents to process")
    return documents

# Load documents
documents = load_mongo_documents()
print(documents)

# Step 3: Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

print(embeddings)


# Step 4: Connect to PostgreSQL and store vectors using PGVector
CONNECTION_STRING = "postgresql+psycopg2://postgres:1331@localhost:5432/vector_db"
COLLECTION_NAME = "resume_vectors"

# Step 5: Create the PostgreSQL table with pgvector extension if not exists
def initialize_postgres():
    conn = psycopg2.connect(
        host="localhost",        # Database host
        port="5432",            # Default PostgreSQL port
        database="vector_db",        # Replace with your database name
        user="postgres",        # Replace with your username
        password="1331" # Replace with your password
    )
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    cursor.close()
    conn.close()

print("hi6")

initialize_postgres()


# Store chunks in PostgreSQL with pgvector
vector_store = PGVector.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    use_jsonb=True
)