import pymongo
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import uuid
import logging
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Connect to MongoDB (running locally)
try:
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client["Resume_data"]
    collection = db["users"]
    logger.info("Successfully connected to MongoDB")
    print("hi")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

# Step 2: Custom MongoDB Loader
def load_mongo_documents():
    """Load documents from MongoDB and convert to LangChain Document objects."""
    try:
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
            else:
                logger.warning(f"Skipping document with empty content: {metadata['source']}")
        logger.info(f"Loaded {len(documents)} documents from MongoDB")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents from MongoDB: {e}")
        raise

# Load documents
try:
    documents = load_mongo_documents()
    if not documents:
        logger.error("No valid documents loaded from MongoDB. Check 'Resume_data.users' collection.")
        raise ValueError("No valid documents to process")
    logger.debug(f"Sample document content: {documents[0].page_content[:100]}...")
except Exception as e:
    logger.error(f"Failed to load documents: {e}")
    raise

# Step 3: Split documents into chunks
try:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks")
    logger.debug(f"Sample chunk content: {chunks[0].page_content[:100]}...")
except Exception as e:
    logger.error(f"Error splitting documents: {e}")
    raise

# Step 4: Initialize Ollama embeddings
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    logger.info("Initialized Ollama embeddings")
    print("hi2")
except Exception as e:
    logger.error(f"Failed to initialize Ollama embeddings: {e}")
    raise

# Step 5: Store chunks in FAISS (local vector database)
try:
    start_time = time.time()
    logger.debug("Starting FAISS document indexing")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("./faiss_db")
    logger.info(f"Successfully stored chunks in FAISS (took {time.time() - start_time:.2f} seconds)")
    print("hi3")
except Exception as e:
    logger.error(f"Failed to store chunks in FAISS: {e}")
    raise

# Step 6: Set up the LLM (Ollama with Llama3.1)
try:
    llm = Ollama(model="llama3.2:1b", base_url="http://127.0.0.1:11434", timeout=60)
    logger.info("Initialized Ollama LLM")
    print("hi4")
except Exception as e:
    logger.error(f"Failed to initialize Ollama LLM: {e}")
    raise

# Step 7: Define the RAG prompt template
prompt_template = """
You are a helpful assistant. Use the following context to answer the user's question accurately and concisely.

Context: {context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

print("hi5")

# Step 8: Create the RetrievalQA chain
try:
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    logger.info("Initialized RetrievalQA chain")
except Exception as e:
    logger.error(f"Failed to initialize RetrievalQA chain: {e}")
    raise

print("hi6")

# Step 9: Function to query the RAG system
def query_rag(question):
    try:
        result = qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": [doc.page_content for doc in result["source_documents"]]
        }
    except Exception as e:
        logger.error(f"Error querying RAG system: {e}")
        raise

print("hi7")

# Example usage
if __name__ == "__main__":
    try:
        question = "What are the main skills listed in the documents?"
        response = query_rag(question)
        print("Answer:", response["answer"])
        print("\nSource Documents:")
        for doc in response["source_documents"]:
            print(doc[:200] + "..." if len(doc) > 200 else doc)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise