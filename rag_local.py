import pymongo
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import uuid

# Step 1: Connect to MongoDB (running locally)
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["Resume_data"]
collection = db["users"]
print("hi")

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
print("hi2")

# Step 3: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print("hi3")

# Step 4: Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
print("hi4")

# Step 5: Store chunks in FAISS (local vector database)
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("./faiss_db")
print("hi5")

# Step 6: Set up the LLM (Ollama with Llama3.1)
llm = OllamaLLM(model="llama3.2:1b", base_url="http://127.0.0.1:11434", timeout=60)
print("hi6")

# Step 7: Define the RAG prompt template
prompt_template = """
You are a helpful assistant. Use the following context to answer the user's question accurately and concisely.

Context: {context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
print("hi7")

# Step 8: Create the RetrievalQA chain
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
print("hi8")

# Step 9: Function to query the RAG system
def query_rag(question):
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "source_documents": [doc.page_content for doc in result["source_documents"]]
    }

print("hi9")

# Example usage
if __name__ == "__main__":
    question = "What are the main skills listed in the documents?"
    response = query_rag(question)
    print("Answer:", response["answer"])
    print("\nSource Documents:")
    for doc in response["source_documents"]:
        print(doc[:200] + "..." if len(doc) > 200 else doc)