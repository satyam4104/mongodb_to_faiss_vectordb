from langchain_ollama import OllamaEmbeddings
import psycopg2

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
text = "This is a test document"
vector = embeddings.embed_query(text)  # This returns a list of floats

conn = psycopg2.connect("dbname=postgres user=postgres password=1331")
cur = conn.cursor()

cur.execute(
    "INSERT INTO doc (content, embedding) VALUES (%s, %s)",
    (text, vector)
)

conn.commit()
cur.close()
conn.close()
