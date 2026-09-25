import os
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"


def build_vector_store():
    client = chromadb.PersistentClient(path=DB_DIR)

    # Sentence-transformers all-MiniLM-L6-v2 embedding model running locally
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    ids = []
    metadatas = []

    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_DIR, filename)
            doc_id = os.path.splitext(filename)[0]
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()

            documents.append(content)
            ids.append(doc_id)
            metadatas.append({"source": filename, "doc_id": doc_id})

    collection.upsert(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Successfully indexed {len(documents)} documents into collection '{COLLECTION_NAME}' at '{DB_DIR}'.")


if __name__ == "__main__":
    build_vector_store()