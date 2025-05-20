from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

# === Load data ===
def load_reviews(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    return pd.read_csv(csv_path)

# === Create Langchain documents ===
def create_documents(df: pd.DataFrame) -> list:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        content = f"""
        Name: {row.get('Name', 'Unknown')}
        Place: {row.get('Place You Want to Review About', 'N/A')}
        Review: {row.get('Review', '')}
        """
        documents.append(Document(page_content=content.strip(), id=str(i)))
        ids.append(str(i))
    
    return documents, ids

# === Main function ===
def create_or_load_vector_store(csv_path="reviews.csv", db_path="./chroma_langchain_db"):
    df = load_reviews(csv_path)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    is_new_db = not os.path.exists(db_path)

    # Create vector store
    vector_store = Chroma(
        collection_name="reviews",
        persist_directory=db_path,
        embedding_function=embeddings
    )

    # Add documents only if new DB
    if is_new_db:
        documents, ids = create_documents(df)
        vector_store.add_documents(documents=documents, ids=ids)

    return vector_store.as_retriever(search_kwargs={"k": 5})

# === Export retriever ===
retriever = create_or_load_vector_store()
