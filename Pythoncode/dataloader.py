import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def load_json_files(directory):
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".json"):
            with open(os.path.join(directory, file), "r") as f:
                data = json.load(f)
                for section, content in data.items():
                    documents.append(Document(page_content=content, metadata={"section": section}))
    return documents

def build_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("./embeddings/faiss_index")
    return vector_store