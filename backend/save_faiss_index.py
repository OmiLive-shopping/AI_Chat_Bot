# save_faiss_index.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

os.makedirs("data", exist_ok=True)

docs_faq = TextLoader("data/omi_faq.txt", encoding="utf-8").load()
docs_kb = TextLoader("data/omilive_knowledge_base.txt", encoding="utf-8").load()
documents = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(docs_faq + docs_kb)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("data/faiss_index")
print("✅ FAISS index saved.")
