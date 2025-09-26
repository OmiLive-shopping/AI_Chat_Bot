import os
import traceback
import time
import logging
import json
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- UPDATED: Import Vertex AI and its embedding model ---
from langchain_google_vertexai import VertexAIEmbeddings
import vertexai

# = an======================
# Setup logging
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# Paths & Config
# =========================
load_dotenv()

# --- NEW: Add GCP / Vertex AI Configuration ---
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
GOOGLE_REGION = os.getenv("GOOGLE_REGION", "us-central1").strip()

DATA_DIR = os.environ.get('DATA_DIR', 'data')
FAQ_PATH = os.path.join(DATA_DIR, "omi_faq.txt")
KB_PATH = os.path.join(DATA_DIR, "omilive_knowledge_base.txt")
BRAND_CSV_PATH = os.path.join(DATA_DIR, "brand_metric_dataset.csv")
QUIZZES_DIR = os.path.join(DATA_DIR, 'quizzes')
OMI_INTRO_PATH = os.path.join(DATA_DIR, "omi_intro.txt")

VECTORSTORE_DIR = os.environ.get('VECTORSTORE_DIR', os.path.join(DATA_DIR, "omi_index"))

# --- NEW: Initialize Vertex AI ---
try:
    if GOOGLE_CLOUD_PROJECT and GOOGLE_REGION:
        vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_REGION)
        logger.info(f"Vertex AI initialized for project: {GOOGLE_CLOUD_PROJECT}, region: {GOOGLE_REGION}")
    else:
        raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_REGION must be set in your .env file.")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI: {e}")
    traceback.print_exc()
    exit(1)


# =========================
# Helpers (No changes needed here)
# =========================
def _safe_read(path: str) -> str:
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return ""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    logger.warning(f"File is empty: {path}")
                return content
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to read {path} after {max_retries} attempts: {e}")
                traceback.print_exc()
                return ""
            logger.warning(f"Attempt {attempt + 1} failed for {path}, retrying...")
            time.sleep(1)

def load_csv_as_text(path: str) -> str:
    if not os.path.exists(path):
        logger.warning(f"CSV not found: {path}")
        return ""
    try:
        df = pd.read_csv(path)
        text_rows = []
        text_rows.append(f"Dataset columns: {', '.join(df.columns)}")
        text_rows.append("=" * 50)
        for idx, row in df.iterrows():
            if idx >= 100:
                text_rows.append(f"... and {len(df) - 100} more rows")
                break
            row_text = " | ".join(f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]))
            text_rows.append(f"Row {idx + 1}: {row_text}")
        return "\n".join(text_rows)
    except Exception as e:
        logger.error(f"Failed to convert CSV to text: {e}")
        traceback.print_exc()
        return ""

def load_quizzes_as_text(quizzes_dir: str) -> list:
    if not os.path.exists(quizzes_dir):
        logger.warning(f"Quizzes directory not found: {quizzes_dir}")
        return []
    quiz_texts = []
    for filename in os.listdir(quizzes_dir):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(quizzes_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                quiz_data = json.load(f)
            quiz_name = quiz_data.get("quiz_name", "Unnamed Quiz")
            quiz_text = [f"Quiz: {quiz_name}", "=" * 50]
            for q in quiz_data.get("questions", []):
                quiz_text.append(f"Q{q.get('id')}: {q.get('question', '')}")
                options = q.get("options", [])
                option_texts = [opt.get("text", "") for opt in options if "text" in opt]
                quiz_text.append("Options: " + ", ".join(option_texts))
            quiz_texts.append("\n".join(quiz_text))
            logger.info(f"Loaded quiz: {filename}")
        except Exception as e:
            logger.error(f"Failed to load quiz {filename}: {e}")
    return quiz_texts

# =========================
# Build FAISS
# =========================
def build_faiss_index():
    logger.info("Loading source documents...")
    docs = []
    loaded_files = []

    # 1️⃣ Load FAQ and KB
    for path in [FAQ_PATH, KB_PATH]:
        text_content = _safe_read(path)
        if text_content:
            docs.extend(TextLoader(path, encoding="utf-8").load())
            loaded_files.append(os.path.basename(path))
            logger.info(f"Loaded: {os.path.basename(path)}")

    # 2️⃣ Load Omi intro
    if os.path.exists(OMI_INTRO_PATH):
        try:
            docs.extend(TextLoader(OMI_INTRO_PATH, encoding="utf-8").load())
            loaded_files.append("omi_intro")
            logger.info("Loaded: omi_intro.txt")
        except Exception as e:
            logger.error(f"Failed to load omi_intro.txt: {e}")

    # 3️⃣ Load Brand CSV
    brand_text = load_csv_as_text(BRAND_CSV_PATH)
    if brand_text:
        temp_csv_txt = os.path.join(DATA_DIR, "brand_temp.txt")
        with open(temp_csv_txt, "w", encoding="utf-8") as f:
            f.write(brand_text)
        docs.extend(TextLoader(temp_csv_txt, encoding="utf-8").load())
        loaded_files.append("brand_metrics")
        logger.info("Loaded: brand_metrics.csv")
        os.remove(temp_csv_txt)

    # 4️⃣ Load quizzes
    quiz_texts = load_quizzes_as_text(QUIZZES_DIR)
    for i, q_text in enumerate(quiz_texts):
        temp_quiz_txt = os.path.join(DATA_DIR, f"quiz_temp_{i}.txt")
        with open(temp_quiz_txt, "w", encoding="utf-8") as f:
            f.write(q_text)
        docs.extend(TextLoader(temp_quiz_txt, encoding="utf-8").load())
        os.remove(temp_quiz_txt)
        loaded_files.append(f"quiz_{i}")
    
    if not docs:
        raise RuntimeError("No documents loaded. Check your data files!")

    logger.info(f"Total documents loaded: {len(docs)} from {len(loaded_files)} sources")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Total chunks after splitting: {len(chunks)}")

    # --- UPDATED: Use Vertex AI Embeddings ---
    logger.info("Initializing Vertex AI Embeddings model...")
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_REGION
    )

    # Build FAISS
    logger.info("Building FAISS index from document chunks... (This may take a moment)")
    vect = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vect.save_local(VECTORSTORE_DIR)
    logger.info(f"✅ FAISS index saved to {VECTORSTORE_DIR}, total chunks: {len(chunks)}")

# =========================
# Run
# =========================
if __name__ == "__main__":
    start_time = time.time()
    logger.info("Starting FAISS index build process...")
    
    try:
        build_faiss_index()
        end_time = time.time()
        logger.info(f"FAISS index build completed successfully in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"FAISS index build failed: {e}")
        traceback.print_exc()
        exit(1)
