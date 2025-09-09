import os
import traceback
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time
import logging

# =========================
# Setup logging
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# Paths
# =========================
DATA_DIR = os.environ.get('DATA_DIR', 'data')
FAQ_PATH = os.path.join(DATA_DIR, "omi_faq.txt")
KB_PATH = os.path.join(DATA_DIR, "omilive_knowledge_base.txt")
BRAND_CSV_PATH = os.path.join(DATA_DIR, "brand_metric_dataset_clean.csv")

VECTORSTORE_DIR = os.path.join(DATA_DIR, "omi_index")

# =========================
# Helpers
# =========================
def _safe_read(path: str) -> str:
    """Safely read a text file with retry logic for cloud environments."""
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
            time.sleep(1)  # Wait before retrying

def load_csv_as_text(path: str) -> str:
    """Convert a CSV (brand metrics) into readable text for embedding."""
    if not os.path.exists(path):
        logger.warning(f"CSV not found: {path}")
        return ""
    
    try:
        df = pd.read_csv(path)
        text_rows = []
        
        # Add column names as context
        columns_info = f"Dataset columns: {', '.join(df.columns)}"
        text_rows.append(columns_info)
        text_rows.append("=" * 50)
        
        # Add sample data rows
        for idx, row in df.iterrows():
            if idx >= 100:  # Limit to first 100 rows to avoid excessive content
                text_rows.append(f"... and {len(df) - 100} more rows")
                break
                
            row_text = " | ".join(f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]))
            text_rows.append(f"Row {idx + 1}: {row_text}")
        
        return "\n".join(text_rows)
    except Exception as e:
        logger.error(f"Failed to convert CSV to text: {e}")
        traceback.print_exc()
        return ""

# =========================
# Build FAISS
# =========================
def build_faiss_index():
    """Build and save FAISS index from documents."""
    logger.info("Loading source documents...")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory not found: {DATA_DIR}")
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    docs = []
    loaded_files = []

    # FAQ and KB
    for path in [FAQ_PATH, KB_PATH]:
        if os.path.exists(path):
            try:
                text_content = _safe_read(path)
                if text_content:
                    docs.extend(TextLoader(path, encoding="utf-8").load())
                    loaded_files.append(os.path.basename(path))
                    logger.info(f"Loaded: {os.path.basename(path)}")
                else:
                    logger.warning(f"Empty or unreadable file: {path}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
        else:
            logger.warning(f"File not found, skipping: {path}")

    # Brand CSV as text
    if os.path.exists(BRAND_CSV_PATH):
        try:
            brand_text = load_csv_as_text(BRAND_CSV_PATH)
            if brand_text:
                # Save temporary text file to load
                temp_csv_txt = os.path.join(DATA_DIR, "brand_temp.txt")
                with open(temp_csv_txt, "w", encoding="utf-8") as f:
                    f.write(brand_text)
                
                docs.extend(TextLoader(temp_csv_txt, encoding="utf-8").load())
                loaded_files.append("brand_metrics")
                logger.info("Loaded: brand_metrics.csv")
                
                # Clean up temporary file
                if os.path.exists(temp_csv_txt):
                    os.remove(temp_csv_txt)
            else:
                logger.warning("Brand CSV conversion produced no content")
        except Exception as e:
            logger.error(f"Failed to process brand CSV: {e}")
            traceback.print_exc()
    else:
        logger.warning(f"Brand CSV not found, skipping: {BRAND_CSV_PATH}")

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

    # Validate we have chunks to process
    if not chunks:
        raise RuntimeError("No text chunks created from documents")

    # Embeddings with error handling
    try:
        logger.info("Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Test the embeddings model
        test_embedding = embeddings.embed_query("test")
        if not test_embedding or len(test_embedding) == 0:
            raise ValueError("Embeddings model returned empty vector")
            
        logger.info("Embeddings model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load embeddings model: {e}")
        traceback.print_exc()
        raise

    # Build FAISS with progress indication
    logger.info("Creating FAISS index...")
    try:
        vect = FAISS.from_documents(chunks, embeddings)
        
        # Save with directory creation
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        vect.save_local(VECTORSTORE_DIR)
        
        # Verify the index was saved
        if (os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")) and 
            os.path.exists(os.path.join(VECTORSTORE_DIR, "index.pkl"))):
            logger.info(f"✅ FAISS index saved successfully to: {VECTORSTORE_DIR}")
            logger.info(f"Index size: {len(chunks)} chunks")
        else:
            raise RuntimeError("FAISS index files were not created properly")
            
    except Exception as e:
        logger.error(f"Failed to create or save FAISS index: {e}")
        traceback.print_exc()
        raise

# =========================
# Run with proper error handling
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
        exit(1)  # Exit with error code for script failure