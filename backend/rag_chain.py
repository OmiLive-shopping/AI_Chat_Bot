# rag_chain.py (Full and Final Version - Cloud Ready)
import os
import re
import difflib
import traceback
import warnings
import random
from typing import List, Optional, Any
import json
from collections import Counter
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# LangChain / embeddings / vectorstore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# --- Vertex AI Integrations ---
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
import vertexai

# --- NEW: Import Firestore ---
from google.cloud import firestore

# =========================
# Setup & Globals
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

load_dotenv()

# --- GCP / Vertex config ---
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "omi-live-backend").strip()
GOOGLE_REGION = os.getenv("GOOGLE_REGION", "us-central1").strip()

# --- Initialize Vertex AI (Correctly using ADC) ---
try:
    if GOOGLE_CLOUD_PROJECT and GOOGLE_REGION:
        vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_REGION)
        print(f"[INFO] Vertex AI initialized for project: {GOOGLE_CLOUD_PROJECT}, region: {GOOGLE_REGION}")
    else:
        raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_REGION must be set.")
except Exception as e:
    print(f"[ERROR] Failed to initialize Vertex AI: {e}")
    traceback.print_exc()

# --- NEW: Initialize Firestore Client ---
try:
    db = firestore.Client()
    print("[INFO] Firestore client initialized successfully in rag_chain.")
except Exception as e:
    print(f"[ERROR] Failed to initialize Firestore client in rag_chain: {e}")
    db = None

DATA_DIR = os.environ.get('DATA_DIR', 'data')
FAQ_PATH = os.path.join(DATA_DIR, "omi_faq.txt")
KB_PATH = os.path.join(DATA_DIR, "omilive_knowledge_base.txt")
BRAND_CSV = os.path.join(DATA_DIR, "brand_metric_dataset.csv")
WORKBOOK_FILENAME = "Omi_Live_-_Live_Sales_Tactical_Workbook.doc"
WORKBOOK_PATH = os.path.join(DATA_DIR, WORKBOOK_FILENAME)

QUIZZES_DIR = "quizzes"
VECTORSTORE_DIR = os.environ.get('VECTORSTORE_DIR', os.path.join(DATA_DIR, "omi_index"))

# Persona
SYSTEM_PERSONA = (
    "You are OMI — a warm, concise, upbeat sustainability guide for OMI Live. "
    "Tone: friendly, encouraging, and practical. Use plain language. "
    'Always answer directly and to-the-point first, then add 1–3 short bullets if helpful. '
    'When you don\'t know, say "I don\'t know." Never invent facts.'
)

# Varied greetings
VARIED_GREETINGS = [
    "Hey there! 👋 I'm Omi Bot, your guide to sustainable living and eco-friendly shopping!",
    "Hello! 🌱 I'm Omi Bot, here to help you discover amazing sustainable brands and products!",
    "Hi friend! ✨ I'm Omi Bot, ready to explore eco-friendly living and conscious shopping with you!",
    "Greetings! 🛍️ I'm Omi Bot, your companion for sustainable brands and live shopping experiences!",
    "Hey! 🌿 I'm Omi Bot, excited to help you on your journey to more eco-conscious living!"
]

# Prompts
QA_PROMPT_GENERAL = PromptTemplate.from_template(
    """{persona}

ONLY use the information in the context. If the answer is not present, say "I don't know."

Context:
{context}

Question: {question}
Direct answer:"""
)

QA_PROMPT_BRAND = PromptTemplate.from_template(
    """{persona}

You are answering a BRAND ranking/overview. Use only the brand table information provided after this instruction.
Return a concise summary: brand name, total score, and any available sub-scores (if present). If not found, say "I don't know."

Question: {question}

Brand info table (may contain multiple — choose the best match):
{context}

Answer:"""
)

# =========================
# User Session Management (UPDATED with Firestore)
# =========================
class UserSessionManager:
    def __init__(self, db_client):
        self.db = db_client
        if self.db:
            self.collection_ref = self.db.collection("user_sessions")
        else:
            # Fallback for local testing if Firestore isn't available
            self.local_sessions = {}
            print("[WARN] Firestore client is not available. Using in-memory session storage.")

    def get_session(self, user_id: str) -> dict:
        """Get a user's session data from Firestore, or create a new one."""
        if not self.db:  # Firestore unavailable fallback
            if user_id not in self.local_sessions:
                self.local_sessions[user_id] = self._get_default_session(user_id)
            return self.local_sessions[user_id]

        doc_ref = self.collection_ref.document(user_id)
        doc = doc_ref.get()

        if doc.exists:
            session_data = doc.to_dict()
            # Ensure legacy sessions have all needed keys
            if 'response_count' not in session_data:
                session_data['response_count'] = 0
            if 'greeting_index' not in session_data:
                session_data['greeting_index'] = 0
            return session_data
        else:
            default_session = self._get_default_session(user_id)
            doc_ref.set(default_session)
            return default_session

    def update_session(self, user_id: str, updates: dict):
        """Update a user's session data in Firestore."""
        if not self.db:  # Firestore unavailable fallback
            if user_id in self.local_sessions:
                self.local_sessions[user_id].update(updates)
            return

        try:
            doc_ref = self.collection_ref.document(user_id)
            doc_ref.update(updates)
        except Exception as e:
            print(f"[ERROR] Failed to update session in Firestore for user {user_id}: {e}")
            traceback.print_exc()

    def _get_default_session(self, user_id: str) -> dict:
        """Returns the default dictionary for a new user session."""
        return {
            "user_id": user_id,
            "interaction_count": 0,
            "email": None,
            "last_prompted_at": None,
            "response_count": 0,
            "greeting_index": 0,
            "created_at": firestore.SERVER_TIMESTAMP if self.db else datetime.now().isoformat()
        }

# Instantiate the manager with the database client
_session_manager = UserSessionManager(db)


def get_user_id(session: Any) -> str:
    """Gets/sets a unique ID in the user's secure Flask session cookie."""
    if isinstance(session, dict):
        if "user_id" not in session:
            session["user_id"] = os.urandom(16).hex()
        return session["user_id"]
    try:
        return str(session)
    except Exception:
        return os.urandom(16).hex()

# =========================
# Routine Intent Detection
# =========================
def detect_routine_intent(question: str) -> Optional[str]:
    cleaned_q = _clean_text(question)
    hair_keywords = ['hair', 'shampoo', 'conditioner', 'curl', 'scalp', 'haircare']
    skin_keywords = ['skin', 'face', 'acne', 'wrinkle', 'dry skin', 'oily skin', 'routine', 'regimen', 'skincare']
    has_hair = any(word in cleaned_q for word in hair_keywords)
    has_skin = any(word in cleaned_q for word in skin_keywords)
    if has_skin and not has_hair:
        return 'skin'
    elif has_hair and not has_skin:
        return 'hair'
    else:
        return None

# Globals created once
_llm: Optional[ChatVertexAI] = None
_retriever = None
_brand_df: pd.DataFrame = pd.DataFrame()
_waiting_for_workbook_confirmation: bool = False
_current_quiz_session = None
_quiz_answers = []
GREETINGS = ("hi", "hello", "hey", "good morning", "good evening", "good afternoon")

# =========================
# Utilities
# =========================
def _safe_read(path: str) -> str:
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        traceback.print_exc()
        return ""

def _make_key(text: str) -> str:
    s = str(text).lower()
    s = re.sub(r"&", "and", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _clean_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", str(text).lower()).strip()

# =========================
# Vertex helpers: LLM + Embeddings
# =========================
def get_llm() -> ChatVertexAI:
    """
    Return a ChatVertexAI instance. Try the newer model name first
    and fallback to a more generic alias if necessary.
    """
    global _llm
    if _llm is not None:
        return _llm

    preferred_models = ["gemini-1.5-pro", "gemini-pro", "text-bison@001"]
    for model in preferred_models:
        try:
            print(f"[INFO] Attempting to initialize ChatVertexAI with model: {model}")
            candidate = ChatVertexAI(
                model_name=model,
                temperature=0.4,
                max_output_tokens=512,
                project=GOOGLE_CLOUD_PROJECT,
                location=GOOGLE_REGION
            )
            test_resp = candidate.invoke("Hello, are you online?")
            if not getattr(test_resp, "content", None):
                print(f"[WARN] Model {model} responded with empty content. Trying next.")
                continue
            _llm = candidate
            print(f"[INFO] ChatVertexAI initialized and verified with model: {model}")
            return _llm
        except Exception as e:
            print(f"[WARN] Could not initialize model {model}: {e}")
            traceback.print_exc()
            continue
    raise RuntimeError("Failed to initialize any Vertex AI model. Check Vertex SDK, permissions and model availability.")


def get_embeddings() -> VertexAIEmbeddings:
    # Single place to control the embedding model name
    try:
        # Pass the project ID and location explicitly
        return VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_REGION
        )
    except Exception as e:
        print(f"[ERROR] Could not initialize VertexAIEmbeddings: {e}")
        traceback.print_exc()
        raise

# =========================
# Documents → Embeddings → Retriever
# =========================
def build_retriever(save_local: bool = True):
    """
    Build FAISS vectorstore from source docs (FAQ + KB).
    If save_local=True, save the vectorstore to VECTORSTORE_DIR for future loads.
    """
    faq_text = _safe_read(FAQ_PATH)
    kb_text = _safe_read(KB_PATH)
    if not faq_text and not kb_text:
        raise RuntimeError("No source text found. Ensure omi_faq.txt and omilive_knowledge_base.txt exist in data/.")
    docs = []
    if faq_text: docs += TextLoader(FAQ_PATH, encoding="utf-8").load()
    if kb_text: docs += TextLoader(KB_PATH, encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()
    vect = FAISS.from_documents(chunks, embeddings)
    if save_local:
        try:
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            vect.save_local(VECTORSTORE_DIR)
            print(f"[INFO] Saved FAISS vectorstore to: {VECTORSTORE_DIR}")
        except Exception as e:
            print("[WARN] Could not save FAISS vectorstore to disk:", e)
    return vect.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def load_retriever_from_disk():
    """
    Try to load the FAISS vectorstore saved in VECTORSTORE_DIR.
    Returns a retriever or raises on failure.
    """
    embeddings = get_embeddings()
    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(f"Vectorstore directory not found: {VECTORSTORE_DIR}")
    print(f"[INFO] Loading FAISS vectorstore from disk: {VECTORSTORE_DIR}")
    vect = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return vect.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def get_retriever():
    """
    Return the global retriever. If it's None, try to load from disk; if that fails, build it.
    """
    global _retriever
    if _retriever is not None:
        return _retriever
    try:
        _retriever = load_retriever_from_disk()
        print("[INFO] Retriever loaded from disk.")
    except Exception as e:
        print(f"[WARN] Could not load retriever from disk: {e}. Building from source...")
        _retriever = build_retriever(save_local=True)
        print("[INFO] Retriever built from source and saved locally.")
    return _retriever

def retrieve_context(query: str, k: int = 5) -> str:
    try:
        retriever = get_retriever()
        docs = retriever.get_relevant_documents(query)[:k]
        return "\n\n".join(d.page_content for d in docs if d and d.page_content)
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        return ""

def preload_faiss_index():
    """
    Public helper: ensure the retriever is initialized (load from disk or build & save).
    Call this at container startup so first HTTP request doesn't pay the cost.
    """
    print("[INFO] preload_faiss_index() called.")
    return get_retriever()

# =========================
# Brand data & helpers
# =========================
def load_brand_df() -> pd.DataFrame:
    if not os.path.exists(BRAND_CSV):
        print(f"[WARN] Brand CSV not found at {BRAND_CSV}. Brand features will be disabled.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(BRAND_CSV)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        possible_score_cols = [c for c in df.columns if "final" in c and "score" in c]
        if possible_score_cols:
            df = df.rename(columns={possible_score_cols[0]: "final_score"})
        if "brand_name" not in df.columns:
            for c in df.columns:
                if "brand" in c and "name" in c:
                    df = df.rename(columns={c: "brand_name"})
                    break
        df = df[df["brand_name"].astype(str).str.strip().ne("")]
        df["brand_key"] = df["brand_name"].apply(_make_key)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load brand metrics: {e}")
        return pd.DataFrame()

def get_brand_df() -> pd.DataFrame:
    global _brand_df
    if _brand_df.empty:
        _brand_df = load_brand_df()
    return _brand_df

def list_all_brands() -> List[str]:
    df = get_brand_df()
    if df.empty or "brand_name" not in df.columns:
        return []
    return sorted({str(b).strip() for b in df["brand_name"].dropna() if str(b).strip()})

def list_all_brands_str() -> str:
    return ", ".join(list_all_brands())

def fuzzy_lookup_brand_candidates(user_text: str, top_n: int = 5, strict: bool = False) -> List[str]:
    df = get_brand_df()
    if df.empty:
        return []
    key = _make_key(user_text)
    keys = df["brand_key"].tolist()
    cutoff = 0.85 if strict else 0.5
    matches = difflib.get_close_matches(key, keys, n=top_n, cutoff=cutoff)
    if not matches and len(key) >= 3:
        subset = df[df["brand_key"].str.contains(re.escape(key))]
        if not subset.empty:
            return subset["brand_name"].tolist()[:top_n]
    return df[df["brand_key"].isin(matches)]["brand_name"].tolist()

def get_brand_ranking_single(best_name: str) -> str:
    df = get_brand_df()
    if df.empty:
        return ""
    key = _make_key(best_name)
    matched = df[df["brand_key"] == key]
    if matched.empty:
        matched = df[df["brand_key"].str.contains(re.escape(key))]
        if matched.empty:
            return ""
    row = matched.iloc[0]
    score = row.get("final_score", "N/A")
    breakdown_cols = [
        "recycled/upcycled_materials",
        "end_of_life_solutions_(compostable_packaging/zero_waste)",
        "worker_welfare/living_wage",
        "local_sourcing",
        "sustainability_data_accessibility",
        "marketing_honesty/_certifications",
    ]
    breakdown_text = []
    for col in breakdown_cols:
        if col in df.columns and pd.notna(row.get(col, None)):
            label = col.replace("_", " ").title()
            breakdown_text.append(f"- {label}: {row[col]}")
    breakdown = "\n".join(breakdown_text)
    if breakdown:
        breakdown = f"\n\n**Breakdown:**\n{breakdown}"
    return f"🌍 *{row['brand_name']}* — Sustainability score **{score} / 30**.{breakdown}"

def respond_list_all_brands() -> str:
    brands_str = list_all_brands_str()
    if not brands_str:
        return "I don't currently have brand rankings in my dataset."
    return "📊 **Yes! I track these brands:**\n" + brands_str

# =========================
# Quiz JSON loader
# =========================
def load_quiz(file_name: str) -> dict:
    path = os.path.join(QUIZZES_DIR, file_name)
    if not os.path.exists(path):
        print(f"[ERROR] Quiz file not found: {path}")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Malformed quiz JSON {path}: {e}")
        return {}
    except Exception as e:
        print(f"[ERROR] Could not load quiz {path}: {e}")
        return {}

# =========================
# Quiz logic
# =========================
def start_quiz(quiz_type: str) -> str:
    global _current_quiz_session, _quiz_answers
    quiz_file = "hair.json" if quiz_type == "hair" else "skin.json"
    quiz_data = load_quiz(quiz_file)
    if not quiz_data:
        return "Sorry, I couldn't load the quiz."
    _current_quiz_session = {
        "quiz_data": quiz_data,
        "question_idx": 0,
        "quiz_type": quiz_type
    }
    _quiz_answers = []
    return (
        f"### ✨ {quiz_type.capitalize()} Routine Quiz\n"
        f"I can definitely help with a {quiz_type} routine! To personalize it,\n"
        f"I'll ask a few quick questions.\n\n"
        f"**Ready to start the {quiz_type} quiz?** Type **'start'** to begin!"
    )

def get_next_quiz_question() -> str:
    global _current_quiz_session
    if not _current_quiz_session:
        return "No active quiz."
    idx = _current_quiz_session["question_idx"]
    questions = _current_quiz_session["quiz_data"]["questions"]
    if idx >= len(questions):
        return finish_quiz()

    q = questions[idx]
    options_text = "\n".join([f"{i+1}. {opt['text']}" for i, opt in enumerate(q["options"])])
    return f"**Q{q['id']}**: {q['question']}\n{options_text}"

def answer_quiz_option(option_num: int) -> str:
    global _current_quiz_session, _quiz_answers
    if not _current_quiz_session:
        return "No active quiz."
    questions = _current_quiz_session["quiz_data"]["questions"]
    idx = _current_quiz_session["question_idx"]
    if idx >= len(questions):
        return finish_quiz()

    q = questions[idx]
    if 1 <= option_num <= len(q["options"]):
        _quiz_answers.append(q["options"][option_num-1]["type"])
        _current_quiz_session["question_idx"] += 1
        return get_next_quiz_question()
    else:
        return f"Invalid choice. Please select a number between 1 and {len(q['options'])}."

# =========================
# Quiz recommendation helper (Markdown-formatted)
# =========================
def get_quiz_recommendation(result_type: str, category: str) -> str:
    query = f"{result_type} {category} routine recommendation"
    context = retrieve_context(query, k=3)

    llm = get_llm()
    prompt = f"""
You are OMI. Based on the context below, recommend products and a routine for {category} type: {result_type}.
Be concise, practical, and friendly. Only use the information provided in the context.

Context:
{context}

Format the response in markdown with:
- A clear H3 header for the routine
- Short paragraph (1–2 sentences) summary
- Bullet points for steps with **bold** product names
- Optional links in [text](url) format
- Finish with a short tip block prefixed by "💡 Tip:"

Response:
"""
    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or (str(resp) if resp is not None else None) or "I don't know."
        text = text.strip()
        if not re.search(r"^#{3}\s", text):
            text = f"### 🌿 Recommended {category.capitalize()} Routine for *{result_type}*\n\n{text}"
        return text
    except Exception as e:
        print(f"[ERROR] LLM recommendation failed: {e}")
        return "I don't know."

def finish_quiz() -> str:
    global _current_quiz_session, _quiz_answers
    if not _quiz_answers:
        _current_quiz_session = None
        return "No answers recorded."

    result_type = Counter(_quiz_answers).most_common(1)[0][0]
    quiz_type = _current_quiz_session["quiz_type"]
    recommendation = get_quiz_recommendation(result_type, quiz_type)
    _current_quiz_session = None
    _quiz_answers = []

    return (
        f"**Your {quiz_type} type:** *{result_type}*\n\n"
        f"{recommendation}\n\n"
        f"📘 Want me to send the *Omi Live Tactical Workbook* for **{quiz_type}** care?"
    )

# =========================
# Core RAG answer
# =========================
def answer_with_context(question: str, context: str) -> str:
    llm = get_llm()
    prompt = QA_PROMPT_GENERAL.format(persona=SYSTEM_PERSONA, context=context, question=question)
    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or (str(resp) if resp is not None else "I don't know.")
        return text.strip()
    except Exception as e:
        print(f"[ERROR] LLM invocation failed: {e}")
        return "⚠️ Error generating an answer."

def get_rag_response(question: str, chat_session: Any) -> str:
    global _waiting_for_workbook_confirmation

    # --- Session Management ---
    user_id = get_user_id(chat_session)
    session_data = _session_manager.get_session(user_id)
    
    # Use a dictionary for session updates to minimize DB writes
    session_updates = {}

    if not question or not str(question).strip():
        return "I don't know."

    raw_q = str(question).strip()
    cleaned_q = _clean_text(raw_q)
    print(f"[DEBUG] User '{user_id}' asked: {raw_q}")

    # --- Email Submission ---
    if re.match(r"[^@]+@[^@]+\.[^@]+", raw_q) and session_data.get('email') is None:
        session_updates['email'] = raw_q
        _session_manager.update_session(user_id, session_updates)
        return "🎉 **Thanks for signing up!** You'll hear from us soon. How can I help you next?"

    # --- Newsletter Prompt ---
    if (session_data.get('response_count', 0) >= 3 and session_data.get('email') is None and
            not _waiting_for_workbook_confirmation and not _current_quiz_session and
            not any(cleaned_q.startswith(g) for g in GREETINGS)):
        session_updates['last_prompted_at'] = datetime.now().isoformat()
        session_updates['response_count'] = session_data.get('response_count', 0) + 1
        _session_manager.update_session(user_id, session_updates)
        return (
            "We're totally vibing! 💫 **Join our newsletter?**\n"
            "- Early access to sustainable brand deals\n"
            "- New eco finds and community tips\n"
            "- Free live shopping workbook for creators/brands\n\n"
            "**Drop your email** and I'll add you. 🌱"
        )

    response = "" # Default empty response
    
    # ===== Quiz handling =====
    if _current_quiz_session:
        if cleaned_q.isdigit():
            response = answer_quiz_option(int(cleaned_q))
        elif cleaned_q in ["start", "yes", "begin"]:
            response = get_next_quiz_question()
        else:
            response = "Please enter the number of your choice for the quiz or type **'start'** to begin."

    # ===== Routine Intent Detection & Quiz Trigger =====
    elif not _current_quiz_session and session_data.get('email') is not None:
        routine_type = detect_routine_intent(raw_q)
        if routine_type:
            response = start_quiz(routine_type)

    # ===== Quiz commands =====
    elif cleaned_q in {"start hair quiz", "hair quiz"}:
        response = start_quiz("hair")
    elif cleaned_q in {"start skin quiz", "skin quiz"}:
        response = start_quiz("skin")

    # ===== Greetings =====
    elif any(cleaned_q.startswith(g) for g in GREETINGS):
        greeting_index = session_data.get('greeting_index', 0)
        response = f"{VARIED_GREETINGS[greeting_index % len(VARIED_GREETINGS)]} I also have a workbook — *Omi Live Tactical Workbook* 📘. Would you like me to send it?"
        session_updates['greeting_index'] = (greeting_index + 1)

    # ===== Workbook logic =====
    elif "workbook" in cleaned_q:
        _waiting_for_workbook_confirmation = True
        response = "I have the *Omi Live Tactical Workbook* 📘 — do you want it in **.doc** format where you can download?"

    elif _waiting_for_workbook_confirmation and cleaned_q in {"yes", "sure", "okay", "ok", "yep", "yeah"}:
        _waiting_for_workbook_confirmation = False
        response = "Great! 🎉 You can download the workbook here: [**Download Workbook**](/get_workbook)"

    # ===== Brand rankings =====
    elif re.search(r"\b(rank(ing)?\s*brands?|brand\s*ranking|do\s+you\s+rank)\b", cleaned_q):
        response = respond_list_all_brands()
    
    elif re.match(r"^\s*(rank|ranking)\s+(.*)$", raw_q, flags=re.IGNORECASE):
        m = re.match(r"^\s*(rank|ranking)\s+(.*)$", raw_q, flags=re.IGNORECASE)
        brand_candidate = m.group(2).strip()
        if not brand_candidate:
            response = respond_list_all_brands()
        else:
            candidates = fuzzy_lookup_brand_candidates(brand_candidate, top_n=5, strict=False)
            if not candidates:
                response = "I couldn't find that brand. Try one from my list: " + list_all_brands_str()
            elif len(candidates) == 1:
                response = get_brand_ranking_single(candidates[0]) or "I don't know."
            else:
                response = "Did you mean one of these?\n- " + "\n- ".join(candidates)

    # This is a broad match, so it's placed later in the logic
    elif len(cleaned_q.split()) <= 4:
        brand_candidates = fuzzy_lookup_brand_candidates(raw_q, top_n=1, strict=True)
        if brand_candidates:
            response = get_brand_ranking_single(brand_candidates[0])

    # If any intent was matched and a response was generated, return it now
    if response:
        session_updates['response_count'] = session_data.get('response_count', 0) + 1
        _session_manager.update_session(user_id, session_updates)
        return response

    # ===== General RAG (Fallback) =====
    context = retrieve_context(raw_q, k=5)
    if not context.strip():
        print("[INFO] No context found in FAISS. Falling back to Vertex AI.")
        llm = get_llm()
        try:
            resp = llm.invoke(raw_q)
            answer = getattr(resp, "content", None) or str(resp) or "I don't know."
            if not re.search(r"^#{1,3}\s", answer):
                answer = f"### ✨ Here's what I found\n\n{answer.strip()}"
        except Exception as e:
            print(f"[ERROR] Vertex fallback failed: {e}")
            answer = "⚠️ Error occurred. Please try again."
    else:
        answer = answer_with_context(raw_q, context)

    if any(k in cleaned_q for k in ["brand", "brands", "rank", "score", "sustainable brands"]):
        if list_all_brands_str():
            answer += "\n\n📊 You can ask me to rank a brand (e.g., **rank Patagonia**)."

    if "i don't know" in answer.lower() and len(cleaned_q.split()) <= 5:
        answer = "Could you clarify your question a bit?"

    session_updates['response_count'] = session_data.get('response_count', 0) + 1
    session_updates['interaction_count'] = session_data.get('interaction_count', 0) + 1
    _session_manager.update_session(user_id, session_updates)
    
    return answer

# =========================
# CLI test
# =========================
if __name__ == "__main__":
    import sys
    print("[INFO] Preloading FAISS index for CLI...")
    try:
        preload_faiss_index()
        print("[INFO] ✅ FAISS index ready.")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to preload FAISS index: {e}")
        sys.exit(1)

    print("🔍 Testing LLM connectivity...")
    try:
        llm = get_llm()
        ping = llm.invoke("Hello, are you online?")
        print("[TEST] Gemini working ✅:", bool(getattr(ping, "content", "")))
    except Exception as e:
        print("❌ Gemini failed:", e)
        sys.exit(1)

    print("\n🤖 OMI Bot is ready! Start chatting (type 'quit' to exit).")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        response = get_rag_response(user_input, "cli_user")
        print(f"OMI: {response}")