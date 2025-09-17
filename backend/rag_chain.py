import os
import re
import difflib
import traceback
import warnings
import random
from typing import List, Optional
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

# =========================
# Setup & Globals
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

load_dotenv()
# --- Use the Google Vertex API Key ---
# Note: Google's best practice is to use Application Default Credentials.
# This code assumes your API key is configured as an environment variable.
GOOGLE_VERTEX_API_KEY = os.getenv("GOOGLE_VERTEX_API")
print("[DEBUG] GOOGLE_VERTEX_API loaded:", bool(GOOGLE_VERTEX_API_KEY))
# Set the environment variable for the library
# This is an alternative to standard gcloud auth.
if GOOGLE_VERTEX_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_VERTEX_API_KEY

DATA_DIR = os.environ.get('DATA_DIR', 'data')
FAQ_PATH = os.path.join(DATA_DIR, "omi_faq.txt")
KB_PATH = os.path.join(DATA_DIR, "omilive_knowledge_base.txt")
BRAND_CSV = os.path.join(DATA_DIR, "brand_metric_dataset.csv")
WORKBOOK_FILENAME = "Omi_Live_-_Live_Sales_Tactical_Workbook.doc"
WORKBOOK_PATH = os.path.join(DATA_DIR, WORKBOOK_FILENAME)

QUIZZES_DIR = "quizzes"  # folder containing hair.json, skin.json

# Where FAISS index and metadata will be saved/loaded
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
# User Session Management
# =========================
class UserSessionManager:
    _sessions_file = os.path.join(DATA_DIR, "user_sessions.json")

    def __init__(self):
        self.sessions = self._load_sessions()

    def _load_sessions(self) -> dict:
        try:
            if os.path.exists(self._sessions_file):
                with open(self._sessions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load sessions: {e}")
        return {}

    def _save_sessions(self):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(self._sessions_file, 'w', encoding='utf-8') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save sessions: {e}")

    def get_session(self, user_id: str) -> dict:
        """Get a user's session data, creating a new one if it doesn't exist."""
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                "interaction_count": 0,
                "email": None,
                "last_prompted_at": None,
                "response_count": 0,
                "greeting_index": 0
            }
            self._save_sessions()
        elif "response_count" not in self.sessions[user_id]:
            self.sessions[user_id]["response_count"] = 0
            self.sessions[user_id]["greeting_index"] = 0
            self._save_sessions()
        return self.sessions[user_id]

    def update_session(self, user_id: str, updates: dict):
        """Update a user's session data and save the file."""
        if user_id in self.sessions:
            self.sessions[user_id].update(updates)
            self._save_sessions()

# Global session manager
_session_manager = UserSessionManager()

# Helper function to get a user ID (Placeholder for CLI / web integration)
def get_user_id():
    # NOTE: In production, replace with real user id (cookie, auth ID, IP fallback, etc.)
    return "cli_user"

# =========================
# Routine Intent Detection
# =========================
def detect_routine_intent(question: str) -> Optional[str]:
    """
    Detects if the user is asking for a hair or skin routine.
    Returns 'hair', 'skin', or None.
    """
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

# Quiz globals
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
# LLM
# =========================
def get_llm() -> ChatVertexAI:
    global _llm
    if _llm is not None:
        return _llm
    if not GOOGLE_VERTEX_API_KEY:
        raise RuntimeError("GOOGLE_VERTEX_API missing. Add it to your .env.")
    
    # --- Using ChatVertexAI with the Gemini Pro model ---
    _llm = ChatVertexAI(
        model_name="gemini-pro",
        temperature=0.4,
        project="main-entropy-467501-b6" # 👈 Your Project ID added here
    )
    return _llm

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
    if faq_text:
        docs += TextLoader(FAQ_PATH, encoding="utf-8").load()
    if kb_text:
        docs += TextLoader(KB_PATH, encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    print(f"[DEBUG] Total chunks: {len(chunks)}")

    # --- Using VertexAIEmbeddings for embeddings ---
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project="main-entropy-467501-b6" # 👈 Your Project ID added here
    )

    print("[INFO] Creating FAISS index from documents (this may take a moment)...")
    vect = FAISS.from_documents(chunks, embeddings)

    if save_local:
        try:
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            vect.save_local(VECTORSTORE_DIR)
            print(f"[INFO] Saved FAISS vectorstore to: {VECTORSTORE_DIR}")
        except Exception as e:
            print("[WARN] Could not save FAISS vectorstore to disk:", e)
            traceback.print_exc()

    return vect.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def load_retriever_from_disk():
    """
    Try to load the FAISS vectorstore saved in VECTORSTORE_DIR.
    Returns a retriever or raises on failure.
    """
    # --- Using VertexAIEmbeddings for embeddings ---
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project="main-entropy-467501-b6" # 👈 Your Project ID added here
    )
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

    # Try loading persistent index first (faster)
    try:
        _retriever = load_retriever_from_disk()
        print("[INFO] Retriever loaded from disk.")
        return _retriever
    except Exception as e:
        print("[WARN] Could not load retriever from disk:", e)

    # Fallback: build retriever from source texts and save local copy
    try:
        _retriever = build_retriever(save_local=True)
        print("[INFO] Retriever built from source and saved locally.")
        return _retriever
    except Exception as e:
        print("[ERROR] Failed to build retriever:", e)
        traceback.print_exc()
        raise

def retrieve_context(query: str, k: int = 5) -> str:
    try:
        retriever = get_retriever()
        docs = retriever.get_relevant_documents(query)[:k]
        return "\n\n".join(d.page_content for d in docs if d and d.page_content)
    except Exception as e:
        print("[ERROR] Retrieval failed:", e)
        traceback.print_exc()
        return ""

# Add an explicit preload function you can call from app.py
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
        return pd.DataFrame(columns=["brand_name", "final_score", "brand_key"])
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
        print("[DEBUG] Brand metrics loaded:", df.shape)
        return df
    except Exception as e:
        print("[ERROR] Failed to load brand metrics:", e)
        traceback.print_exc()
        return pd.DataFrame(columns=["brand_name", "final_score", "brand_key"])

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
        traceback.print_exc()
        return {}
    except Exception as e:
        print(f"[ERROR] Could not load quiz {path}: {e}")
        traceback.print_exc()
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
    # Use the retriever to find the best routine for this type
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
        # Use safe extraction
        text = getattr(resp, "content", None) or (str(resp) if resp is not None else None) or "I don't know."
        text = text.strip()
        # Ensure there is at least an H3 header
        if not re.search(r"^#{3}\s", text):
            text = f"### 🌿 Recommended {category.capitalize()} Routine for *{result_type}*\n\n{text}"
        return text
    except Exception as e:
        print("[ERROR] LLM recommendation failed:", e)
        traceback.print_exc()
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
        print("[ERROR] LLM invocation failed:", e)
        traceback.print_exc()
        return "⚠️ Error generating an answer."

def get_rag_response(question: str, chat_history: Optional[list] = None) -> str:
    global _waiting_for_workbook_confirmation

    # --- Session Management ---
    user_id = get_user_id()
    session = _session_manager.get_session(user_id)
    session['interaction_count'] += 1

    if 'response_count' not in session:
        session['response_count'] = 0
    if 'greeting_index' not in session:
        session['greeting_index'] = 0

    if not question or not str(question).strip():
        return "I don't know."

    raw_q = str(question).strip()
    cleaned_q = _clean_text(raw_q)
    print(f"[DEBUG] Incoming question: {raw_q}")
    print(f"[DEBUG] Cleaned question: {cleaned_q}")

    # --- Email Submission ---
    if re.match(r"[^@]+@[^@]+\.[^@]+", raw_q) and session.get('email') is None:
        _session_manager.update_session(user_id, {'email': raw_q})
        return "🎉 **Thanks for signing up!** You'll hear from us soon. How can I help you next?"

    # --- Newsletter Prompt after 4 bot responses (inc. greeting) ---
    if (session.get('response_count', 0) >= 3 and session.get('email') is None and
            not _waiting_for_workbook_confirmation and not _current_quiz_session and
            not any(cleaned_q.startswith(g) for g in GREETINGS)):
        _session_manager.update_session(user_id, {'last_prompted_at': datetime.now().isoformat()})
        session['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session['response_count']})
        return (
            "We're totally vibing! 💫 **Join our newsletter?**\n"
            "- Early access to sustainable brand deals\n"
            "- New eco finds and community tips\n"
            "- Free live shopping workbook for creators/brands\n\n"
            "**Drop your email** and I'll add you. 🌱"
        )

    # ===== Quiz handling =====
    if _current_quiz_session:
        if cleaned_q.isdigit():
            response = answer_quiz_option(int(cleaned_q))
            session['response_count'] += 1
            _session_manager.update_session(user_id, {'response_count': session['response_count']})
            return response
        elif cleaned_q in ["start", "yes", "begin"]:
            response = get_next_quiz_question()
            session['response_count'] += 1
            _session_manager.update_session(user_id, {'response_count': session['response_count']})
            return response
        else:
            session['response_count'] += 1
            _session_manager.update_session(user_id, {'response_count': session['response_count']})
            return "Please enter the number of your choice for the quiz or type **'start'** to begin."

    # ===== Routine Intent Detection & Quiz Trigger =====
    if not _current_quiz_session and session.get('email') is not None:
        routine_type = detect_routine_intent(raw_q)
        if routine_type:
            response = start_quiz(routine_type)
            session['response_count'] += 1
            _session_manager.update_session(user_id, {'response_count': session['response_count']})
            return response

    # ===== Quiz commands =====
    if cleaned_q in {"start hair quiz", "hair quiz"}:
        response = start_quiz("hair")
        session['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session['response_count']})
        return response
    if cleaned_q in {"start skin quiz", "skin quiz"}:
        response = start_quiz("skin")
        session['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session['response_count']})
        return response

    # ===== Greetings =====
    if any(cleaned_q.startswith(g) for g in GREETINGS):
        greeting_index = session.get('greeting_index', 0)
        greeting = VARIED_GREETINGS[greeting_index % len(VARIED_GREETINGS)]
        response = f"{greeting} I also have a workbook — *Omi Live Tactical Workbook* 📘. Would you like me to send it?"
        session['response_count'] += 1
        session['greeting_index'] = (greeting_index + 1) % len(VARIED_GREETINGS)
        _session_manager.update_session(user_id, {
            'response_count': session['response_count'],
            'greeting_index': session['greeting_index']
        })
        return response

    # ===== Workbook logic =====
    if "workbook" in cleaned_q:
        _waiting_for_workbook_confirmation = True
        response = "I have the *Omi Live Tactical Workbook* 📘 — do you want it in **.doc** format where you can download?"
        session['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session['response_count']})
        return response

    if _waiting_for_workbook_confirmation and cleaned_q in {"yes", "sure", "okay", "ok", "yep", "yeah"}:
        _waiting_for_workbook_confirmation = False
        response = "Great! 🎉 You can download the workbook here: [**Download Workbook**](/get_workbook)"
        session['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session['response_count']})
        return response

    # ===== Brand rankings =====
    if re.search(r"\b(rank(ing)?\s*brands?|brand\s*ranking|do\s+you\s+rank)\b", cleaned_q):
        response = respond_list_all_brands()
        session['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session['response_count']})
        return response

    m = re.match(r"^\s*(rank|ranking)\s+(.*)$", raw_q, flags=re.IGNORECASE)
    if m:
        brand_candidate = m.group(2).strip()
        if not brand_candidate:
            response = respond_list_all_brands()
            session['response_count'] += 1
            _session_manager.update_session(user_id, {'response_count': session['response_count']})
            return response
        candidates = fuzzy_lookup_brand_candidates(brand_candidate, top_n=5, strict=False)
        if not candidates:
            response = "I couldn't find that brand. Try one from my list: " + list_all_brands_str()
            session['response_count'] += 1
            _session_manager.update_session(user_id, {'response_count': session['response_count']})
            return response
        if len(candidates) == 1:
            ans = get_brand_ranking_single(candidates[0])
            response = ans or "I don't know."
            session['response_count'] += 1
            _session_manager.update_session(user_id, {'response_count': session['response_count']})
            return response
        response = "Did you mean one of these?\n- " + "\n- ".join(candidates)
        session['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session['response_count']})
        return response

    brand_candidates = fuzzy_lookup_brand_candidates(raw_q, top_n=3, strict=False)
    if brand_candidates and len(cleaned_q.split()) <= 4:
        strict_candidates = fuzzy_lookup_brand_candidates(raw_q, top_n=1, strict=True)
        target = strict_candidates[0] if strict_candidates else (brand_candidates[0] if len(brand_candidates) == 1 else None)
        if target:
            ans = get_brand_ranking_single(target)
            if ans:
                session['response_count'] += 1
                _session_manager.update_session(user_id, {'response_count': session['response_count']})
                return ans
        if len(brand_candidates) > 1:
            response = "Did you mean one of these?\n- " + "\n- ".join(brand_candidates)
            session['response_count'] += 1
            _session_manager.update_session(user_id, {'response_count': session['response_count']})
            return response

    # ===== General RAG =====
    context = retrieve_context(raw_q, k=5)
    if not context.strip():
        response = "I don't know."
        session['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session['response_count']})
        return response

    answer = answer_with_context(raw_q, context)

    if any(k in cleaned_q for k in ["brand", "brands", "rank", "score", "sustainable brands"]):
        if list_all_brands():
            answer += "\n\n📊 You can ask me to rank a brand (e.g., **rank Patagonia**)."

    if "i don't know" in answer.lower() and len(cleaned_q.split()) <= 5:
        response = "Could you clarify your question a bit?"
        session['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session['response_count']})
        return response

    session['response_count'] += 1
    _session_manager.update_session(user_id, {'response_count': session['response_count']})
    return answer

# =========================
# CLI test
# =========================
if __name__ == "__main__":
    print("🔧 Building retriever...")
    try:
        _ = get_retriever()
        print("✅ Retriever ready.")
    except Exception as e:
        print("❌ Retriever failed:", e)
        traceback.print_exc()
        exit(1)

    print("🔍 Testing LLM connectivity...")
    try:
        llm = get_llm()
        ping = llm.invoke("Hello, are you online?")
        print("[TEST] Gemini working ✅:", bool(getattr(ping, "content", "")))
    except Exception as e:
        print("❌ Gemini failed:", e)
        traceback.print_exc()
        exit(1)

    # Simple test loop
    print("\n🤖 OMI Bot is ready! Start chatting (type 'quit' to exit).")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        response = get_rag_response(user_input)
        print(f"OMI: {response}")