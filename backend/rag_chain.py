import os
import re
import difflib
import traceback
import warnings
import random
import json
from typing import List, Optional, Any, Dict
from collections import Counter
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# LangChain / vectorstore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Vertex AI SDK wrappers
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
import vertexai

# Firestore (optional; fallback to in-memory storage if not configured)
from google.cloud import firestore

# -----------------------------------------------------------------------------
# Basic setup & config
# -----------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Load environment variables from .env if present
load_dotenv()

# Default GCP / Vertex configuration (override with env vars)
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "omi-live-backend").strip()
GOOGLE_REGION = os.getenv("GOOGLE_REGION", "us-central1").strip()

# Directory configuration defaults
DATA_DIR = os.environ.get("DATA_DIR", "data")
BRAND_CSV = os.path.join(DATA_DIR, "brand_metric_dataset.csv")
QUIZZES_DIR = os.path.join(DATA_DIR, "quizzes")
VECTORSTORE_DIR = os.environ.get("VECTORSTORE_DIR", os.path.join(DATA_DIR, "omi_index"))

# -----------------------------------------------------------------------------
# Initialize Vertex AI (using Application Default Credentials in GCP)
# -----------------------------------------------------------------------------
try:
    if GOOGLE_CLOUD_PROJECT and GOOGLE_REGION:
        vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_REGION)
        print(f"[INFO] Vertex AI initialized for project={GOOGLE_CLOUD_PROJECT}, region={GOOGLE_REGION}")
    else:
        raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_REGION must be set.")
except Exception as e:
    print(f"[ERROR] Vertex AI initialization failed: {e}")
    traceback.print_exc()

# -----------------------------------------------------------------------------
# Firestore client initialization (optional; use in-memory fallback when not present)
# -----------------------------------------------------------------------------
try:
    db = firestore.Client()
    print("[INFO] Firestore client initialized for rag_chain.")
except Exception as e:
    print(f"[WARN] Firestore initialization failed - using in-memory session store. Error: {e}")
    db = None

# -----------------------------------------------------------------------------
# Persona & Prompt templates for RAG LLM prompt
# -----------------------------------------------------------------------------
SYSTEM_PERSONA = (
    "You are OMI — a warm, concise, upbeat sustainability guide for OMI Live. "
    "Tone: friendly, encouraging, and practical. Use plain language. "
    "Always answer directly and to-the-point first, then add 1–3 short bullets if helpful. "
    "When you don't know, say \"I don't know.\" Never invent facts."
)

QA_PROMPT_GENERAL = PromptTemplate.from_template(
    """{persona}

ONLY use the information in the context. If the answer is not present, say "I don't know."

Context:
{context}

Question: {question}
Direct answer:"""
)

PROACTIVE_SUGGESTIONS = [
    "You can also ask me about eco-friendly laundry swaps.",
    "You can also ask me: 'Why are bees important?'",
    "You can also ask me: 'How do I choose a reef-safe sunscreen?'",
    "You can also ask me about the health benefits of bamboo.",
    "You can also ask me for tips on eating more sustainably.",
    "You can also ask me: 'What are the dangers in conventional tampons?'",
    "You can also ask me about superfood drinks for glowing skin."
]

# -----------------------------------------------------------------------------
# User session manager (Firestore-backed with in-memory fallback)
# -----------------------------------------------------------------------------
class UserSessionManager:
    """Manages per-user session data. Uses Firestore when available; otherwise in-memory."""

    def __init__(self, db_client):
        self.db = db_client
        if self.db:
            self.collection_ref = self.db.collection("user_sessions")
        else:
            self.local_sessions: Dict[str, dict] = {}
            print("[WARN] Using in-memory session store (no Firestore).")

    def _get_default_session(self, user_id: str) -> dict:
        """Return a fresh default session dictionary for a new user."""
        return {
            "user_id": user_id,
            "response_count": 0,
            "current_quiz_session": None,
            "quiz_answers": [],
            "waiting_for_quiz_start": False,
            "quiz_type_pending": None,
            "waiting_for_rank_confirmation": False,
            "brand_to_rank": None,
            "waiting_for_workbook_confirmation": False,
            "waiting_for_user_classification": False,
            "user_type": None,
            "waiting_for_email": False,
            "email_prompt_sent": False,
            "email_prompt_denied": False,
            "offered_suggestions": [],
            "created_at": firestore.SERVER_TIMESTAMP if self.db else datetime.now().isoformat(),
        }

    def get_session(self, user_id: str) -> dict:
        """Return the session dict for a user_id, create if missing."""
        if not self.db:
            return self.local_sessions.setdefault(user_id, self._get_default_session(user_id))
        try:
            doc = self.collection_ref.document(user_id).get()
            if not doc.exists:
                default = self._get_default_session(user_id)
                self.update_session(user_id, default)
                return default
            return doc.to_dict()
        except Exception as e:
            print(f"[ERROR] get_session (Firestore) failed for {user_id}: {e}")
            return self.local_sessions.setdefault(user_id, self._get_default_session(user_id))

    def update_session(self, user_id: str, session_data: dict):
        """Persist the session either to Firestore or in-memory store."""
        if not self.db:
            self.local_sessions[user_id] = session_data
            return
        try:
            self.collection_ref.document(user_id).set(session_data, merge=True)
        except Exception as e:
            print(f"[ERROR] update_session failed for {user_id}: {e}")

# Instantiate a global session manager
_session_manager = UserSessionManager(db)

def get_user_id(session_info: Any) -> str:
    """Utility that returns or generates a user id from session info."""
    if isinstance(session_info, str):
        return session_info
    if isinstance(session_info, dict):
        return session_info.setdefault("user_id", os.urandom(16).hex())
    return os.urandom(16).hex()

# -----------------------------------------------------------------------------
# Utility helpers (text normalization, small NLP utils)
# -----------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Lowercase and remove non-alphanumeric (keeps spaces)."""
    return re.sub(r"[^a-z0-9\s]", "", str(text).lower()).strip()

def _make_key(text: str) -> str:
    """Create normalized key for fuzzy matching brand names."""
    s = str(text).lower()
    s = re.sub(r"&", "and", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

AFFIRMATIONS = {
    "yes", "please", "yes please", "start", "ok", "okay", "sure", "yup", "yep", "of course", "sounds good", "great"
}
NEGATIONS = {"no", "nope", "no thanks", "nah"}

def is_affirmative_response(text: str) -> bool:
    """Return True if the cleaned text is an affirmative (exact match)."""
    cleaned = _clean_text(text)
    return cleaned in AFFIRMATIONS or any(cleaned.startswith(a + " ") for a in AFFIRMATIONS)

def is_negative_response(text: str) -> bool:
    """Return True if the cleaned text is a negative response."""
    cleaned = _clean_text(text)
    return cleaned in NEGATIONS or any(cleaned.startswith(n + " ") for n in NEGATIONS)

def detect_routine_intent(question: str) -> Optional[str]:
    """Detect if the user is asking about a routine — returns 'hair' or 'skin' or None."""
    cleaned_q = _clean_text(question)
    quiz_trigger_keywords = [
        "routine", "regimen", "help with my", "my hair", "my skin", "for my hair", "for my skin", "hair care", "skin care"
    ]
    if 'quiz' in cleaned_q or any(trigger in cleaned_q for trigger in quiz_trigger_keywords):
        hair_keywords = ['hair', 'shampoo', 'conditioner', 'curl', 'scalp', 'haircare']
        skin_keywords = ['skin', 'face', 'acne', 'wrinkle']
        has_hair = any(word in cleaned_q for word in hair_keywords)
        has_skin = any(word in cleaned_q for word in skin_keywords)
        if has_hair and not has_skin:
            return 'hair'
        if has_skin and not has_hair:
            return 'skin'
    return None

# -----------------------------------------------------------------------------
# Brand ranking / dataset loading / fuzzy lookup
# -----------------------------------------------------------------------------
_brand_df: pd.DataFrame = pd.DataFrame()  # cached DataFrame

def get_brand_df() -> pd.DataFrame:
    """Load and normalize brand CSV into a DataFrame (cached)."""
    global _brand_df
    if not _brand_df.empty:
        return _brand_df
    if not os.path.exists(BRAND_CSV):
        print(f"[WARN] brand CSV not found at {BRAND_CSV}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(BRAND_CSV)
        # Normalize column names to snake_case lower
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        # Heuristic: first column is brand name, near-last is final_score (as in your provided CSV)
        df = df.rename(columns={df.columns[0]: "brand_name", df.columns[-2]: "final_score"})
        # Drop rows missing critical fields
        df = df.dropna(subset=['brand_name', 'final_score'])
        df = df[df["brand_name"].astype(str).str.strip().ne("")]
        df["brand_key"] = df["brand_name"].apply(_make_key)
        _brand_df = df
        print(f"[INFO] Loaded brand CSV with {len(df)} rows.")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load brand CSV: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def fuzzy_lookup_brand_candidates(user_text: str, max_results: int = 5) -> List[str]:
    """Return up to max_results matching brand names for a given user_text using difflib."""
    if _clean_text(user_text) in NEGATIONS:
        return []
    df = get_brand_df()
    if df.empty:
        return []
    key = _clean_text(user_text)
    if not key:
        return []
    keys = df["brand_key"].tolist()
    matches = difflib.get_close_matches(key, keys, n=max_results, cutoff=0.75)
    # Also include substring matches (e.g., partial brand words)
    if not matches:
        for bk in keys:
            if key in bk.split():
                matches.append(bk)
            if len(matches) >= max_results:
                break
    return df[df["brand_key"].isin(matches)]["brand_name"].tolist()

def get_brand_ranking_single(brand_name: str) -> str:
    """
    Return formatted ranking for a single brand with explanation line-by-line.
    Format:
      🌍 BRAND — Sustainability score X / 30
      <blank line>
      - Metric 1: value
      - Metric 2: value
      ...
    """
    df = get_brand_df()
    if df.empty:
        return "I don't have brand ranking information available right now."
    key = _make_key(brand_name)
    row_df = df[df["brand_key"] == key]
    if row_df.empty:
        # Try fuzzy alternative
        candidates = fuzzy_lookup_brand_candidates(brand_name, max_results=3)
        if candidates:
            return "I couldn't find an exact match. Did you mean one of these?\n- " + "\n- ".join(candidates)
        return f"I couldn't find a ranking for '{brand_name}'."
    row = row_df.iloc[0]

    # Get numeric score if available
    score = row.get('final_score', 'N/A')
    try:
        # Print with normalized formatting if numeric
        if isinstance(score, (float, int)):
            score_display = f"{float(score)}"
        else:
            score_display = str(score)
    except Exception:
        score_display = str(score)

    # Collect metric columns (exclude meta columns)
    metric_cols = [c for c in df.columns if c not in ['brand_name', 'brand_key', 'final_score']]
    # Prefer readable metric names (turn underscores to spaces & title case)
    def pretty(col):
        return col.replace("_", " ").replace("/", " / ").title()

    lines = []
    # Header: brand + total score
    header = f"🌍 {row['brand_name']} — Sustainability score {score_display} / 30"
    lines.append(header)
    lines.append("")  # blank line as requested (separates header from metrics)

    # For each metric, line-by-line feature and points. Keep ordering stable:
    shown = 0
    for mc in metric_cols:
        if shown >= 12:  # safety cap; show up to 12 metrics to avoid huge lists
            break
        val = row.get(mc, None)
        # Only show non-empty numeric or text values
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        # Convert NaN-like strings to blank
        sval = str(val).strip()
        if sval == "" or sval.lower() in ["nan", "none", "n/a", "na"]:
            continue
        lines.append(f"- {pretty(mc)}: {sval}")
        shown += 1

    # If nothing else shown, fallback to the generic basis sentence
    if shown == 0:
        lines.append("This score is the brand's sustainability rating as defined in our dataset.")

    # Combine and return
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Quiz logic (load quiz JSONs, track state in session_data)
# -----------------------------------------------------------------------------
def offer_quiz(session_data: dict, quiz_type: str) -> str:
    """
    Offer quiz to the user. This sets waiting_for_quiz_start = True and stores pending quiz_type.
    The frontend should show a Yes/No prompt or accept 'yes'/'no' messages to start or skip.
    """
    session_data["waiting_for_quiz_start"] = True
    session_data["quiz_type_pending"] = quiz_type
    return "Want to personalize your experience? Take our quick quiz to identify your hair & skin types so we can recommend the best products. Shall we start?"

def start_quiz(session_data: dict, quiz_type: str) -> str:
    """
    Initialize a quiz session for this user. Clears the 'waiting_for_quiz_start' flag so yes doesn't re-trigger.
    Loads quiz JSON from QUIZZES_DIR/<quiz_type>.json
    """
    session_data["waiting_for_quiz_start"] = False
    session_data["quiz_type_pending"] = None
    session_data["quiz_answers"] = []
    try:
        qpath = os.path.join(QUIZZES_DIR, f"{quiz_type}.json")
        with open(qpath, "r", encoding="utf-8") as fh:
            quiz_data = json.load(fh)
    except Exception:
        # If quiz file missing, return a friendly message and clear state
        session_data["current_quiz_session"] = None
        return "I'm sorry — my quiz materials aren't available right now."
    # Set quiz session
    session_data["current_quiz_session"] = {"quiz_data": quiz_data, "question_idx": 0}
    # Return the first question
    return get_next_quiz_question(session_data)

def get_next_quiz_question(session_data: dict) -> str:
    """Return the next quiz question or finish the quiz if done."""
    quiz_session = session_data.get("current_quiz_session")
    if not quiz_session:
        return "No active quiz session."
    idx = quiz_session["question_idx"]
    questions = quiz_session["quiz_data"].get("questions", [])
    if idx >= len(questions):
        return finish_quiz(session_data)
    q = questions[idx]
    options_text = "\n".join([f"{i+1}. {opt['text']}" for i, opt in enumerate(q.get("options", []))])
    return f"**Question {q.get('id', idx+1)}**: {q.get('question')}\n{options_text}"

def answer_quiz_option(session_data: dict, option_num: int) -> str:
    """
    Record an option answer (by numeric index) and advance the quiz.
    If invalid option, prompt for valid input.
    """
    quiz_session = session_data.get("current_quiz_session")
    if not quiz_session:
        return "No active quiz session."
    questions = quiz_session["quiz_data"].get("questions", [])
    idx = quiz_session["question_idx"]
    if idx >= len(questions):
        return finish_quiz(session_data)
    q = questions[idx]
    opts = q.get("options", [])
    if 1 <= option_num <= len(opts):
        # Save the normalized answer key expected by results_logic
        session_data.setdefault("quiz_answers", []).append(opts[option_num - 1].get("answer"))
        quiz_session["question_idx"] = idx + 1
        # If more questions remain, return next; otherwise finish
        if quiz_session["question_idx"] < len(questions):
            return get_next_quiz_question(session_data)
        else:
            return finish_quiz(session_data)
    else:
        return f"Invalid choice. Please select a number from 1 to {len(opts)}."

def finish_quiz(session_data: dict) -> str:
    """Compute quiz result from recorded answers and return routine."""
    quiz_session = session_data.get("current_quiz_session")
    if not quiz_session:
        return "No active quiz session."
    try:
        quiz_data = quiz_session["quiz_data"]
        answers = session_data.get("quiz_answers", [])
        if not answers:
            session_data["current_quiz_session"] = None
            session_data["quiz_answers"] = []
            return "It looks like you didn't answer any questions. Quiz cancelled."
        # Determine most common answer type (this follows your prior logic)
        most_common_answer = Counter(answers).most_common(1)[0][0]
        result_type = quiz_data["results_logic"].get(most_common_answer)
        routine = quiz_data["routines"].get(result_type, "Sorry, couldn't determine a routine.")
        # Clear quiz state after finishing
        session_data["current_quiz_session"] = None
        session_data["quiz_answers"] = []
        return f"Based on your answers, it looks like you have **{result_type}**!\n\nHere’s a simple routine for you:\n{routine}"
    except Exception as e:
        print(f"[ERROR] finish_quiz failed: {e}")
        traceback.print_exc()
        session_data["current_quiz_session"] = None
        session_data["quiz_answers"] = []
        return "I had a little trouble generating your routine."

# -----------------------------------------------------------------------------
# LLM / Retriever (Vertex AI Gemini + Vertex Embeddings + FAISS)
# -----------------------------------------------------------------------------
_llm: Optional[ChatVertexAI] = None
_retriever = None

def get_llm() -> ChatVertexAI:
    """Lazy initialize the ChatVertexAI LLM client."""
    global _llm
    if _llm is not None:
        return _llm
    try:
        _llm = ChatVertexAI(model_name="gemini-2.5-pro", temperature=0.4, max_output_tokens=1536)
        print("[INFO] ChatVertexAI (gemini-2.5-pro) client created.")
        return _llm
    except Exception as e:
        print(f"[ERROR] Failed to initialize ChatVertexAI: {e}")
        raise

def get_retriever():
    """Load FAISS local index and return a retriever object. Must be called after embeddings available."""
    global _retriever
    if _retriever: 
        return _retriever
    try:
        embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
        if not os.path.exists(VECTORSTORE_DIR):
            print(f"[WARN] VECTORSTORE_DIR ({VECTORSTORE_DIR}) missing. FAISS load will likely fail.")
        vect = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
        _retriever = vect.as_retriever(search_kwargs={"k": 3})
        print("[INFO] FAISS retriever loaded.")
        return _retriever
    except Exception as e:
        print(f"[ERROR] Failed to load FAISS retriever: {e}")
        traceback.print_exc()
        return None

def preload_faiss_index():
    """Helper to warm up retriever and brand data."""
    print("[INFO] Preloading FAISS index and brand dataset...")
    get_retriever()
    get_brand_df()
    print("[INFO] Preload complete.")

# -----------------------------------------------------------------------------
# Follow-up suggestion helper
# -----------------------------------------------------------------------------
def get_follow_up_suggestion(session_data: dict) -> str:
    """Return a proactive suggestion not previously offered to the user this session."""
    offered = session_data.get("offered_suggestions", [])
    available = [s for s in PROACTIVE_SUGGESTIONS if s not in offered]
    if not available:
        return ""
    suggestion = random.choice(available)
    offered.append(suggestion)
    session_data["offered_suggestions"] = offered
    return f"\n\n_Psst... {suggestion}_"

# -----------------------------------------------------------------------------
# Main RAG response pipeline (handles stateful flows + RAG fallback)
# -----------------------------------------------------------------------------
def get_rag_response(question: str, user_id: str) -> str:
    """
    Primary entrypoint for query handling.
    - Loads and updates session state from session manager
    - Handles onboarding, quiz, brand ranking flows
    - Falls back to RAG (FAISS + Gemini) for general questions
    - Returns the assistant's textual response.
    """
    session_data = _session_manager.get_session(user_id)
    raw_q = str(question).strip()
    if not raw_q:
        return "I don't know."

    # Update response count for prompt cadence measures
    session_data['response_count'] = session_data.get('response_count', 0) + 1

    answer = ""
    add_suggestion = True

    # Boolean helpers
    is_affirmative = is_affirmative_response(raw_q)
    is_negative = is_negative_response(raw_q)
    cleaned_q = _clean_text(raw_q)

    # -------------------
    # BLOCK A — handle explicit stateful interactions first (onboarding, quiz, email, ranking confirmations)
    # -------------------

    # ONBOARDING trigger (sent by frontend initially as "__GET_ONBOARDING__")
    if raw_q == "__GET_ONBOARDING__":
        # Mark we are waiting for the user classification (frontend will show role choices)
        session_data['waiting_for_user_classification'] = True
        _session_manager.update_session(user_id, session_data)
        return "To personalize your experience, please let me know who you are."

    # When the session expects the user to identify as eco shopper / creator / brand owner
    if session_data.get("waiting_for_user_classification"):
        # We expect the frontend to send "Eco Shopper", "Creator", or "Brand Owner"
        cleaned_choice = _clean_text(raw_q)
        # Reset the waiting flag immediately to avoid loops (frontend controls UI)
        session_data['waiting_for_user_classification'] = False

        if 'eco shopper' in cleaned_choice or 'eco' in cleaned_choice:
            session_data['user_type'] = 'eco_shopper'
            answer = (
                "🌱 Welcome to Omi Live, your eco living girlie! Ready to explore REAL eco-friendly brands?\n\n"
                "Here’s what you’ll get as part of the Omi Fam:\n"
                "• Our AI Green Rating System to shop transparently\n"
                "• Exclusive offers & discounts\n"
                "• Direct interaction with brand owners\n"
                "• A front-row seat to watch eco-friendly brands grow"
            )
        elif 'creator' in cleaned_choice:
            session_data['user_type'] = 'creator'
            session_data['waiting_for_workbook_confirmation'] = True
            answer = (
                "🎥 Hey there! Are you a creator interested in live shopping?\n\n"
                "With Omi Live, you can monetize your influence through:\n"
                "• Free product samples\n"
                "• Sales commissions\n"
                "• Flat fee partnerships\n"
                "• Expanding your reach with eco-conscious buyers\n\n"
                "We’ve built a Live Sales Workbook for Creators — it shows you how to maximize earnings and grow with us. Want it?"
            )
        elif 'brand owner' in cleaned_choice or 'brand' in cleaned_choice:
            session_data['user_type'] = 'brand_owner'
            session_data['waiting_for_workbook_confirmation'] = True
            answer = (
                "👋 Hi! Welcome to Omi Live — the AI-powered retail tech for eco-friendly brands. Are you a brand owner looking to grow sales?\n\n"
                "We help brands like yours achieve 20% sales conversion through:\n"
                "• A loyal eco-conscious community\n"
                "• Smart product listing & discovery tools\n"
                "• Live storytelling that builds trust\n\n"
                "Would you like our Live Sales Workbook? It’s packed with strategies to boost sales. Want it?"
            )
        else:
            # Unexpected input — re-enable the classification flag so frontend can re-prompt selection
            session_data['waiting_for_user_classification'] = True
            answer = "Please choose a valid option (Eco Shopper, Creator, or Brand Owner)."
        add_suggestion = False
        _session_manager.update_session(user_id, session_data)
        return answer

    # EMAIL capture: If the user provides an email pattern, treat it as submission
    if re.match(r"[^@]+@[^@]+\.[^@]+", raw_q):
        # When user shares email, assume this confirms any pending workbook/email prompt
        session_data['waiting_for_email'] = False
        session_data['email_prompt_sent'] = True
        # Persist or trigger downstream actions here as needed (e.g., add to mailing list)
        user_type = session_data.get('user_type')
        if user_type == 'creator':
            answer = "✅ Perfect! Your creator sales workbook is on the way."
        elif user_type == 'brand_owner':
            answer = "✅ Got it! Your workbook is on the way.\n\nWould you also like to see how brands use our smart product listing and live storytelling to grow sales?"
        else:
            answer = "You’re all set! We’ll share tips, community insights, and opportunities to feature your brand on Omi Live."
        add_suggestion = False
        _session_manager.update_session(user_id, session_data)
        return answer

    # If a quiz is actively in progress (stored in current_quiz_session), process numeric answers
    if session_data.get("current_quiz_session"):
        # Accept numeric answers only; if not numeric, cancel quiz to avoid getting stuck.
        if _clean_text(raw_q).isdigit():
            # Accept numeric option and return next question or finish
            option_num = int(_clean_text(raw_q))
            answer = answer_quiz_option(session_data, option_num)
            add_suggestion = False
            _session_manager.update_session(user_id, session_data)
            return answer
        else:
            # Non-numeric input cancels the quiz (we could be more lenient, but keep it simple)
            session_data["current_quiz_session"] = None
            session_data["quiz_answers"] = []
            _session_manager.update_session(user_id, session_data)
            return "Quiz cancelled. How can I help?"

    # If waiting for quiz start (we sent offer_quiz earlier), listen for 'yes' / 'no'
    if session_data.get("waiting_for_quiz_start"):
        if is_affirmative:
            # Start quiz and return first question
            quiz_type = session_data.get("quiz_type_pending", "skin")
            answer = start_quiz(session_data, quiz_type)
            add_suggestion = False
            _session_manager.update_session(user_id, session_data)
            return answer
        elif is_negative:
            # User declined the quiz
            session_data["waiting_for_quiz_start"] = False
            session_data["quiz_type_pending"] = None
            _session_manager.update_session(user_id, session_data)
            return "No problem — we won't start the quiz now. What else can I help with?"
        else:
            # If the user responds with other text (not yes/no), don't start quiz; treat message normally
            # But do not auto-re-offer quiz immediately to avoid loops.
            # We simply clear waiting_for_quiz_start to avoid repeated prompts.
            session_data["waiting_for_quiz_start"] = False
            session_data["quiz_type_pending"] = None
            _session_manager.update_session(user_id, session_data)
            # fall through to normal handling

    # If waiting for workbook confirmation (user_type creator or brand_owner), accept yes/no
    if session_data.get("waiting_for_workbook_confirmation"):
        if is_affirmative:
            session_data['waiting_for_email'] = True
            session_data['waiting_for_workbook_confirmation'] = False
            _session_manager.update_session(user_id, session_data)
            if session_data.get('user_type') == 'creator':
                return "Awesome! Please share your email and we’ll send the creator workbook + early invites."
            else:
                return "Great! Please drop your email so we can send the Live Sales Workbook and early access materials."
        elif is_negative:
            session_data['waiting_for_workbook_confirmation'] = False
            _session_manager.update_session(user_id, session_data)
            return "No problem! What else can I help you with today?"
        # if neither, let it fall through to other handling

    # If waiting for rank confirmation (we found a brand candidate and asked "would you like ranking?")
    if session_data.get("waiting_for_rank_confirmation"):
        if is_affirmative:
            brand_to_rank = session_data.get("brand_to_rank")
            session_data.update({"waiting_for_rank_confirmation": False, "brand_to_rank": None})
            _session_manager.update_session(user_id, session_data)
            if brand_to_rank:
                return get_brand_ranking_single(brand_to_rank)
            else:
                return "I don't have that brand saved — try asking the brand name again."
        elif is_negative:
            session_data.update({"waiting_for_rank_confirmation": False, "brand_to_rank": None})
            _session_manager.update_session(user_id, session_data)
            return "Got it — no ranking. What else can I help you with?"
        # else fall through

    # If user says "no" while waiting for email prompt — treat as explicit rejection
    if is_negative and session_data.get('waiting_for_email'):
        session_data['email_prompt_denied'] = True
        session_data['waiting_for_email'] = False
        _session_manager.update_session(user_id, session_data)
        return "👍 No worries! We'll keep chatting here."

    # -------------------
    # BLOCK B — stateless or new queries: brand listing, ranking, quiz offers, and RAG fallback
    # -------------------
    # Reset some ephemeral session flags before interpreting new queries
    # (Important: do NOT clear waiting_for_user_classification / waiting_for_email here)
    session_data.update({
        'waiting_for_quiz_start': session_data.get('waiting_for_quiz_start', False),
        # do not wipe quiz_type_pending if already awaiting
    })

    # If user asks for 'rank' or 'ranking' explicitly, list brands available
    if any(w in cleaned_q for w in ['rank', 'ranking', 'rankings']):
        df = get_brand_df()
        if df.empty:
            _session_manager.update_session(user_id, session_data)
            return "I don't have brand ranking information available right now."
        # Provide a concise list (first 50) with no extra blank lines
        brands = df['brand_name'].drop_duplicates().tolist()
        top_list = brands[:50]
        answer = "Here are some brands I can rank for you:\n- " + "\n- ".join(top_list)
        answer += "\n\nYou can ask me to 'rank <brand name>' or ask for more details about any brand."
        add_suggestion = False
        _session_manager.update_session(user_id, session_data)
        return answer

    # If the user's message looks like a brand name, attempt fuzzy match:
    candidates = fuzzy_lookup_brand_candidates(raw_q)
    if len(candidates) == 1:
        # Found single candidate — ask user for confirmation (this prevents accidental ranking)
        brand_name = candidates[0]
        session_data["waiting_for_rank_confirmation"] = True
        session_data["brand_to_rank"] = brand_name
        _session_manager.update_session(user_id, session_data)
        return f"I found the brand '{brand_name}'. Would you like me to provide its sustainability ranking?"
    elif len(candidates) > 1:
        # Multiple candidates — prompt user to pick
        _session_manager.update_session(user_id, session_data)
        return "Did you mean one of these brands? You can ask me to 'rank' one.\n- " + "\n- ".join(candidates)

    # If the user appears to be asking about their routine (hair/skin), offer quiz
    quiz_type = detect_routine_intent(raw_q)
    if quiz_type:
        # Offer quiz; set waiting flag and pending quiz_type. The frontend should keep the input active.
        answer = offer_quiz(session_data, quiz_type)
        session_data['quiz_type_pending'] = quiz_type
        add_suggestion = False
        _session_manager.update_session(user_id, session_data)
        return answer

    # FALLBACK: Use RAG pipeline (FAISS retriever + prompt to LLM)
    retriever = get_retriever()
    if not retriever:
        # If retriever missing, try to answer with LLM alone or a friendly fallback
        try:
            # Try plain LLM call with a short prompt
            prompt = f"{SYSTEM_PERSONA}\nUser: {raw_q}\nAssistant:"
            resp = get_llm().invoke(prompt).content
            _session_manager.update_session(user_id, session_data)
            return resp
        except Exception as e:
            print(f"[ERROR] LLM fallback failed: {e}")
            _session_manager.update_session(user_id, session_data)
            return "I'm having trouble accessing my knowledge base right now. Try again later."

    # Use retriever to get context documents (support different retriever interfaces)
    try:
        if hasattr(retriever, "get_relevant_documents"):
            context_docs = retriever.get_relevant_documents(raw_q)
        elif hasattr(retriever, "retrieve"):
            context_docs = retriever.retrieve(raw_q)
        elif hasattr(retriever, "invoke"):
            context_docs = retriever.invoke(raw_q)
        else:
            context_docs = []
    except Exception as e:
        print(f"[ERROR] retriever call failed: {e}")
        context_docs = []

    # Build context string
    context = "\n\n".join(getattr(d, "page_content", str(d)) for d in context_docs) if context_docs else ""
    if not context.strip():
        # No context found — ask user to rephrase or fallback to general LLM
        try:
            fallback_prompt = f"{SYSTEM_PERSONA}\nUser: {raw_q}\nAssistant:"
            resp = get_llm().invoke(fallback_prompt).content
            _session_manager.update_session(user_id, session_data)
            return resp
        except Exception:
            _session_manager.update_session(user_id, session_data)
            return "I'm not sure how to answer that. Could you try rephrasing?"

    # Compose final prompt and call LLM
    prompt = QA_PROMPT_GENERAL.format(persona=SYSTEM_PERSONA, context=context, question=raw_q)
    try:
        llm_resp = get_llm().invoke(prompt)
        # `invoke` returns an object; ensure we pick .content or str
        answer = getattr(llm_resp, "content", str(llm_resp))
    except Exception as e:
        print(f"[ERROR] LLM invoke error: {e}")
        traceback.print_exc()
        _session_manager.update_session(user_id, session_data)
        return "I had trouble contacting the language model. Try again later."

    # -------------------
    # Final step — consider newsletter / email prompting (only once per session)
    # -------------------
    should_prompt_email = False
    user_type = session_data.get("user_type")
    response_count = session_data.get("response_count", 0)

    # Only prompt if we haven't already prompted or recorded denial
    if not session_data.get("waiting_for_email") and not session_data.get("email_prompt_sent") and not session_data.get("email_prompt_denied"):
        # Simple heuristic: prompt after 3 interactions for ecoshopper/creator/brand_owner
        if user_type == "eco_shopper" and response_count >= 3:
            should_prompt_email = True
        elif user_type in ["creator", "brand_owner"] and not session_data.get("waiting_for_workbook_confirmation") and response_count >= 3:
            should_prompt_email = True

    if should_prompt_email:
        session_data["waiting_for_email"] = True
        session_data["email_prompt_sent"] = True
        # Append a friendly newsletter prompt to the answer
        answer += (
            "\n\n💫 We're totally vibing! I'd love to keep this going - want to join our exclusive newsletter? "
            "Drop your email and we’ll add you to our Omi Fam newsletter."
        )
        add_suggestion = False

    # Optionally append a proactive suggestion if nothing else appended
    if add_suggestion:
        answer += get_follow_up_suggestion(session_data)

    # Persist session updates and return
    _session_manager.update_session(user_id, session_data)
    return answer

# -----------------------------------------------------------------------------
# CLI test harness (convenient local testing)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔎 Preloading FAISS index and brand dataset (if available)...")
    preload_faiss_index()
    print("🤖 OMI Bot (rag_chain) ready — type 'quit' to exit.")
    user = "cli_user"
    while True:
        try:
            q = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if q.lower() in ("quit", "exit"):
            break
        resp = get_rag_response(q, user)
        print("\nOMI:", resp)
