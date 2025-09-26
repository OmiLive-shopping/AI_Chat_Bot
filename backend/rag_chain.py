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

QUIZZES_DIR = os.path.join(DATA_DIR, "quizzes")
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

# =========================
# User Session Management
# =========================
class UserSessionManager:
    def __init__(self, db_client):
        self.db = db_client
        if self.db:
            self.collection_ref = self.db.collection("user_sessions")
        else:
            self.local_sessions = {}
            print("[WARN] Firestore client is not available. Using in-memory session storage.")

    def get_session(self, user_id: str) -> dict:
        if not self.db:
            if user_id not in self.local_sessions:
                self.local_sessions[user_id] = self._get_default_session(user_id)
            return self.local_sessions[user_id]
        doc_ref = self.collection_ref.document(user_id)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else self._get_default_session(user_id)

    def update_session(self, user_id: str, session_data: dict):
        if not self.db:
            self.local_sessions[user_id] = session_data
            return
        try:
            self.collection_ref.document(user_id).set(session_data)
        except Exception as e:
            print(f"[ERROR] Failed to update session for user {user_id}: {e}")

    def _get_default_session(self, user_id: str) -> dict:
        return {
            "user_id": user_id, "interaction_count": 0, "email": None,
            "last_prompted_at": None, "response_count": 0, "greeting_index": 0,
            "waiting_for_workbook_confirmation": False, "current_quiz_session": None,
            "quiz_answers": [],
            "created_at": firestore.SERVER_TIMESTAMP if self.db else datetime.now().isoformat()
        }

_session_manager = UserSessionManager(db)

def get_user_id(session: Any) -> str:
    if isinstance(session, dict):
        if "user_id" not in session:
            session["user_id"] = os.urandom(16).hex()
        return session["user_id"]
    return str(session) if session else os.urandom(16).hex()

# =========================
# Core Bot Logic (Stateless Helpers)
# =========================
def detect_routine_intent(question: str) -> Optional[str]:
    cleaned_q = _clean_text(question)
    hair_keywords = ['hair', 'shampoo', 'conditioner', 'curl', 'scalp', 'haircare', 'hair care']
    skin_keywords = ['skin', 'face', 'acne', 'wrinkle', 'dry skin', 'oily skin', 'routine', 'regimen', 'skincare', 'skin care']
    has_hair = any(word in cleaned_q for word in hair_keywords)
    has_skin = any(word in cleaned_q for word in skin_keywords)
    if has_skin and not has_hair: return 'skin'
    if has_hair and not has_skin: return 'hair'
    return None

def _clean_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", str(text).lower()).strip()

def _make_key(text: str) -> str:
    s = str(text).lower()
    s = re.sub(r"&", "and", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Globals for stateless, expensive-to-create objects
_llm: Optional[ChatVertexAI] = None
_retriever = None
_brand_df: pd.DataFrame = pd.DataFrame()
GREETINGS = ("hi", "hello", "hey", "good morning", "good evening", "good afternoon")

def get_llm() -> ChatVertexAI:
    global _llm
    if _llm: return _llm
    _llm = ChatVertexAI(model_name="gemini-2.5-pro", temperature=0.5, max_output_tokens=1536)
    return _llm

def get_retriever():
    global _retriever
    if _retriever: return _retriever
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    try:
        vect = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    except:
        docs = TextLoader(FAQ_PATH).load() + TextLoader(KB_PATH).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80).split_documents(docs)
        vect = FAISS.from_documents(chunks, embeddings)
        vect.save_local(VECTORSTORE_DIR)
    _retriever = vect.as_retriever(search_kwargs={"k": 5})
    return _retriever

def preload_faiss_index():
    print("[INFO] Preloading FAISS index...")
    get_retriever()
    print("[INFO] FAISS index is ready.")

# =========================
# Brand Logic
# =========================
def get_brand_df() -> pd.DataFrame:
    global _brand_df
    if not _brand_df.empty: return _brand_df
    if not os.path.exists(BRAND_CSV): return pd.DataFrame()
    _brand_df = pd.read_csv(BRAND_CSV)
    _brand_df.columns = [_make_key(c) for c in _brand_df.columns]
    _brand_df["brand_key"] = _brand_df["brand_name"].apply(_make_key)
    return _brand_df

def respond_list_all_brands() -> str:
    df = get_brand_df()
    if df.empty: return "I don't have brand information right now."
    brands = sorted(df["brand_name"].dropna().unique())
    return "📊 **Yes! I track these brands:**\n" + ", ".join(brands)

def fuzzy_lookup_brand_candidates(user_text: str) -> List[str]:
    df = get_brand_df()
    if df.empty: return []
    key = _make_key(user_text)
    keys = df["brand_key"].tolist()
    matches = difflib.get_close_matches(key, keys, n=5, cutoff=0.6)
    return df[df["brand_key"].isin(matches)]["brand_name"].tolist()
    
# =========================
# Quiz Logic
# =========================
def start_quiz(session_data: dict, quiz_type: str) -> str:
    session_data.update({"current_quiz_session": None, "quiz_answers": []})
    try:
        with open(os.path.join(QUIZZES_DIR, f"{quiz_type}.json"), 'r') as f:
            quiz_data = json.load(f)
    except Exception:
        return "I'm sorry, my quiz materials are missing. I can still help with other questions!"
    
    session_data["current_quiz_session"] = {"quiz_data": quiz_data, "question_idx": 0, "quiz_type": quiz_type}
    return (f"**✨ {quiz_type.capitalize()} Routine Quiz**\n"
            f"I can definitely help! To personalize it, I'll ask a few quick questions.\n\n"
            f"**Ready to start?** Type **'start'** to begin!")

def get_next_quiz_question(session_data: dict) -> str:
    quiz_session = session_data["current_quiz_session"]
    idx = quiz_session["question_idx"]
    questions = quiz_session["quiz_data"]["questions"]
    if idx >= len(questions): return finish_quiz(session_data)
    q = questions[idx]
    options_text = "\n".join([f"{i+1}. {opt['text']}" for i, opt in enumerate(q["options"])])
    return f"**Q{q['id']}**: {q['question']}\n{options_text}"

def answer_quiz_option(session_data: dict, option_num: int) -> str:
    quiz_session = session_data["current_quiz_session"]
    idx, q = quiz_session["question_idx"], quiz_session["quiz_data"]["questions"][quiz_session["question_idx"]]
    if 1 <= option_num <= len(q["options"]):
        session_data["quiz_answers"].append(q["options"][option_num-1]["type"])
        quiz_session["question_idx"] += 1
        return get_next_quiz_question(session_data)
    return f"Invalid choice. Please select a number from 1 to {len(q['options'])}."

def finish_quiz(session_data: dict) -> str:
    result_type = Counter(session_data["quiz_answers"]).most_common(1)[0][0]
    quiz_type = session_data["current_quiz_session"]["quiz_type"]
    context = get_retriever().invoke(f"{result_type} {quiz_type} routine recommendation")
    prompt = f"You are OMI. Based on this context, recommend a routine for {quiz_type} type: {result_type}.\n\nContext: {''.join(d.page_content for d in context)}"
    recommendation = get_llm().invoke(prompt).content
    session_data.update({"current_quiz_session": None, "quiz_answers": [], "waiting_for_workbook_confirmation": True})
    return f"**Your {quiz_type} type:** *{result_type}*\n\n{recommendation}\n\n📘 Want me to send the *Omi Live Tactical Workbook*?"

# =========================
# Main Response Generator
# =========================
def get_rag_response(question: str, chat_session: Any) -> str:
    user_id = get_user_id(chat_session)
    session_data = _session_manager.get_session(user_id)
    raw_q = str(question).strip()
    if not raw_q: return "I don't know."
    cleaned_q = _clean_text(raw_q)
    print(f"[DEBUG] User '{user_id}' asked: '{raw_q}'")

    answer = ""
    # --- Highest Priority: Handle ongoing stateful conversations ---
    if session_data.get("current_quiz_session"):
        if cleaned_q.isdigit(): answer = answer_quiz_option(session_data, int(cleaned_q))
        elif cleaned_q in ["start", "yes", "begin"]: answer = get_next_quiz_question(session_data)
        else:
            session_data["current_quiz_session"] = None
            answer = "Quiz cancelled. How can I help with something else?"
    elif session_data.get("waiting_for_workbook_confirmation") and cleaned_q in {"yes", "sure", "okay", "ok", "yep", "yeah"}:
        session_data['waiting_for_workbook_confirmation'] = False
        answer = "Great! 🎉 You can download the workbook here: [**Download Workbook**](/get_workbook)"

    # --- Second Priority: Handle intents that change the state ---
    if not answer:
        routine_type = detect_routine_intent(raw_q)
        if routine_type: answer = start_quiz(session_data, routine_type)
        elif re.match(r"[^@]+@[^@]+\.[^@]+", raw_q) and not session_data.get('email'):
            session_data['email'] = raw_q
            answer = "🎉 **Thanks for signing up!** You'll hear from us soon. What can I help you next?"
        elif any(cleaned_q.startswith(g) for g in GREETINGS):
            greeting_index = session_data.get('greeting_index', 0)
            answer = f"{VARIED_GREETINGS[greeting_index % len(VARIED_GREETINGS)]} I also have a workbook — *Omi Live Tactical Workbook* 📘. Would you like me to send it?"
            session_data.update({'waiting_for_workbook_confirmation': True, 'greeting_index': greeting_index + 1})
        elif cleaned_q in ["how are you", "how are you doing"]:
            answer = "I'm doing great, thanks for asking! I'm ready to help you with your sustainability questions. What's on your mind?"
        elif re.search(r"\b(rank|list)\s*brands?\b", cleaned_q):
            answer = respond_list_all_brands()
        elif len(cleaned_q.split()) <= 4: # Fuzzy match for brand names
            candidates = fuzzy_lookup_brand_candidates(raw_q)
            if len(candidates) == 1:
                # This part can be enhanced to show the ranking details. For now, a simple confirmation.
                answer = f"Yes, {candidates[0]} is one of the brands I track. You can ask me to rank it specifically!"
            elif len(candidates) > 1:
                answer = "Did you mean one of these brands?\n- " + "\n- ".join(candidates)


    # --- Fallback: General RAG for everything else ---
    if not answer:
        session_data['waiting_for_workbook_confirmation'] = False # Reset state if user changes topic
        context_docs = get_retriever().invoke(raw_q)
        context = "\n\n".join(d.page_content for d in context_docs)
        if not context: answer = get_llm().invoke(raw_q).content
        else:
            prompt = QA_PROMPT_GENERAL.format(persona=SYSTEM_PERSONA, context=context, question=raw_q)
            answer = get_llm().invoke(prompt).content
        
        if cleaned_q in ["awesome", "perfect", "great", "thanks", "thank you"]:
            answer = "You're very welcome! Glad I could help. Is there anything else you're curious about?"

    # --- Final Step: Append Newsletter Prompt ---
    if session_data.get('response_count', 0) == 2 and not session_data.get('email'):
        answer += "\n\nWe're totally vibing! 💫 **Want to join our newsletter?** Just drop your email to sign up! 🌱"
        session_data['last_prompted_at'] = datetime.now().isoformat()

    session_data['response_count'] += 1
    _session_manager.update_session(user_id, session_data)
    return answer

# CLI test function
if __name__ == "__main__":
    preload_faiss_index()
    print("🤖 OMI Bot is ready!")
    cli_session = {"user_id": "cli_user"}
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']: break
        response = get_rag_response(user_input, cli_session)
        print(f"OMI: {response}")

