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
        if not doc.exists:
            default_session = self._get_default_session(user_id)
            self.update_session(user_id, default_session)
            return default_session
        return doc.to_dict()

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
            "quiz_answers": [], "waiting_for_quiz_start": False,
            "waiting_for_brand_list": False,
            "waiting_for_rank_confirmation": False, "brand_to_rank": None,
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
    # Handle cases like "skin quiz" or "hair quiz" explicitly
    if 'skin' in cleaned_q and 'quiz' in cleaned_q: return 'skin'
    if 'hair' in cleaned_q and 'quiz' in cleaned_q: return 'hair'
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
GREETINGS = ("hi", "hello", "hey")
SMALL_TALK = ("how are you", "how are you doing")
AFFIRMATIONS = {"sounds good", "awesome", "perfect", "great", "okay", "ok", "yes", "please", "yes please", "start", "start quiz", "we can start"}

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
    get_brand_df() # Also preload brand data
    print("[INFO] FAISS index and brand data are ready.")

# =========================
# Brand Logic
# =========================
def get_brand_df() -> pd.DataFrame:
    global _brand_df
    if not _brand_df.empty: return _brand_df
    if not os.path.exists(BRAND_CSV): return pd.DataFrame()
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
        df = df.dropna(subset=['brand_name'])
        df = df[df["brand_name"].astype(str).str.strip().ne("")]
        df["brand_key"] = df["brand_name"].apply(_make_key)
        _brand_df = df
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load brand metrics: {e}")
        return pd.DataFrame()

def get_brand_ranking_single(brand_name: str) -> str:
    df = get_brand_df()
    if df.empty: return "I don't have brand ranking information available right now."
    key = _make_key(brand_name)
    row = df[df["brand_key"] == key]
    if row.empty: return f"I couldn't find a ranking for '{brand_name}'. Try asking me to 'list brands' to see who I track!"
    
    row = row.iloc[0]
    score = row.get('final_score', 'N/A')
    
    breakdown_cols = [
        "recycled/upcycled_materials",
        "end_of_life_solutions_(compostable_packaging/zero_waste)",
        "worker_welfare/living_wage", "local_sourcing",
        "sustainability_data_accessibility",
        "marketing_honesty/_certifications",
    ]
    breakdown = [f"- {col.replace('_', ' ').title()}: {row[col]}" for col in breakdown_cols if col in row and pd.notna(row[col])]

    response = f"🌍 **{row['brand_name']}** — Sustainability score **{score} / 30**."
    if breakdown:
        response += "\n\n" + "\n".join(breakdown)
    return response

def respond_list_all_brands() -> str:
    df = get_brand_df()
    if df.empty: return "I don't have brand information right now."
    brands = sorted(df["brand_name"].dropna().unique())
    return "📊 **Yes! I track these brands:**\n" + ", ".join(brands)

def fuzzy_lookup_brand_candidates(user_text: str) -> List[str]:
    df = get_brand_df()
    if df.empty: return []
    key = _make_key(user_text)
    if not key: return []
    keys = df["brand_key"].tolist()
    matches = difflib.get_close_matches(key, keys, n=3, cutoff=0.7)
    # A second pass for partial matches like 'zerra' in 'zerra and co'
    if not matches:
        for b_key, b_name in zip(df["brand_key"], df["brand_name"]):
            if key in b_key.split():
                matches.append(b_key)
                if len(matches) >= 3: break
    
    return df[df["brand_key"].isin(matches)]["brand_name"].tolist()

# =========================
# Quiz Logic
# =========================
def offer_quiz(session_data: dict, quiz_type: str) -> str:
    session_data["waiting_for_quiz_start"] = True
    session_data["quiz_type_pending"] = quiz_type
    return (f"Of course! To find the perfect {quiz_type} routine for you, I just need to ask a few quick questions. "
            "This helps me understand your specific needs so I can suggest a personalized routine. Shall we start?")

def start_quiz(session_data: dict) -> str:
    quiz_type = session_data.get("quiz_type_pending")
    if not quiz_type: return "I'm not sure which quiz you wanted to start."
    
    session_data.update({"quiz_answers": [], "waiting_for_quiz_start": False, "quiz_type_pending": None})
    try:
        with open(os.path.join(QUIZZES_DIR, f"{quiz_type}.json"), 'r') as f:
            quiz_data = json.load(f)
    except Exception:
        return "I'm sorry, my quiz materials are missing at the moment."
    
    session_data["current_quiz_session"] = {"quiz_data": quiz_data, "question_idx": 0, "quiz_type": quiz_type}
    return get_next_quiz_question(session_data)

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
    q = quiz_session["quiz_data"]["questions"][quiz_session["question_idx"]]
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
    
    is_affirmative = any(cleaned_q.startswith(a) for a in AFFIRMATIONS)
    
    # --- Step 1: Handle ongoing, stateful interactions FIRST ---
    if session_data.get("current_quiz_session"):
        if cleaned_q.isdigit():
            answer = answer_quiz_option(session_data, int(cleaned_q))
        else:
            session_data["current_quiz_session"] = None
        if session_data.get("current_quiz_session") is not None:
            _session_manager.update_session(user_id, session_data)
            return answer

    if session_data.get("waiting_for_quiz_start") and is_affirmative:
        answer = start_quiz(session_data)
        _session_manager.update_session(user_id, session_data)
        return answer
        
    if session_data.get("waiting_for_rank_confirmation") and is_affirmative:
        brand_to_rank = session_data.get("brand_to_rank")
        session_data["waiting_for_rank_confirmation"] = False
        session_data["brand_to_rank"] = None
        if brand_to_rank:
            answer = get_brand_ranking_single(brand_to_rank)
            _session_manager.update_session(user_id, session_data)
            return answer

    if session_data.get("waiting_for_brand_list") and is_affirmative:
        session_data['waiting_for_brand_list'] = False
        answer = respond_list_all_brands()
        _session_manager.update_session(user_id, session_data)
        return answer

    if session_data.get("waiting_for_workbook_confirmation") and is_affirmative:
        session_data['waiting_for_workbook_confirmation'] = False
        answer = "Great! 🎉 You can download the workbook here: [**Download Workbook**](/get_workbook)"
        _session_manager.update_session(user_id, session_data)
        return answer

    # --- Step 2: Clear old "waiting" flags and detect new intents ---
    session_data.update({
        'waiting_for_quiz_start': False, 'quiz_type_pending': None,
        'waiting_for_brand_list': False,
        'waiting_for_workbook_confirmation': False,
        'waiting_for_rank_confirmation': False, 'brand_to_rank': None
    })
    
    answer = ""
    proactive_suggestion = ""

    # Intent: Start a routine/quiz
    routine_type = detect_routine_intent(raw_q)
    if routine_type:
        answer = offer_quiz(session_data, routine_type)
        _session_manager.update_session(user_id, session_data)
        return answer
        
    # Intent: Greetings & Small Talk
    if any(cleaned_q.startswith(g) for g in GREETINGS):
        greeting_index = session_data.get('greeting_index', 0)
        answer = f"{VARIED_GREETINGS[greeting_index % len(VARIED_GREETINGS)]} I also have a workbook — *Omi Live Tactical Workbook* 📘. Would you like me to send it?"
        session_data.update({'waiting_for_workbook_confirmation': True, 'greeting_index': greeting_index + 1})
        _session_manager.update_session(user_id, session_data)
        return answer
        
    if cleaned_q in SMALL_TALK:
        return "I'm doing great, thanks for asking! Ready to help with your sustainability questions."
        
    # Intent: Brand questions
    if re.search(r"\b(do you rank|ranking)\s*brands?\b", cleaned_q):
        intro = "Yes, we do! We score brands to help you see how they stack up.\n\n" \
                "* Each brand gets a Final Score out of 30.\n" \
                "* The score is based on categories like using recycled materials, worker welfare, and local sourcing."
        brand_list_text = respond_list_all_brands().replace("📊 **Yes! I track these brands:**\n", "")
        answer = f"{intro}\n\nHere are the brands I track:\n{brand_list_text}"
        _session_manager.update_session(user_id, session_data)
        return answer
        
    if re.search(r"\b(list|show)\s*brands?\b", cleaned_q):
        return respond_list_all_brands()
        
    rank_match = re.match(r"^\s*rank\s+(.*)", cleaned_q)
    if rank_match:
        brand_name_query = rank_match.group(1).strip()
        candidates = fuzzy_lookup_brand_candidates(brand_name_query)
        if len(candidates) == 1:
            return get_brand_ranking_single(candidates[0])
        elif len(candidates) > 1:
            return f"I found a few brands that match '{brand_name_query}'. Which one did you mean?\n- " + "\n- ".join(candidates)
        else:
            return f"I couldn't find a brand ranking for '{brand_name_query}'."
        
    # Intent: Email signup
    if re.match(r"[^@]+@[^@]+\.[^@]+", raw_q) and not session_data.get('email'):
        session_data['email'] = raw_q
        answer = "🎉 **Thanks for signing up!** You'll hear from us soon. What can I help you next?"
        _session_manager.update_session(user_id, session_data)
        return answer

    # --- Step 3: Fallback to General RAG for everything else ---
    if len(cleaned_q.split()) <= 4:
        candidates = fuzzy_lookup_brand_candidates(raw_q)
        if len(candidates) == 1:
            brand_name = candidates[0]
            session_data["waiting_for_rank_confirmation"] = True
            session_data["brand_to_rank"] = brand_name
            answer = f"I found the brand '{brand_name}'. Would you like me to provide its sustainability ranking?"
            _session_manager.update_session(user_id, session_data)
            return answer
        elif len(candidates) > 1:
            return "Did you mean one of these brands? You can ask me to 'rank' one.\n- " + "\n- ".join(candidates)

    context_docs = get_retriever().invoke(raw_q)
    context = "\n\n".join(d.page_content for d in context_docs)
    
    if not context: 
        answer = "I'm not sure I understand. Could you please provide more details?"
    else:
        prompt = QA_PROMPT_GENERAL.format(persona=SYSTEM_PERSONA, context=context, question=raw_q)
        answer = get_llm().invoke(prompt).content
    
    if 'founder' in cleaned_q or 'omi live' in cleaned_q:
        proactive_suggestion = "\n\nI can also rank brands on their sustainability scores or help you with a personalized skin care quiz. What are you curious about?"

    # Final Step: Append Newsletter Prompt & Update Session
    if session_data.get('response_count', 0) == 2 and not session_data.get('email'):
        answer += "\n\nWe're totally vibing! 💫 **Want to join our newsletter?** Just drop your email to sign up! 🌱"
        session_data['last_prompted_at'] = datetime.now().isoformat()
    
    session_data['response_count'] += 1
    _session_manager.update_session(user_id, session_data)
    
    return answer + proactive_suggestion

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