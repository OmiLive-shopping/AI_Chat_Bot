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
BRAND_CSV = os.path.join(DATA_DIR, "cleaned_brand_metrics.csv")
WORKBOOK_FILENAME = "Live_Sales_Tactical_Workbook.docx"
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

# Prompts
QA_PROMPT_GENERAL = PromptTemplate.from_template(
    """{persona}

ONLY use the information in the context. If the answer is not present, say "I don't know."

Context:
{context}

Question: {question}
Direct answer:"""
)

# Proactive Follow-up Suggestions
PROACTIVE_SUGGESTIONS = [
    "You can also ask me about eco-friendly laundry swaps.",
    "You can also ask me: 'Why are bees important?'",
    "You can also ask me: 'How do I choose a reef-safe sunscreen?'",
    "You can also ask me about the health benefits of bamboo.",
    "You can also ask me for tips on eating more sustainably.",
    "You can also ask me: 'What are the dangers in conventional tampons?'",
    "You can also ask me about superfood drinks for glowing skin."
]

# NEW: Workbook keywords for mid-conversation detection
WORKBOOK_KEYWORDS = [
    'tactical workbook', 'live sales workbook', 'workbook',
    'sales guide', 'creator guide', 'creator workbook', 'brand workbook'
]

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
            self.collection_ref.document(user_id).set(session_data, merge=True)
        except Exception as e:
            print(f"[ERROR] Failed to update session for user {user_id}: {e}")

    def _get_default_session(self, user_id: str) -> dict:
        return {
            "user_id": user_id, "response_count": 0, "current_quiz_session": None,
            "quiz_answers": [], "waiting_for_quiz_start": False,
            "waiting_for_rank_confirmation": False, "brand_to_rank": None,
            "waiting_for_workbook_confirmation": False,
            "waiting_for_user_classification": False, "user_type": None,
            "waiting_for_email": False, "offered_suggestions": [],
            "email_prompt_denied": False, "workbook_sent": False,
            "email_address": None,  # NEW: Track if email was provided
            "created_at": firestore.SERVER_TIMESTAMP if self.db else datetime.now().isoformat()
        }

_session_manager = UserSessionManager(db)

def get_user_id(session_info: Any) -> str:
    if isinstance(session_info, str): return session_info
    if isinstance(session_info, dict):
        if "user_id" not in session_info:
            session_info["user_id"] = os.urandom(16).hex()
        return session_info["user_id"]
    return os.urandom(16).hex()

# =========================
# Core Bot Logic
# =========================
def _clean_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", str(text).lower()).strip()

def _make_key(text: str) -> str:
    s = str(text).lower()
    s = re.sub(r"&", "and", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

AFFIRMATIONS = {"sounds good", "awesome", "perfect", "great", "okay", "ok", "yes", "please", "yes please", "start", "start quiz", "we can start", "we can", "sure", "yup", "yep", "of course", "ofcourse"}
NEGATIONS = {"no", "nope", "no thanks", "i don't", "no i don't", "nah"}

def is_affirmative_response(text: str) -> bool:
    cleaned = _clean_text(text)
    return any(cleaned == a or cleaned.startswith(a + " ") for a in AFFIRMATIONS)

def is_negative_response(text: str) -> bool:
    cleaned = _clean_text(text)
    return any(cleaned == n or cleaned.startswith(n + " ") for n in NEGATIONS)

def detect_routine_intent(question: str) -> Optional[str]:
    cleaned_q = _clean_text(question)
    quiz_trigger_keywords = [
        'routine', 'regimen', 'help with my', 'help with', 'quiz', 'take quiz', 'begin quiz',
        'skin', 'skin care', 'skincare', 'skin routine', 'face', 'face care', 'moisturizer',
        'serum', 'toner', 'acne', 'blemish', 'pimple', 'dry skin', 'oily skin', 'combination skin',
        'sensitive skin', 'wrinkle', 'aging', 'anti aging', 'sunscreen', 'spf',
        'hair', 'hair care', 'haircare', 'hair routine', 'hair regimen', 'shampoo', 'conditioner',
        'split ends', 'frizz', 'hair loss', 'dandruff', 'scalp', 'styling', 'curly', 'curly hair',
        'straight hair', 'color treated', 'chemically treated', 'leave in', 'hair mask', 'hair oil'
    ]
    hair_keywords = [
        'hair', 'shampoo', 'conditioner', 'curl', 'curly', 'straight', 'scalp', 'dandruff',
        'split ends', 'frizz', 'styling', 'haircare', 'hair care', 'hair loss', 'color treated',
        'leave in', 'hair mask', 'hair oil'
    ]
    skin_keywords = [
        'skin', 'skincare', 'skin care', 'face', 'moistur', 'serum', 'acne', 'blemish', 'pimple',
        'dry skin', 'oily skin', 'combination', 'sensitive', 'wrinkle', 'aging', 'sunscreen', 'spf'
    ]
    if any(trigger in cleaned_q for trigger in quiz_trigger_keywords):
        has_hair = any(word in cleaned_q for word in hair_keywords)
        has_skin = any(word in cleaned_q for word in skin_keywords)
        if has_hair and not has_skin: return 'hair'
        if has_skin and not has_hair: return 'skin'
        if 'skin' in cleaned_q or 'skincare' in cleaned_q or 'face' in cleaned_q: return 'skin'
        if 'hair' in cleaned_q or 'shampoo' in cleaned_q or 'conditioner' in cleaned_q: return 'hair'
        return 'skin'
    return None

_llm: Optional[ChatVertexAI] = None
_retriever = None
_brand_df: pd.DataFrame = pd.DataFrame()

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
    except Exception as e:
        print(f"[ERROR] Could not load FAISS index. Please run build_faiss.py. Error: {e}")
        return None
    _retriever = vect.as_retriever(search_kwargs={"k": 3})
    return _retriever

def preload_faiss_index():
    print("[INFO] Preloading FAISS index...")
    get_retriever()
    get_brand_df()
    print("[INFO] FAISS index and brand data are ready.")

def get_follow_up_suggestion(session_data: dict) -> str:
    offered = session_data.get("offered_suggestions", [])
    available = [s for s in PROACTIVE_SUGGESTIONS if s not in offered]
    if not available:
        return ""
    suggestion = random.choice(available)
    offered.append(suggestion)
    session_data["offered_suggestions"] = offered
    return f"\n_Psst... {suggestion}_"

# =========================
# Brand & Workbook Logic
# =========================
def _get_workbook_response() -> str:
    """
    Generates the markdown link for the workbook.
    The workbook file must be publicly accessible in a GCS bucket.
    """
    # !!! IMPORTANT !!!
    # Ensure 'Live_Sales_Tactical_Workbook.docx' is uploaded to your GCS bucket
    # and has public read permissions.
    workbook_url = f"https://storage.googleapis.com/{GOOGLE_CLOUD_PROJECT}/{WORKBOOK_FILENAME}"
    return (
        f"✅ Perfect! Your workbook is ready.\n\n"
        f"You can download it here: [**Click to download {WORKBOOK_FILENAME}**]({workbook_url})"
    )

def get_brand_df() -> pd.DataFrame:
    global _brand_df
    if not _brand_df.empty: return _brand_df
    cleaned_csv_path = os.path.join(DATA_DIR, "cleaned_brand_metrics.csv")
    if not os.path.exists(cleaned_csv_path): return pd.DataFrame()
    try:
        df = pd.read_csv(cleaned_csv_path)
        df["brand_key"] = df["brand_name"].apply(_make_key)
        _brand_df = df
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load cleaned brand metrics: {e}")
        return pd.DataFrame()

def get_brand_ranking_single(brand_name: str) -> str:
    df = get_brand_df()
    if df.empty: return "I don't have brand ranking information available right now."
    key = _make_key(brand_name)
    row = df[df["brand_key"] == key]
    if row.empty: return f"I couldn't find a ranking for '{brand_name}'."
    row = row.iloc[0]
    score = row.get('final_score', 'N/A')
    breakdown_cols = [c for c in df.columns if c not in ['brand_name', 'brand_key', 'final_score']]
    breakdown = [f"- {col.replace('_', ' ').title()}: {row[col]}" for col in breakdown_cols if col in row and pd.notna(row[col])]
    response = f"🌍 **{row['brand_name']}** — Sustainability score **{score} / 30**."
    if breakdown:
        response += "\n" + "\n".join(breakdown)
    return response

def fuzzy_lookup_brand_candidates(user_text: str) -> List[str]:
    if _clean_text(user_text) in NEGATIONS: return []
    df = get_brand_df()
    if df.empty: return []
    key = _clean_text(user_text)
    if not key: return []
    keys = df["brand_key"].tolist()
    matches = difflib.get_close_matches(key, keys, n=3, cutoff=0.8)
    if not matches:
        for b_key in keys:
            if key in b_key.split():
                if b_key not in matches:
                    matches.append(b_key)
                if len(matches) >= 3: break
    return df[df["brand_key"].isin(matches)]["brand_name"].tolist()

def respond_with_brand_info() -> str:
    df = get_brand_df()
    if df.empty or 'final_score' not in df.columns:
        return "I don't have brand ranking information right now."
    top_ranked = df.sort_values(by='final_score', ascending=False).head(5)
    response_lines = ["Sure! Here are some of the top eco-friendly brands we've ranked on a scale of 30:\n"]
    for i, row in enumerate(top_ranked.itertuples(), 1):
        response_lines.append(f"{i}. **{row.brand_name}** (Score: {row.final_score})")
    all_brands = sorted(df["brand_name"].dropna().unique())
    response_lines.append("\nI also track: " + ", ".join(all_brands))
    return "\n".join(response_lines)

# =========================
# Quiz Logic
# =========================
def offer_quiz(session_data: dict, quiz_type: str) -> str:
    session_data["waiting_for_quiz_start"] = True
    session_data["quiz_type_pending"] = quiz_type
    return "Of course! To recommend the best products, I can ask you a few quick questions to identify your hair & skin type. Shall we start the quiz?"

def start_quiz(session_data: dict, quiz_type: str) -> str:
    session_data.update({"quiz_answers": [], "waiting_for_quiz_start": False})
    try:
        with open(os.path.join(QUIZZES_DIR, f"{quiz_type}.json"), 'r') as f:
            quiz_data = json.load(f)
    except Exception:
        return "I'm sorry, my quiz materials are missing at the moment."
    session_data["current_quiz_session"] = {"quiz_data": quiz_data, "question_idx": 0}
    return get_next_quiz_question(session_data)

def get_next_quiz_question(session_data: dict) -> str:
    quiz_session = session_data["current_quiz_session"]
    idx = quiz_session["question_idx"]
    questions = quiz_session["quiz_data"]["questions"]
    if idx >= len(questions): return finish_quiz(session_data)
    q = questions[idx]
    options_text = "\n".join([f"{i+1}. {opt['text']}" for i, opt in enumerate(q["options"])])
    return f"**Question {q['id']}**: {q['question']}\n{options_text}"

def answer_quiz_option(session_data: dict, option_num: int) -> str:
    quiz_session = session_data["current_quiz_session"]
    q = quiz_session["quiz_data"]["questions"][quiz_session["question_idx"]]
    if 1 <= option_num <= len(q["options"]):
        session_data["quiz_answers"].append(q["options"][option_num-1]["answer"])
        quiz_session["question_idx"] += 1
        return get_next_quiz_question(session_data)
    return f"Invalid choice. Please select a number from 1 to {len(q['options'])}."

def finish_quiz(session_data: dict) -> str:
    try:
        quiz_data = session_data["current_quiz_session"]["quiz_data"]
        most_common_answer = Counter(session_data["quiz_answers"]).most_common(1)[0][0]
        result_type = quiz_data["results_logic"][most_common_answer]
        routine = quiz_data["routines"][result_type]
        session_data.update({"current_quiz_session": None, "quiz_answers": []})
        return f"Based on your answers, it looks like you have **{result_type}**!\n\nHere’s a simple routine for you:\n{routine}"
    except Exception as e:
        print(f"[ERROR] in finish_quiz: {e}")
        session_data.update({"current_quiz_session": None, "quiz_answers": []})
        return "I had a little trouble generating your routine."

# =========================
# Main Response Generator
# =========================
def get_rag_response(question: str, user_id: str) -> str:
    session_data = _session_manager.get_session(user_id)
    raw_q = str(question).strip()
    if not raw_q: return "I don't know."

    answer = ""
    session_data['response_count'] = session_data.get('response_count', 0) + 1
    add_suggestion = True

    is_affirmative = is_affirmative_response(raw_q)
    is_negative = is_negative_response(raw_q)

    # --- BLOCK A: Handle special triggers and stateful responses FIRST ---
    if raw_q == "__GET_ONBOARDING__":
        session_data['waiting_for_user_classification'] = True
        _session_manager.update_session(user_id, session_data)
        return ""

    elif session_data.get("waiting_for_user_classification"):
        cleaned_q = _clean_text(raw_q)
        session_data['waiting_for_user_classification'] = False
        if 'eco shopper' in cleaned_q or 'ecoshopper' in cleaned_q:
            session_data['user_type'] = 'eco_shopper'
            answer = ("🌱 Welcome to Omi Live, your eco living girlie! Ready to explore REAL eco-friendly brands?\n\n"
                      "Here’s what you’ll get as part of the Omi Fam:\n"
                      "• Our AI Green Rating System to shop transparently\n"
                      "• Exclusive offers & discounts\n"
                      "• Direct interaction with brand owners\n"
                      "• A front-row seat to watch eco-friendly brands grow")
        elif 'creator' in cleaned_q:
            session_data['user_type'] = 'creator'
            answer = ("🎥 Hey there! Are you a creator interested in live shopping?\n\n"
                      "With Omi Live, you can monetize your influence through:\n"
                      "• Free product samples\n"
                      "• Sales commissions\n"
                      "• Flat fee partnerships\n"
                      "• Expanding your reach with eco-conscious buyers")
            session_data['waiting_for_workbook_confirmation'] = True
            answer += "\nWe’ve built a Live Sales Workbook for Creators — it shows you how to maximize earnings and grow with us. Want it?"
        elif 'brand owner' in cleaned_q or 'brandowner' in cleaned_q:
            session_data['user_type'] = 'brand_owner'
            answer = ("👋 Hi! Welcome to Omi Live — the AI-powered retail tech for eco-friendly brands. Are you a brand owner looking to grow sales?\n\n"
                      "We help brands like yours achieve 20% sales conversion through:\n"
                      "• A loyal eco-conscious community\n"
                      "• Smart product listing & discovery tools\n"
                      "• Live storytelling that builds trust")
            session_data['waiting_for_workbook_confirmation'] = True
            answer += "\nWould you like our Live Sales Workbook? It’s packed with strategies to boost sales. Want it?"
        else:
            session_data['waiting_for_user_classification'] = True
            answer = "Please choose a valid option by clicking one of the buttons below."
        add_suggestion = False
    
    # MODIFIED: Handle email submission to deliver workbook
    elif re.match(r"[^@]+@[^@]+\.[^@]+", raw_q):
        session_data['waiting_for_email'] = False
        session_data['email_address'] = raw_q  # Save email to session
        user_type = session_data.get('user_type')

        if user_type in ['creator', 'brand_owner']:
            session_data['workbook_sent'] = True
            answer = _get_workbook_response()  # Send workbook link
        else:  # Eco Shopper
            answer = "You’re all set! We’ll share tips, community insights, and opportunities to feature your brand on Omi Live.\n\nWe’re having your personalized matches brewing. Stay tuned on Omi updates!"
        add_suggestion = False

    elif session_data.get("current_quiz_session"):
        if _clean_text(raw_q).isdigit():
            answer = answer_quiz_option(session_data, int(_clean_text(raw_q)))
        else:
            session_data["current_quiz_session"] = None
            answer = "Quiz cancelled. How can I help?"
        add_suggestion = False

    elif session_data.get("waiting_for_quiz_start"):
        if is_affirmative:
            quiz_type = session_data.get('quiz_type_pending', 'skin')
            answer = start_quiz(session_data, quiz_type)
        else:
            session_data["waiting_for_quiz_start"] = False
            answer = "No problem — if you change your mind, I can start the quiz anytime."
        add_suggestion = False

    elif session_data.get("waiting_for_workbook_confirmation"):
        if is_affirmative:
            session_data['waiting_for_email'] = True
            if session_data.get('user_type') == 'creator':
                answer = "Awesome! Share your email so we can send the workbook + early invites to campaigns."
            else:  # Brand Owner
                answer = "Great! Drop your email so we can send you the workbook + early access to our tools."
        elif is_negative:
            answer = "No problem! What else can I help you with today?"
        session_data['waiting_for_workbook_confirmation'] = False
        add_suggestion = False

    elif session_data.get("waiting_for_rank_confirmation"):
        if is_affirmative:
            brand_to_rank = session_data.get("brand_to_rank")
            if brand_to_rank: answer = get_brand_ranking_single(brand_to_rank)
        elif is_negative:
            answer = "Got it, no problem! How else can I help?"
        session_data.update({"waiting_for_rank_confirmation": False, "brand_to_rank": None})
        add_suggestion = False

    elif is_negative_response(raw_q) and session_data.get('waiting_for_email'):
        session_data['email_prompt_denied'] = True
        session_data['waiting_for_email'] = False
        answer = "👍 No worries! We'll keep chatting here."
        add_suggestion = False

    # --- BLOCK B: If no stateful response, handle new query ---
    if not answer:
        add_suggestion = True
        session_data.update({
            'waiting_for_quiz_start': False, 'quiz_type_pending': None,
            'waiting_for_rank_confirmation': False, 'brand_to_rank': None
        })

        cleaned_q = _clean_text(raw_q)

        # NEW: High-priority check for workbook requests
        if any(keyword in cleaned_q for keyword in WORKBOOK_KEYWORDS):
            user_type = session_data.get('user_type')
            if user_type in ['creator', 'brand_owner']:
                if session_data.get('email_address'):
                    answer = _get_workbook_response()  # Email on file, send link
                else:
                    session_data['waiting_for_email'] = True  # No email, ask for it
                    answer = "Of course! To send you the workbook, what's your email?"
            else:  # Eco-shopper or unclassified
                answer = "The Live Sales Workbook is designed for Creators and Brand Owners. Let me know if you'd like to learn more about those roles!"
            add_suggestion = False

        if not answer:
            if "rank brand" in cleaned_q or "brand ranking" in cleaned_q or "suggest brand" in cleaned_q or "list brand" in cleaned_q:
                answer = respond_with_brand_info()
            else:
                quiz_type = detect_routine_intent(raw_q)
                if quiz_type:
                    answer = offer_quiz(session_data, quiz_type)
                else:
                    candidates = fuzzy_lookup_brand_candidates(raw_q)
                    if len(candidates) == 1:
                        brand_name = candidates[0]
                        session_data["waiting_for_rank_confirmation"] = True
                        session_data["brand_to_rank"] = brand_name
                        answer = f"I found the brand '{brand_name}'. Would you like me to provide its sustainability ranking?"
                        add_suggestion = False
                    elif len(candidates) > 1:
                        answer = "Did you mean one of these brands? You can ask me to 'rank' one.\n- " + "\n- ".join(candidates)
                        add_suggestion = False

                    if not answer:
                        retriever = get_retriever()
                        if not retriever:
                            answer = "My knowledge base is currently unavailable. Please try again later."
                        else:
                            context_docs = retriever.invoke(raw_q)
                            context = "\n\n".join(d.page_content for d in context_docs)
                            if not context.strip():
                                answer = "I'm not sure how to answer that. Could you try rephrasing?"
                            else:
                                prompt = QA_PROMPT_GENERAL.format(persona=SYSTEM_PERSONA, context=context, question=raw_q)
                                answer = get_llm().invoke(prompt).content

    # --- Final Step: Append newsletter or suggestion ---
    # This logic remains untouched, as it handles the general newsletter prompt, not the workbook delivery.

    should_prompt_email = False
    user_type = session_data.get('user_type')
    response_count = session_data.get('response_count', 0)

    if not session_data.get('waiting_for_email') and not session_data.get('email_prompt_denied'):
        if user_type == 'eco_shopper' and response_count == 5:
            should_prompt_email = True
        # Don't prompt creators/brands for a general newsletter if they were already in a workbook flow
        elif user_type in ['creator', 'brand_owner'] and not session_data.get('workbook_sent') and not session_data.get('waiting_for_workbook_confirmation') and response_count == 3:
            should_prompt_email = True

    if should_prompt_email:
        session_data['waiting_for_email'] = True
        answer += ("\n💫 We're totally vibing! I'd love to keep this going - want to join our exclusive newsletter? "
                   "What's your email? 🌱")
        add_suggestion = False

    if add_suggestion:
        follow = get_follow_up_suggestion(session_data)
        if follow and follow.strip() not in answer:
            answer += follow

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
        response = get_rag_response(user_input, cli_session['user_id'])
        print(f"OMI: {response}")