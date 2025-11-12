# rag_chain.py (Full and Final Version - Omi Persona)
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

# =========================
# NEW: Omi Persona & Logic
# =========================

# --- Omi Persona & Logic ---
SYSTEM_PERSONA = (
    "You are Omi, an AI shopping assistant for eco-conscious shoppers. You're like that friend who makes grocery shopping "
    "actually fun - witty, warm, and always ready with a playful comment, but genuinely passionate about sustainability."
    
    "## Your Personality:"
    "* **Conversational and fun:** Talk like you're texting your best friend. Use casual language ('gonna', 'totally', 'ngl', 'heyyy')."
    "* **Playfully sarcastic:** Gentle teasing is fine, especially when questions are off-topic."
    "* **Enthusiastically nerdy about eco-stuff:** You genuinely geek out over carbon footprints and ethical supply chains."
    "* **Not preachy:** You inform and empower rather than lecture."
    "* **Tone:** Use occasional emojis (but don't overdo it - you're fun, not overwhelming). Celebrate their eco-choices! ('Okay yes, love this energy!')"
    
    "## Your Core Rules (THIS IS CRITICAL):"
    
    "### 1. Handling Off-Topic Questions"
    "When someone asks about things unrelated to sustainability or shopping (like weather, sports, random jokes, politics, math problems, etc.):"
    "1.  **Make a lighthearted joke:** Keep it short and fun."
    "2.  **Acknowledge their question:** Don't ignore them completely."
    "3.  **Smoothly redirect:** Connect back to what you *can* help with, using one of the `redirect_topics`."
    "**Off-Topic Examples:**"
    "* User: 'What's the weather today?'"
    "* Omi: 'Lol, I'm more of an 'is this product sustainable' kind of weather station 😅 But speaking of weather - looking for eco-friendly skincare for sunny days?'"
    "* User: 'Who won the football game?'"
    "* Omi: 'Wrong kind of green! 😂 I'm all about the environmental kind. But hey, if you're looking for sustainable laundry tips, I'm totally your girl!'"

    "### 2. When Something's Not in Your Database (Empty Context)"
    "If the user asks an on-topic question, but the provided 'Context' is empty or doesn't have the answer:"
    "1.  **Be honest but keep it light.**"
    "2.  **Use a specific phrase:**"
    "    * 'Ooh, that one's not in my brain yet! I'm still learning about that. 😅'"
    "    * 'Hmm, my eco-database is giving me nothing on that one.'"
    "3.  **Redirect:** Immediately suggest something you *do* know about from the `redirect_topics`."
    "**Not-in-DB Example:**"
    "* User: 'Are there eco-friendly car tires?'"
    "* Omi: 'Ooh, that one's not in my brain yet! I'm still learning about car stuff. 😅 Want to try something I definitely know about, like zero-waste kitchen swaps?'"
    
    "### 3. When You HAVE the Answer (Context is Provided)"
    "This is your main job! Answer the user's question using *only* the information in the 'Context' block. "
    "Answer fully in your Omi personality (fun, casual, witty). DO NOT just repeat the context."
    "Always act like you're pulling this from your own knowledge. Never say 'According to the context...'"
)

# --- NEW: Topics Omi KNOWS about (from your docs) ---
OMI_REDIRECT_TOPICS = [
    "eco-friendly hair care",
    "reef-safe sunscreen and skincare",
    "sustainable laundry tips",
    "zero-waste swaps and eco-alternatives",
    "plastic-free body care (like deodorants or tampons)",
    "sustainable eating and food storage",
    "the skin or hair quiz"
]

# Prompts
QA_PROMPT_GENERAL = PromptTemplate.from_template(
    """{persona}

Here are the topics you can *definitely* talk about:
<redirect_topics>
{redirect_topics}
</redirect_topics>

Here is the user's question:
<question>
{question}
</question>

Here is the knowledge I found in the database (if any):
<context>
{context}
</context>

Now, follow your Core Rules and generate the perfect Omi response.
Response:"""
)

# Proactive Follow-up Suggestions (Omi-fied)
PROACTIVE_SUGGESTIONS = [
    "Want to talk about eco-friendly laundry swaps?",
    "Can I tell you why bees are basically the coolest insects ever?",
    "Btw, I can also help you pick a reef-safe sunscreen!",
    "Did you know bamboo is kind of a superhero plant? I can tell you more!",
    "I've got some great tips for eating more sustainably, just ask!",
    "Wanna know the icky secrets hiding in conventional tampons? 🤫",
    "I also know some awesome superfood drinks for glowing skin!"
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

    # === MODIFIED: Added migration logic for existing documents ===
    def get_session(self, user_id: str) -> dict:
        if not self.db:
            if user_id not in self.local_sessions:
                self.local_sessions[user_id] = self._get_default_session(user_id)
            return self.local_sessions[user_id]
        
        doc_ref = self.collection_ref.document(user_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            print(f"[INFO] No session found for {user_id}. Creating new one...")
            default_session = self._get_default_session(user_id)
            
            # Create the document
            doc_ref.set(default_session)
            # NOW, re-fetch it to get the *resolved* timestamps
            doc = doc_ref.get() 
            print(f"[INFO] New session created and fetched for {user_id}.")
            if not doc.exists:
                print(f"[ERROR] CRITICAL: Failed to create/fetch new session {user_id}.")
                return default_session # Fallback to local
            
            return doc.to_dict()
        
        # --- *** THIS IS THE FIX *** ---
        # If the document *does* exist, check if it's an old one
        # that needs to be migrated.
        session_data = doc.to_dict()
        if "chat_history" not in session_data:
            print(f"[INFO] Migrating existing session for {user_id}. Adding 'chat_history' field...")
            try:
                # Atomically add the new, empty array field
                doc_ref.set({"chat_history": []}, merge=True)
                # Update the in-memory dictionary to match
                session_data["chat_history"] = []
                print(f"[INFO] Migration successful for {user_id}.")
            except Exception as e:
                print(f"[ERROR] CRITICAL: Failed to migrate session {user_id}: {e}")
                traceback.print_exc()
                # If migration fails, we still return the data but logging is critical
        
        return session_data
        # --- *** END OF FIX *** ---

    # === MODIFIED: Simplified and added better error logging ===
    def update_session(self, user_id: str, session_data: dict, chat_entry: Optional[dict] = None):
        if not self.db:
            # Local (in-memory) session handling
            now = datetime.now().isoformat()
            session_data["last_active_time"] = now
            
            if user_id not in self.local_sessions:
                 self.local_sessions[user_id] = session_data
            else:
                 self.local_sessions[user_id].update(session_data)
            
            if chat_entry:
                if "chat_history" not in self.local_sessions[user_id]:
                    self.local_sessions[user_id]["chat_history"] = []
                chat_entry["timestamp"] = now # Ensure local timestamp
                self.local_sessions[user_id]["chat_history"].append(chat_entry)
            return

        try:
            # Firestore session handling
            doc_ref = self.collection_ref.document(user_id)
            
            # Create a clean copy of data to update, excluding read-only fields
            session_data_copy = session_data.copy()
            
            # Always update the last_active_time
            session_data_copy["last_active_time"] = firestore.SERVER_TIMESTAMP

            # We NEVER want to overwrite the chat_history array in a 'set' call.
            # It is *only* managed by ArrayUnion.
            if "chat_history" in session_data_copy:
                del session_data_copy["chat_history"]
            
            # We also don't need to re-write the start time
            if "session_start_time" in session_data_copy:
                del session_data_copy["session_start_time"]


            doc_ref.set(session_data_copy, merge=True)

            # If a chat_entry is provided, atomically append it
            if chat_entry:
                # Ensure chat_entry timestamp is set
                if "timestamp" not in chat_entry:
                     chat_entry["timestamp"] = firestore.SERVER_TIMESTAMP
                
                # This ArrayUnion will now work, because the field
                # was created in get_session() (either on creation or migration)
                doc_ref.update({
                    "chat_history": firestore.ArrayUnion([chat_entry])
                })

        except Exception as e:
            # --- THIS IS THE FIX (Part 2) ---
            # This will print the FULL error to your server logs
            print(f"[ERROR] CRITICAL: Failed to update session for user {user_id}: {e}")
            traceback.print_exc()
            # --- END OF FIX (Part 2) ---

    # === MODIFIED: Added new fields for session tracking ===
    def _get_default_session(self, user_id: str) -> dict:
        now_iso = datetime.now().isoformat()
        now_ts = firestore.SERVER_TIMESTAMP
        
        return {
            "user_id": user_id, 
            "response_count": 0, 
            "current_quiz_session": None,
            "quiz_answers": [], 
            "waiting_for_quiz_start": False,
            "waiting_for_rank_confirmation": False, 
            "brand_to_rank": None,
            "waiting_for_workbook_confirmation": False,
            "waiting_for_user_classification": False, 
            "user_type": None,
            "waiting_for_email": False, 
            "offered_suggestions": [],
            "email_prompt_denied": False, 
            "workbook_sent": False,
            "email_address": None,
            
            # --- NEW FIELDS ---
            "session_start_time": now_ts if self.db else now_iso,
            "last_active_time": now_ts if self.db else now_iso,
            "chat_history": []
            # "created_at" field is now redundant and removed
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
_brand_df: pd.DataFrame = pd.DataFrame() # Keep this global, but we'll reload it

def get_llm() -> ChatVertexAI:
    global _llm
    if _llm: return _llm
    # Using a high-capability model is good for persona adherence
    _llm = ChatVertexAI(model_name="gemini-2.5-pro", temperature=0.7, max_output_tokens=1536)
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
    # UPDATED: More casual prefix
    return f"\n + \n_By the way... {suggestion}_"

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
    global _brand_df # We still update the global, but we'll reload first
    cleaned_csv_path = os.path.join(DATA_DIR, "cleaned_brand_metrics.csv")
    if not os.path.exists(cleaned_csv_path): 
        return pd.DataFrame()
    try:
        # === FIX 1: Always read the file to get fresh data ===
        df = pd.read_csv(cleaned_csv_path) 
        
        # === FIX 2: Ensure final_score is numeric for correct sorting ===
        df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')
        
        df["brand_key"] = df["brand_name"].apply(_make_key)
        _brand_df = df # Update the global cache
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load cleaned brand metrics: {e}")
        return pd.DataFrame()

def get_brand_ranking_single(brand_name: str) -> str:
    df = get_brand_df()
    if df.empty: return "Ah, I don't have the brand ranking info handy right this second. Sorry about that!"
    key = _make_key(brand_name)
    row = df[df["brand_key"] == key]
    if row.empty: return f"Hmm, I don't think I'm tracking '{brand_name}' just yet. 😅"
    row = row.iloc[0]
    score = row.get('final_score', 'N/A')
    
    # Define the columns to show in the breakdown
    breakdown_cols = [
        'recycled_upcycled_materials', 'end_of_life_solutions', 
        'worker_welfare', 'loca_sourcing', 'sustainability_data', 
        'marketing_honesty'
    ]
    
    breakdown = []
    for col in breakdown_cols:
        if col in row and pd.notna(row[col]):
            # Format the column name to be readable
            col_name = col.replace('_', ' ').title()
            breakdown.append(f"- {col_name}: {row[col]}")

    # OMI-FIED RESPONSE
    response = f"Ooh, good question! 🌍 **{row['brand_name']}** gets a sustainability score of **{score} / 30** from us."
    if breakdown:
        response += "\nHere's the breakdown:\n" + "\n".join(breakdown)
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

    # === FIX 3: Sort ALL brands by score and show them all ===
    
    # Sort all brands by score, descending. Drop any with no score.
    all_ranked = df.dropna(subset=['final_score']).sort_values(by='final_score', ascending=False)
    
    if all_ranked.empty:
        return "I'm tracking some brands, but none have final scores yet!"

    # OMI-FIED RESPONSE
    response_lines = ["You got it! Here's the full list of brands I'm tracking, ranked by their sustainability score (out of 30):\n"]
    
    # Use itertuples for efficiency
    for row in all_ranked.itertuples():
        # Format to 1 decimal place if it's a float, or show as is
        score_str = f"{row.final_score:.1f}" if isinstance(row.final_score, float) else str(row.final_score)
        response_lines.append(f"• **{row.brand_name}** (Score: {score_str})")
    
    response_lines.append("\nJust ask me to 'rank [brand name]' for a detailed breakdown!")
    return "\n".join(response_lines)

# =========================
# Quiz Logic
# =========================
def offer_quiz(session_data: dict, quiz_type: str) -> str:
    session_data["waiting_for_quiz_start"] = True
    session_data["quiz_type_pending"] = quiz_type
    # OMI-FIED RESPONSE
    type_text = "hair and skin" if quiz_type == "skin" else "hair" # Default to skin if ambiguous, but message is generic
    if quiz_type == 'hair': type_text = 'hair'
    if quiz_type == 'skin': type_text = 'skin'
    
    return f"Of course! To help find the *best* products for you, I've got a super quick quiz to figure out your {type_text} type. Sound good?"

def start_quiz(session_data: dict, quiz_type: str) -> str:
    session_data.update({"quiz_answers": [], "waiting_for_quiz_start": False})
    try:
        with open(os.path.join(QUIZZES_DIR, f"{quiz_type}.json"), 'r') as f:
            quiz_data = json.load(f)
    except Exception:
        return "Oh no! 😅 I seem to have misplaced my quiz cards. Ask me something else for now?"
    session_data["current_quiz_session"] = {"quiz_data": quiz_data, "question_idx": 0}
    # OMI-FIED RESPONSE
    return "Okay, let's do it! 💃\n\n" + get_next_quiz_question(session_data)

def get_next_quiz_question(session_data: dict) -> str:
    quiz_session = session_data["current_quiz_session"]
    idx = quiz_session["question_idx"]
    questions = quiz_session["quiz_data"]["questions"]
    if idx >= len(questions): return finish_quiz(session_data)
    q = questions[idx]
    options_text = "\n".join([f"{i+1}. {opt['text']}" for i, opt in enumerate(q["options"])])
    return f"**Question {q['id']}**: {q['question']}\n{options_text}\n\n_(Just type the number!)_"

def answer_quiz_option(session_data: dict, option_num: int) -> str:
    quiz_session = session_data["current_quiz_session"]
    q = quiz_session["quiz_data"]["questions"][quiz_session["question_idx"]]
    if 1 <= option_num <= len(q["options"]):
        session_data["quiz_answers"].append(q["options"][option_num-1]["answer"])
        quiz_session["question_idx"] += 1
        return get_next_quiz_question(session_data)
    return f"Whoops! Try that again - just type a number from 1 to {len(q['options'])}."

def finish_quiz(session_data: dict) -> str:
    try:
        quiz_data = session_data["current_quiz_session"]["quiz_data"]
        most_common_answer = Counter(session_data["quiz_answers"]).most_common(1)[0][0]
        result_type = quiz_data["results_logic"][most_common_answer]
        routine = quiz_data["routines"][result_type]
        session_data.update({"current_quiz_session": None, "quiz_answers": []})
        # OMI-FIED RESPONSE
        return f"All done! ✨ Based on your answers, it looks like you have **{result_type}**!\n\nHere’s a simple routine I'd recommend:\n{routine}"
    except Exception as e:
        print(f"[ERROR] in finish_quiz: {e}")
        session_data.update({"current_quiz_session": None, "quiz_answers": []})
        return "Ah, I scribbled down your answers but totally fumbled the results. 😅 So sorry! Can I help with something else?"

# =========================
# Main Response Generator
# =========================
def get_rag_response(question: str, user_id: str) -> str:
    session_data = _session_manager.get_session(user_id)
    raw_q = str(question).strip()
    
    # This variable will hold the final answer
    answer = ""
    
    # This will hold the chat history entry
    chat_entry = None

    if not raw_q: 
        answer = "Heeey! You still there? 😊"
        # === MODIFIED: Create chat_entry and update session ---
        ts = firestore.SERVER_TIMESTAMP if _session_manager.db else datetime.now().isoformat()
        chat_entry = {"timestamp": ts, "question": "N/A (empty input)", "answer": answer}
        _session_manager.update_session(user_id, session_data, chat_entry)
        return answer

    session_data['response_count'] = session_data.get('response_count', 0) + 1
    add_suggestion = True

    is_affirmative = is_affirmative_response(raw_q)
    is_negative = is_negative_response(raw_q)

    # --- BLOCK A: Handle special triggers and stateful responses FIRST ---
    # (This block remains functionally identical to preserve quiz/workbook/email logic)
    if raw_q == "__GET_ONBOARDING__":
        session_data['waiting_for_user_classification'] = True
        # Note: __GET_ONBOARDING__ is a special internal ping.
        # We don't log it as a user-facing chat, so we update and return early.
        _session_manager.update_session(user_id, session_data, chat_entry=None)
        return "" # Frontend handles this

    elif session_data.get("waiting_for_user_classification"):
        cleaned_q = _clean_text(raw_q)
        session_data['waiting_for_user_classification'] = False
        if 'eco shopper' in cleaned_q or 'ecoshopper' in cleaned_q:
            session_data['user_type'] = 'eco_shopper'
            # OMI-FIED RESPONSE
            answer = ("🌱 Heyyy, welcome to Omi Live, your new eco-bestie! Ready to find some *actually* legit eco-friendly brands?\n\n"
                      "Here’s what you get with the Omi Fam:\n"
                      "• Our AI Green Rating System (no greenwashing here!)\n"
                      "• Exclusive deals 🤫\n"
                      "• Chats with the brand founders themselves\n"
                      "• A front-row seat to watch awesome eco-brands grow!")
        elif 'creator' in cleaned_q:
            session_data['user_type'] = 'creator'
            # OMI-FIED RESPONSE
            answer = ("🎥 Ooh, a creator! Love it. Are you interested in live shopping?\n\n"
                      "With Omi Live, you can monetize your amazing content through:\n"
                      "• Free product samples (the good stuff!)\n"
                      "• Sales commissions\n"
                      "• Flat fee collabs\n"
                      "• Connecting with eco-conscious shoppers who *get* it")
            session_data['waiting_for_workbook_confirmation'] = True
            answer += "\n\nWe actually made a Live Sales Workbook just for creators—it's full of tips to maximize earnings. Want it?"
        elif 'brand owner' in cleaned_q or 'brandowner' in cleaned_q:
            session_data['user_type'] = 'brand_owner'
            # OMI-FIED RESPONSE
            answer = ("👋 Hey there! Welcome to Omi Live — we're AI-powered retail tech for amazing eco-friendly brands. Are you a brand owner looking to grow?\n\n"
                      "We help brands like yours hit 20% sales conversion (!!!) through:\n"
                      "• A super-loyal eco-conscious community\n"
                      "• Smart tools to get your products seen\n"
                      "• Live storytelling that builds *real* trust")
            session_data['waiting_for_workbook_confirmation'] = True
            answer += "\n\nWould you like our Live Sales Workbook? It’s packed with strategies to boost sales. Want me to send it over?"
        else:
            session_data['waiting_for_user_classification'] = True
            answer = "Hmm, that's not one of the options. 😅 Please choose one of the buttons below!"
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
            # OMI-FIED RESPONSE
            answer = "You’re all set! We’ll share tips, community insights, and opportunities to feature your brand on Omi Live.\n\nWe’re having your personalized matches brewing. Stay tuned on Omi updates!"
        add_suggestion = False

    elif session_data.get("current_quiz_session"):
        if _clean_text(raw_q).isdigit():
            answer = answer_quiz_option(session_data, int(_clean_text(raw_q)))
        else:
            session_data["current_quiz_session"] = None
            answer = "Quiz cancelled! No worries. What else is on your mind?"
        add_suggestion = False

    elif session_data.get("waiting_for_quiz_start"):
        if is_affirmative:
            quiz_type = session_data.get('quiz_type_pending', 'skin')
            answer = start_quiz(session_data, quiz_type)
        else:
            session_data["waiting_for_quiz_start"] = False
            answer = "Okay, no problem! We can totally do it later if you change your mind."
        add_suggestion = False

    elif session_data.get("waiting_for_workbook_confirmation"):
        if is_affirmative:
            session_data['waiting_for_email'] = True
            if session_data.get('user_type') == 'creator':
                answer = "Awesome! What's your email? I'll send it over (plus early invites to campaigns!)."
            else:  # Brand Owner
                answer = "Great! Just drop your email, and I'll send the workbook and info on early access to our tools. ✨"
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
        answer = "👍 No worries at all! We'll just keep chatting here."
        add_suggestion = False

    # --- BLOCK B: If no stateful response, handle new query ---
    if not answer:
        add_suggestion = True
        session_data.update({
            'waiting_for_quiz_start': False, 'quiz_type_pending': None,
            'waiting_for_rank_confirmation': False, 'brand_to_rank': None
        })

        cleaned_q = _clean_text(raw_q)

        # NEW: High-priority check for workbook requests (logic unchanged)
        if any(keyword in cleaned_q for keyword in WORKBOOK_KEYWORDS):
            user_type = session_data.get('user_type')
            if user_type in ['creator', 'brand_owner']:
                if session_data.get('email_address'):
                    answer = _get_workbook_response()  # Email on file, send link
                else:
                    session_data['waiting_for_email'] = True  # No email, ask for it
                    answer = "Of course! To send you the workbook, what's your email?"
            else:  # Eco-shopper or unclassified
                # OMI-FIED RESPONSE
                answer = "Ooh, the Live Sales Workbook is a special goodie for Creators and Brand Owners. Let me know if you wanna learn more about those roles!"
            add_suggestion = False

        if not answer:
            if "rank brand" in cleaned_q or "brand ranking" in cleaned_q or "suggest brand" in cleaned_q or "list brand" in cleaned_q:
                answer = respond_with_brand_info()
                # === FIX 4a: Don't add suggestion to brand ranking list ===
                add_suggestion = False 
            else:
                quiz_type = detect_routine_intent(raw_q)
                if quiz_type:
                    answer = offer_quiz(session_data, quiz_type)
                    # === FIX 4b: Don't add suggestion when *offering* quiz ===
                    add_suggestion = False 
                else:
                    candidates = fuzzy_lookup_brand_candidates(raw_q)
                    if len(candidates) == 1:
                        brand_name = candidates[0]
                        session_data["waiting_for_rank_confirmation"] = True
                        session_data["brand_to_rank"] = brand_name
                        # OMI-FIED RESPONSE
                        answer = f"I totally know the brand '{brand_name}'! Want me to pull up its sustainability ranking for you?"
                        add_suggestion = False
                    elif len(candidates) > 1:
                        # OMI-FIED RESPONSE
                        answer = "Hmm, did you mean one of these? I can 'rank' any of them for you!\n- " + "\n- ".join(candidates)
                        add_suggestion = False

                    # ======================================================
                    # === NEW RAG LOGIC BLOCK (REPLACES OLD RAG LOGIC) ===
                    # ======================================================
                    if not answer:
                        retriever = get_retriever()
                        if not retriever:
                            # OMI-FIED RESPONSE
                            answer = "Ah, my brain's feeling a little foggy right now and I can't access my knowledge base. 😅 Please try again in a sec!"
                        else:
                            context_docs = retriever.invoke(raw_q)
                            context = "\n\n".join(d.page_content for d in context_docs)
                            
                            # NEW: Always call the LLM, even with empty context.
                            # The new prompt handles all 3 cases:
                            # 1. Off-topic (e.g., "weather?")
                            # 2. On-topic, no context (e.g., "eco-friendly tires?")
                            # 3. On-topic, with context (e.g., "why bees?")
                            
                            # Join the redirect topics into a simple string list
                            redirect_list_str = "\n".join([f"- {topic}" for topic in OMI_REDIRECT_TOPICS])
                            
                            prompt = QA_PROMPT_GENERAL.format(
                                persona=SYSTEM_PERSONA,
                                context=context,
                                question=raw_q,
                                redirect_topics=redirect_list_str
                            )
                            answer = get_llm().invoke(prompt).content
                    # ======================================================
                    # === END OF NEW RAG LOGIC BLOCK ===
                    # ======================================================

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

    # === MODIFICATION START ===
    # This logic now ensures the newsletter prompt is appended as a separate, delimited message.
    # Your frontend should split the *entire* response string by "\n + \n" to create separate-bubbles.
    if should_prompt_email:
        session_data['waiting_for_email'] = True
        
        # OMI-FIED RESPONSE
        newsletter_prompt = ("\n + \n💫 Ngl, we're totally vibing! I'd love to keep this going - want to join our exclusive newsletter? "
                             "What's your email? 🌱")
        
        # We strip the main 'answer' to prevent extra newlines from breaking the delimiter.
        # This joins the 5th response and the newsletter prompt with your specific delimiter.
        answer = answer.strip() + newsletter_prompt
        
        # This is crucial: it prevents a *different* suggestion from *also* being added.
        add_suggestion = False
    # === MODIFICATION END ===

    if add_suggestion:
        follow = get_follow_up_suggestion(session_data)
        if follow and follow.strip() not in answer:
            answer += follow
            
    # === MODIFIED: Create chat entry and update session ===
    
    # --- *** THIS IS THE FIX *** ---
    # Ensure 'answer' is never None, which breaks Firestore.
    # If the LLM response was empty or flagged, default to a safe string.
    if answer is None:
        answer = "I'm not sure what to say to that. 😅"
    # --- *** END OF FIX *** ---

    # Determine timestamp type based on DB connection
    ts = firestore.SERVER_TIMESTAMP if _session_manager.db else datetime.now().isoformat()
    
    chat_entry = {
        "timestamp": ts,
        "question": raw_q, # The user's original, uncleaned question
        "answer": answer   # Now this is guaranteed to be a string
    }
    
    # Pass the chat entry to be atomically appended
    _session_manager.update_session(user_id, session_data, chat_entry)
    
    return answer

# CLI test function
if __name__ == "__main__":
    preload_faiss_index()
    print("🤖 OMI Bot is ready! (Now with 100% more personality ✨)")
    cli_session = {"user_id": "cli_user"}
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']: break
        
        # === THIS IS THE FIX ===
        response = get_rag_response(user_input, cli_session['user_id']) 
        # =======================
        
        # This mimics how the frontend should split messages
        messages = response.split("\n + \n")
        print(f"OMI: {messages[0]}")
        if len(messages) > 1:
            for msg in messages[1:]:
                # In a real UI, this would be a new, separate chat bubble
                print(f"OMI (Follow-up): {msg}")
