import os
import re
import csv
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

from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import safe_join

# LangChain / embeddings / vectorstore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Vertex AI Integrations
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain.prompts import PromptTemplate

# =========================
# Setup & Globals
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

load_dotenv()
GOOGLE_VERTEX_API_KEY = os.getenv("GOOGLE_VERTEX_API")
if GOOGLE_VERTEX_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_VERTEX_API_KEY

DATA_DIR = os.environ.get('DATA_DIR', 'data')
FAQ_PATH = os.path.join(DATA_DIR, "omi_faq.txt")
KB_PATH = os.path.join(DATA_DIR, "omilive_knowledge_base.txt")
BRAND_CSV = os.path.join(DATA_DIR, "brand_metric_dataset.csv")
WORKBOOK_FILENAME = "Omi_Live_-_Live_Sales_Tactical_Workbook.docx" # Updated file extension
WORKBOOK_PATH = os.path.join(DATA_DIR, WORKBOOK_FILENAME)

QUIZZES_DIR = "quizzes"

VECTORSTORE_DIR = os.environ.get('VECTORSTORE_DIR', os.path.join(DATA_DIR, "omi_index"))

# Flask App Setup
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "https://omilivechatbot.netlify.app,http://localhost:3000")
CORS(app, origins=[origin.strip() for origin in allowed_origins.split(",") if origin.strip()], supports_credentials=True)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "omi-chatbot-secret-fallback-key")
app.config.update(SESSION_COOKIE_SECURE=True, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

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
        if user_id in self.sessions:
            self.sessions[user_id].update(updates)
            self._save_sessions()

_session_manager = UserSessionManager()
def get_user_id():
    # Simple, non-production user ID from Flask session
    if "user_id" not in session:
        session["user_id"] = os.urandom(16).hex()
    return session["user_id"]

# =========================
# RAG Chain Logic
# =========================
_llm: Optional[ChatVertexAI] = None
_retriever = None
_brand_df: pd.DataFrame = pd.DataFrame()
_waiting_for_workbook_confirmation: bool = False
_current_quiz_session = None
_quiz_answers = []
GREETINGS = ("hi", "hello", "hey", "good morning", "good evening", "good afternoon")

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

def get_llm() -> ChatVertexAI:
    global _llm
    if _llm is not None:
        return _llm
    if not GOOGLE_VERTEX_API_KEY:
        raise RuntimeError("GOOGLE_VERTEX_API missing. Add it to your .env.")
    _llm = ChatVertexAI(
        model_name="gemini-pro",
        temperature=0.4,
    )
    return _llm

def build_retriever(save_local: bool = True):
    faq_text = _safe_read(FAQ_PATH)
    kb_text = _safe_read(KB_PATH)
    if not faq_text and not kb_text:
        raise RuntimeError("No source text found.")

    docs = []
    if faq_text:
        docs += TextLoader(FAQ_PATH, encoding="utf-8").load()
    if kb_text:
        docs += TextLoader(KB_PATH, encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    print("[INFO] Creating FAISS index...")
    vect = FAISS.from_documents(chunks, embeddings)

    if save_local:
        try:
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            vect.save_local(VECTORSTORE_DIR)
            print(f"[INFO] Saved FAISS vectorstore to: {VECTORSTORE_DIR}")
        except Exception as e:
            print("[WARN] Could not save FAISS vectorstore:", e)

    return vect.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def load_retriever_from_disk():
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(f"Vectorstore directory not found: {VECTORSTORE_DIR}")
    print(f"[INFO] Loading FAISS vectorstore from disk: {VECTORSTORE_DIR}")
    vect = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return vect.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    try:
        _retriever = load_retriever_from_disk()
        return _retriever
    except Exception as e:
        print("[WARN] Could not load retriever from disk:", e)
    try:
        _retriever = build_retriever(save_local=True)
        return _retriever
    except Exception as e:
        print("[ERROR] Failed to build retriever:", e)
        traceback.print_exc()
        raise

def preload_faiss_index():
    return get_retriever()

def retrieve_context(query: str, k: int = 5) -> str:
    try:
        retriever = get_retriever()
        docs = retriever.get_relevant_documents(query)[:k]
        return "\n\n".join(d.page_content for d in docs if d and d.page_content)
    except Exception as e:
        print("[ERROR] Retrieval failed:", e)
        traceback.print_exc()
        return ""

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

def load_quiz(file_name: str) -> dict:
    path = os.path.join(QUIZZES_DIR, file_name)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {}

def start_quiz(quiz_type: str) -> str:
    global _current_quiz_session, _quiz_answers
    quiz_file = "hair.json" if quiz_type == "hair" else "skin.json"
    quiz_data = load_quiz(quiz_file)
    if not quiz_data:
        return "Sorry, I couldn't load the quiz."
    _current_quiz_session = {"quiz_data": quiz_data, "question_idx": 0, "quiz_type": quiz_type}
    _quiz_answers = []
    return (f"### ✨ {quiz_type.capitalize()} Routine Quiz\n"
            f"I can definitely help with a {quiz_type} routine! To personalize it, I'll ask a few quick questions.\n\n"
            f"**Ready to start the {quiz_type} quiz?** Type **'start'** to begin!")

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

def get_quiz_recommendation(result_type: str, category: str) -> str:
    query = f"{result_type} {category} routine recommendation"
    context = retrieve_context(query, k=3)
    llm = get_llm()
    prompt = f"""You are OMI. Based on the context below, recommend products and a routine for {category} type: {result_type}.
    Be concise, practical, and friendly. Only use the information provided in the context.
    Context: {context}
    Format the response in markdown with:
    - A clear H3 header for the routine
    - Short paragraph (1–2 sentences) summary
    - Bullet points for steps with **bold** product names
    - Optional links in [text](url) format
    - Finish with a short tip block prefixed by "💡 Tip:"
    Response:"""
    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or (str(resp) if resp is not None else None) or "I don't know."
        text = text.strip()
        if not re.search(r"^#{3}\s", text):
            text = f"### 🌿 Recommended {category.capitalize()} Routine for *{result_type}*\n\n{text}"
        return text
    except Exception as e:
        print("[ERROR] LLM recommendation failed:", e)
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
    return (f"**Your {quiz_type} type:** *{result_type}*\n\n"
            f"{recommendation}\n\n"
            f"📘 Want me to send the *Omi Live Tactical Workbook* for **{quiz_type}** care?")

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

def get_rag_response(question: str) -> str:
    global _waiting_for_workbook_confirmation
    user_id = get_user_id()
    session_data = _session_manager.get_session(user_id)
    session_data['interaction_count'] += 1

    if not question or not str(question).strip():
        return "I don't know."

    raw_q = str(question).strip()
    cleaned_q = _clean_text(raw_q)
    
    if re.match(r"[^@]+@[^@]+\.[^@]+", raw_q) and session_data.get('email') is None:
        _session_manager.update_session(user_id, {'email': raw_q})
        return "🎉 **Thanks for signing up!** You'll hear from us soon. How can I help you next?"

    if (session_data.get('response_count', 0) >= 3 and session_data.get('email') is None and
            not _waiting_for_workbook_confirmation and not _current_quiz_session and
            not any(cleaned_q.startswith(g) for g in GREETINGS)):
        _session_manager.update_session(user_id, {'last_prompted_at': datetime.now().isoformat()})
        session_data['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session_data['response_count']})
        return "We're totally vibing! 💫 **Join our newsletter?**\n- Early access to sustainable brand deals\n- New eco finds and community tips\n- Free live shopping workbook for creators/brands\n\n**Drop your email** and I'll add you. 🌱"

    if _current_quiz_session:
        if cleaned_q.isdigit():
            response = answer_quiz_option(int(cleaned_q))
        elif cleaned_q in ["start", "yes", "begin"]:
            response = get_next_quiz_question()
        else:
            response = "Please enter the number of your choice for the quiz or type **'start'** to begin."
        session_data['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session_data['response_count']})
        return response

    if not _current_quiz_session and session_data.get('email') is not None:
        routine_type = detect_routine_intent(raw_q)
        if routine_type:
            response = start_quiz(routine_type)
            session_data['response_count'] += 1
            _session_manager.update_session(user_id, {'response_count': session_data['response_count']})
            return response

    if cleaned_q in {"start hair quiz", "hair quiz"}:
        response = start_quiz("hair")
        session_data['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session_data['response_count']})
        return response
    if cleaned_q in {"start skin quiz", "skin quiz"}:
        response = start_quiz("skin")
        session_data['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session_data['response_count']})
        return response

    if any(cleaned_q.startswith(g) for g in GREETINGS):
        greeting_index = session_data.get('greeting_index', 0)
        greeting = VARIED_GREETINGS[greeting_index % len(VARIED_GREETINGS)]
        response = f"{greeting} I also have a workbook — *Omi Live Tactical Workbook* 📘. Would you like me to send it?"
        session_data['response_count'] += 1
        session_data['greeting_index'] = (greeting_index + 1) % len(VARIED_GREETINGS)
        _session_manager.update_session(user_id, {'response_count': session_data['response_count'], 'greeting_index': session_data['greeting_index']})
        return response

    if "workbook" in cleaned_q:
        _waiting_for_workbook_confirmation = True
        response = "I have the *Omi Live Tactical Workbook* 📘 — do you want it in **.docx** format where you can download?"
        session_data['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session_data['response_count']})
        return response

    if _waiting_for_workbook_confirmation and cleaned_q in {"yes", "sure", "okay", "ok", "yep", "yeah"}:
        _waiting_for_workbook_confirmation = False
        backend_url = os.environ.get("BACKEND_URL", request.host_url.rstrip("/"))
        response = f'Great! 🎉 You can download the workbook here: <a href="{backend_url}/get_workbook" target="_blank" download>Download Workbook</a>'
        session_data['response_count'] += 1
        _session_manager.update_session(user_id, {'response_count': session_data['response_count']})
        return response

    context = retrieve_context(raw_q, k=5)
    if not context.strip():
        response = "I don't know."
    else:
        answer = answer_with_context(raw_q, context)
        response = answer
    
    session_data['response_count'] += 1
    _session_manager.update_session(user_id, {'response_count': session_data['response_count']})
    return response

# =========================
# Flask Routes
# =========================
@app.before_first_request
def preload_on_startup():
    print("[INFO] Preloading FAISS index...")
    try:
        preload_faiss_index()
        print("[INFO] ✅ FAISS index ready.")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to preload FAISS index: {e}")
        traceback.print_exc()

@app.route("/")
def index():
    session.pop("history", None)
    return jsonify({"status": "API is running", "version": "1.0"})

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/register-email", methods=["POST"])
def register_email():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        email = data.get("email", "").strip()
        if not email or "@" not in email:
            return jsonify({"status": "invalid", "message": "Invalid email format"}), 400
        file_path = os.path.join(DATA_DIR, "user_emails.csv")
        file_exists = os.path.isfile(file_path)
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["email", "timestamp"])
            writer.writerow([email, os.environ.get("DEPLOYMENT_ID", "local")])
        print(f"📩 New user email registered: {email}")
        return jsonify({"status": "success", "message": "Email registered successfully"})
    except Exception as e:
        print(f"[ERROR] Saving email: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if not request.is_json:
            return jsonify({"answer": "Invalid content type"}), 400
        data = request.get_json()
        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"answer": "Empty message received"}), 400
        answer = get_rag_response(user_input)
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"[ERROR] in /chat route: {e}")
        traceback.print_exc()
        return jsonify({"answer": "⚠️ Error occurred. Please try again."}), 500

@app.route("/get_workbook", methods=["GET"])
def get_workbook():
    try:
        file_path = safe_join(os.getcwd(), WORKBOOK_PATH)
        if not os.path.exists(file_path):
            return jsonify({"error": "Workbook not found"}), 404
        return send_from_directory(
            directory=os.path.dirname(file_path),
            path=os.path.basename(file_path),
            as_attachment=True,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception as e:
        print(f"[ERROR] Sending workbook: {e}")
        traceback.print_exc()
        return jsonify({"error": "Error sending workbook"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# =========================
# Run App (for local dev)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)