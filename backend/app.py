from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
import traceback
import os
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import safe_join
from datetime import datetime

# Import from rag_chain
try:
    from rag_chain import get_rag_response, preload_faiss_index, WORKBOOK_PATH, db
except ImportError as e:
    print(f"[WARNING] Could not import rag_chain modules: {e}")
    get_rag_response = lambda *args: "Chat functionality is temporarily unavailable."
    preload_faiss_index = lambda: None
    WORKBOOK_PATH = None
    db = None

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configure CORS for Netlify + localhost + Wix
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "https://omilivechatbot.netlify.app,http://localhost:3000,https://www.omilive.com")
CORS(app, origins=[origin.strip() for origin in allowed_origins.split(",") if origin.strip()], supports_credentials=True)

# Secret key for session
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "omi-chatbot-secret-fallback-key")

# Secure session cookies for cross-domain iframe use
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="None"
)

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return jsonify({"status": "API is running", "version": "1.0"})

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/register-email", methods=["POST"])
def register_email():
    try:
        if not db:
            return jsonify({"status": "error", "message": "Database not configured"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        email = data.get("email", "").strip()
        if not email or "@" not in email:
            return jsonify({"status": "invalid", "message": "Invalid email format"}), 400
        
        email_ref = db.collection('registered_emails').document(email)
        email_ref.set({
            'email': email,
            'timestamp': datetime.utcnow(),
            'source': os.environ.get("DEPLOYMENT_ID", "local")
        })

        print(f"📩 New user email registered in Firestore: {email}")
        return jsonify({"status": "success", "message": "Email registered successfully"})
    except Exception as e:
        print(f"[ERROR] Saving email to Firestore: {e}")
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

        # Support token-based session fallback for Safari/Wix
        session_id = data.get("session_id")
        if session_id:
            session['user_id'] = session_id
        elif 'user_id' not in session:
            session['user_id'] = os.urandom(16).hex()
            print(f"[INFO] New session created with user_id: {session['user_id']}")

        answer = get_rag_response(user_input, session['user_id'])
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"[ERROR] in /chat route: {e}")
        traceback.print_exc()
        return jsonify({"answer": "⚠️ Error occurred. Please try again."}), 500

@app.route("/get_workbook", methods=["GET"])
def get_workbook():
    try:
        if not WORKBOOK_PATH:
            return jsonify({"error": "Workbook path not set."}), 500
        
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
# Run App
# =========================
if __name__ != "__main__":
    print("[INFO] Preloading FAISS index for Gunicorn...")
    try:
        preload_faiss_index()
        print("[INFO] ✅ FAISS index ready.")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to preload FAISS index: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("[INFO] Preloading FAISS index for local development...")
    try:
        preload_faiss_index()
        print("[INFO] ✅ FAISS index ready.")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to preload FAISS index: {e}")

    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
