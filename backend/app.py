from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
import traceback
import os
import csv
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import safe_join

# Import from rag_chain
try:
    from rag_chain import get_rag_response, preload_faiss_index, WORKBOOK_PATH, get_user_id
except ImportError as e:
    print(f"[WARNING] Could not import rag_chain modules: {e}")
    get_rag_response = lambda *args: "Chat functionality is temporarily unavailable."
    preload_faiss_index = lambda: None
    WORKBOOK_PATH = None
    get_user_id = lambda *args: "cli_user"

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configure CORS for Netlify + localhost
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "https://omilivechatbot.netlify.app,http://localhost:3000")
CORS(app, origins=[origin.strip() for origin in allowed_origins.split(",") if origin.strip()], supports_credentials=True)

# Secret key for session
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "omi-chatbot-secret-fallback-key")

# Secure session cookies
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# =========================
# Routes
# =========================
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
        os.makedirs("data", exist_ok=True)
        file_path = "data/user_emails.csv"
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
        
        # All conversational logic is handled by get_rag_response
        # Pass the Flask session object to the RAG chain
        answer = get_rag_response(user_input, session)
        
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
if __name__ == "__main__":
    # Preload the FAISS index on application startup
    print("[INFO] Preloading FAISS index...")
    try:
        preload_faiss_index()
        print("[INFO] ✅ FAISS index ready.")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to preload FAISS index: {e}")
        traceback.print_exc()
        # Exit with error code to prevent Gunicorn from running a broken app
        import sys
        sys.exit(1)

    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
