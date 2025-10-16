from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import os
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime

# Import from rag_chain
try:
    from rag_chain import get_rag_response, preload_faiss_index, db
except ImportError as e:
    print(f"[WARNING] Could not import rag_chain modules: {e}")
    get_rag_response = lambda *args: "Chat functionality is temporarily unavailable."
    preload_faiss_index = lambda: None
    db = None

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)
# Trust the X-Forwarded-For headers from the proxy (e.g., Google Cloud Run)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# --- CORS Configuration ---
# This is still required to allow your Wix site to make requests to this server.
allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://omilivechatbot.netlify.app,http://localhost:3000,https://www.omilive.com"
)
CORS(
    app,
    origins=[origin.strip() for origin in allowed_origins.split(",") if origin.strip()],
    supports_credentials=True # Good practice for allowing custom headers if needed in the future.
)

# NOTE: All Flask session and cookie configuration (app.secret_key, app.config.update)
# has been removed. This server is now fully stateless.

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return jsonify({"status": "OMI Chatbot API is running"})

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/register-email", methods=["POST"])
def register_email():
    """Receives an email from the frontend and saves it to Firestore."""
    try:
        if not db:
            print("[ERROR] Firestore client (db) is not available.")
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
            'source': "chatbot-embed"
        })

        print(f"📩 New user email registered in Firestore: {email}")
        return jsonify({"status": "success", "message": "Email registered successfully"})
    except Exception as e:
        print(f"[ERROR] Failed to save email to Firestore: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": "Internal server error"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint that handles conversation logic using a stateless token."""
    try:
        if not request.is_json:
            return jsonify({"answer": "Invalid request: Content-Type must be application/json"}), 415
        
        data = request.get_json()
        user_input = data.get("message", "").strip()
        # The frontend will send a `session_token` on every request after the first one.
        session_token = data.get("session_token") 

        if not user_input:
            return jsonify({"answer": "Empty message received"}), 400

        # If no token is provided by the client, it's a new user. Generate a token.
        if not session_token:
            user_id = os.urandom(16).hex()
            print(f"[INFO] No token received. New session created for user_id: {user_id}")
        else:
            user_id = session_token
        
        # Get the response from the main RAG chain logic, using the token as the user_id
        answer = get_rag_response(user_input, user_id)
        
        # **THE FIX**: ALWAYS return the token (either the new one or the existing one) back to the client.
        # The frontend is responsible for storing it and sending it with the next request.
        return jsonify({"answer": answer, "session_token": user_id})

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in /chat route: {e}")
        traceback.print_exc()
        return jsonify({"answer": "⚠️ I'm having a little trouble right now. Please try again."}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "This endpoint does not exist."}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "An internal server error occurred."}), 500

# =========================
# Preloading and App Execution
# =========================
if __name__ != "__main__":
    # This block runs when the app is started by a production server like Gunicorn
    print("[INFO] Preloading FAISS index for production server...")
    try:
        preload_faiss_index()
        print("[INFO] ✅ FAISS index has been successfully preloaded.")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to preload FAISS index for production: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # This block runs when you execute 'python app.py' directly for local testing
    print("[INFO] Preloading FAISS index for local development...")
    try:
        preload_faiss_index()
        print("[INFO] ✅ FAISS index ready for local development.")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to preload FAISS index locally: {e}")

    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)