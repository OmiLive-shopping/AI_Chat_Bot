import os
import traceback
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import safe_join

try:
    from rag_chain import WORKBOOK_PATH, db, get_rag_response, preload_faiss_index
except ImportError as e:
    print(f"[WARNING] Could not import from rag_chain: {e}")
    # Define placeholder functions and variables if the import fails
    get_rag_response = lambda *args: "Chat functionality is temporarily unavailable."
    preload_faiss_index = lambda: None
    WORKBOOK_PATH = None
    db = None

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)
# Apply ProxyFix middleware to handle headers from reverse proxies (e.g., for correct IP, protocol)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configure CORS to allow requests from specific origins
allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://omilivechatbot.netlify.app,http://localhost:3000,https://www.omilive.com",
)
CORS(
    app,
    origins=[origin.strip() for origin in allowed_origins.split(",") if origin.strip()],
    supports_credentials=True,
)

# Set the secret key for session management from environment variables or a fallback
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "omi-chatbot-secret-fallback-key")

# Configure secure session cookies for cross-domain contexts (e.g., iframe)
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="None",
)

# =========================
# API Routes
# =========================
@app.route("/")
def index():
    """Root endpoint to confirm the API is running."""
    return jsonify({"status": "API is running", "version": "1.0"})


@app.route("/health")
def health():
    """Health check endpoint for monitoring."""
    return jsonify({"status": "healthy"}), 200


@app.route("/register-email", methods=["POST"])
def register_email():
    """Registers a user's email address in the Firestore database."""
    try:
        if not db:
            return jsonify({"status": "error", "message": "Database not configured"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        email = data.get("email", "").strip()
        if not email or "@" not in email:
            return jsonify({"status": "invalid", "message": "Invalid email format"}), 400

        # Create a new document in the 'registered_emails' collection
        email_ref = db.collection("registered_emails").document(email)
        email_ref.set(
            {
                "email": email,
                "timestamp": datetime.utcnow(),
                "source": os.environ.get("DEPLOYMENT_ID", "local"),
            }
        )
        print(f"📩 New user email registered in Firestore: {email}")
        return jsonify({"status": "success", "message": "Email registered successfully"})

    except Exception as e:
        print(f"[ERROR] Saving email to Firestore: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": "Internal server error"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat messages from the user and returns a RAG response."""
    try:
        if not request.is_json:
            return jsonify({"answer": "Invalid content type"}), 400

        data = request.get_json()
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"answer": "Empty message received"}), 400

        # Create a new session for the user if one doesn't exist
        if "user_id" not in session:
            session["user_id"] = os.urandom(16).hex()
            print(f"[INFO] New session created with user_id: {session['user_id']}")

        # Get the response from the RAG chain
        answer = get_rag_response(user_input, session["user_id"])
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"[ERROR] in /chat route: {e}")
        traceback.print_exc()
        return jsonify({"answer": "⚠️ Error occurred. Please try again."}), 500


@app.route("/get_workbook", methods=["GET"])
def get_workbook():
    """Allows users to download a workbook file."""
    try:
        if not WORKBOOK_PATH:
            return jsonify({"error": "Workbook path not set."}), 500

        # Securely join paths to prevent directory traversal attacks
        file_path = safe_join(os.getcwd(), WORKBOOK_PATH)

        if not os.path.exists(file_path):
            return jsonify({"error": "Workbook not found"}), 404

        # Send the file to the user for download
        return send_from_directory(
            directory=os.path.dirname(file_path),
            path=os.path.basename(file_path),
            as_attachment=True,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    except Exception as e:
        print(f"[ERROR] Sending workbook: {e}")
        traceback.print_exc()
        return jsonify({"error": "Error sending workbook"}), 500


# =========================
# Error Handlers
# =========================
@app.errorhandler(404)
def not_found(error):
    """Handles 404 Not Found errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handles 500 Internal Server errors."""
    return jsonify({"error": "Internal server error"}), 500


# =========================
# Application Runner
# =========================
# Preload the FAISS index when running with a production server like Gunicorn
if __name__ != "__main__":
    print("[INFO] Preloading FAISS index for Gunicorn...")
    try:
        preload_faiss_index()
        print("[INFO] ✅ FAISS index ready.")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to preload FAISS index: {e}")
        traceback.print_exc()

# Run the app directly for local development
if __name__ == "__main__":
    print("[INFO] Preloading FAISS index for local development...")
    try:
        preload_faiss_index()
        print("[INFO] ✅ FAISS index ready.")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to preload FAISS index: {e}")
        traceback.print_exc()

    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
