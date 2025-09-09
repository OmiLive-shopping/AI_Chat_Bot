from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
import traceback, csv, os
from werkzeug.middleware.proxy_fix import ProxyFix

# Import from rag_chain
try:
    from rag_chain import get_rag_response, preload_faiss_index
except ImportError as e:
    print(f"[WARNING] Could not import rag_chain modules: {e}")

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configure CORS properly for production
CORS(app, 
     origins=os.environ.get('ALLOWED_ORIGINS', '').split(',') or ['http://localhost:3000'],
     supports_credentials=True)

# Use environment variable for secret key with fallback
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'omi-chatbot-secret-fallback-key')

# Set session configuration for production
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# =========================
# Preload FAISS index at startup
# =========================
print("[INFO] Preloading FAISS index...")
try:
    preload_faiss_index()
    print("[INFO] ✅ FAISS index ready.")
except Exception as e:
    print(f"[ERROR] ❌ Failed to preload FAISS index: {e}")
    traceback.print_exc()

# =========================
# Routes
# =========================
@app.route("/")
def index():
    """Root route — clears session and returns status."""
    session.pop("history", None)
    return jsonify({"status": "API is running", "version": "1.0"})

@app.route("/health")
def health():
    """Health check route for Cloud Run."""
    return jsonify({"status": "healthy"}), 200

@app.route("/register-email", methods=["POST"])
def register_email():
    """Register a new user email into CSV file."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
            
        email = data.get("email", "").strip()

        if not email or "@" not in email:
            return jsonify({"status": "invalid", "message": "Invalid email format"}), 400

        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        file_path = "data/user_emails.csv"
        
        # Check if file exists to write headers
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["email", "timestamp"])
            writer.writerow([email, os.environ.get('DEPLOYMENT_ID', 'local')])

        print(f"📩 New user email registered: {email}")
        return jsonify({"status": "success", "message": "Email registered successfully"})
    
    except Exception as e:
        print(f"[ERROR] Saving email: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint — handles user messages and returns AI response."""
    try:
        if not request.is_json:
            return jsonify({"answer": "Invalid content type"}), 400
            
        data = request.get_json()
        user_input = data.get("message", "").strip()
        
        if not user_input:
            return jsonify({"answer": "Empty message received"}), 400
            
        print(f"[DEBUG] Received message: {user_input}")

        if "history" not in session:
            session["history"] = []
        chat_history = session["history"]

        # Special handling for workbook request
        if "workbook" in user_input.lower():
            backend_url = os.environ.get('BACKEND_URL', request.host_url.rstrip("/"))
            answer = f'Great! 🎉 <a href="{backend_url}/get_workbook" target="_blank" download>Download Workbook</a>'
        else:
            # Check if rag_chain is available
            if 'get_rag_response' not in globals():
                answer = "⚠️ Chat functionality is temporarily unavailable. Please try again later."
            else:
                answer = get_rag_response(user_input, chat_history)

        print(f"[DEBUG] Answer: {answer}")

        # Update chat history (limit to last 10 exchanges to prevent session overload)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})
        session["history"] = chat_history[-20:]  # Keep only last 10 exchanges

        return jsonify({"answer": answer})
    
    except Exception as e:
        print(f"[ERROR] in /chat route: {e}")
        traceback.print_exc()
        return jsonify({"answer": "⚠️ Error occurred. Please try again."}), 500

@app.route("/get_workbook", methods=["GET"])
def get_workbook():
    """Serve downloadable workbook file."""
    try:
        file_name = "Omi_Live_-_Live_Sales_Tactical_Workbook.docx"
        file_dir = os.path.join(os.getcwd(), "data")

        if not os.path.exists(os.path.join(file_dir, file_name)):
            return jsonify({"error": "Workbook not found"}), 404

        return send_from_directory(
            directory=file_dir,
            path=file_name,
            as_attachment=True,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception as e:
        print(f"[ERROR] Sending workbook: {e}")
        traceback.print_exc()
        return jsonify({"error": "Error sending workbook"}), 500

# Error handlers
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
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host="0.0.0.0", port=port, debug=debug)