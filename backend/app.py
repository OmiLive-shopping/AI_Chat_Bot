from flask import Flask, request, jsonify, session
from rag_chain import get_rag_response
from flask_cors import CORS
import traceback
import csv
import os

app = Flask(__name__)
CORS(app)
app.secret_key = "omi-chatbot-secret"

@app.route('/')
def index():
    session.pop("history", None)
    return 'API is running'

@app.route('/health')
def health():
    return "OK", 200

@app.route('/register-email', methods=['POST'])
def register_email():
    try:
        data = request.get_json()
        email = data.get("email", "").strip()

        if not email or "@" not in email or not (email.endswith(".com") or email.endswith(".edu")):
            return jsonify({"status": "invalid"}), 400

        os.makedirs("data", exist_ok=True)
        with open("data/user_emails.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([email])

        print(f"📩 New user email registered: {email}")
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"[ERROR] Saving email: {e}")
        traceback.print_exc()
        return jsonify({"status": "error"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("message", "")
        print(f"[DEBUG] Received message: {user_input}")

        if "history" not in session:
            session["history"] = []

        chat_history = session["history"]

        answer = get_rag_response(user_input, chat_history)

        print(f"[DEBUG] Answer: {answer}")

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})

        session["history"] = chat_history

        return jsonify({"answer": answer})
    except Exception as e:
        print(f"[ERROR] in /chat route: {e}")
        traceback.print_exc()
        return jsonify({"answer": "⚠️ Error occurred."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # 👈 This line ensures compatibility with Render
    print("🚀 Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=port)
