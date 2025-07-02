from flask import Flask, request, jsonify, session
from rag_chain import get_rag_response
from flask_cors import CORS
import traceback

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

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("message", "")
        print(f"Received message: {user_input}")

        if "history" not in session:
            session["history"] = []

        chat_history = session["history"]
        answer = get_rag_response(user_input, chat_history)
        print(f"Generated answer: {answer}")

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})
        session["history"] = chat_history

        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error in /chat: {e}")
        traceback.print_exc()
        return jsonify({"answer": "⚠️ Error occurred."}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=8000)
