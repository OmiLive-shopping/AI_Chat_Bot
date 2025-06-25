from flask import Flask, render_template, request, jsonify, session
from rag_chain import get_rag_response

app = Flask(__name__)
app.secret_key = "omi-chatbot-secret"

@app.route('/')
def index():
    session.pop("history", None)
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")

    if "history" not in session:
        session["history"] = []

    chat_history = session["history"]
    answer = get_rag_response(user_input, chat_history)

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": answer})
    session["history"] = chat_history

    return jsonify({"answer": answer})

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5050)


