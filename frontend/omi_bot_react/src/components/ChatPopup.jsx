// src/components/ChatPopup.jsx
import React, { useState, useRef, useEffect } from "react";
import ChatMessage from "./ChatMessage";

export default function ChatPopup({ onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const chatBoxRef = useRef(null);

  const scrollToBottom = () => {
    chatBoxRef.current?.scrollTo({
      top: chatBoxRef.current.scrollHeight,
      behavior: "smooth",
    });
  };

  const handleSend = async () => {
    const message = input.trim();
    if (!message) return;

    console.log("Sending message:", message);

    const newMessages = [...messages, { type: "user", text: message }];
    setMessages([...newMessages, { type: "bot", text: "🤖 OmiBot is thinking..." }]);
    setInput("");

    try {
      const res = await fetch("http://localhost:8000/chat", {
        // If Vite proxy doesn't work, uncomment this instead:
        // const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();

      setMessages([
        ...newMessages,
        { type: "bot", text: `🤖 ${data.answer}` },
      ]);
    } catch (err) {
      console.error("Fetch error:", err);
      setMessages([
        ...newMessages,
        { type: "bot", text: `⚠️ Error: ${err.message}` },
      ]);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div
      id="chat-popup"
      className={isFullscreen ? "fullscreen" : ""}
      style={{ display: "flex", flexDirection: "column" }}
    >
      <header className="chat-header">
        <div className="header-left">OmiBot | Omi Live</div>
        <div className="chat-header-right">
          <button className="new-chat-btn" onClick={() => window.location.reload()}>
            New Chat
          </button>
          <button className="fullscreen-btn" onClick={() => setIsFullscreen(!isFullscreen)}>
            ⛶
          </button>
          <button className="close-btn" onClick={onClose}>
            ❌
          </button>
        </div>
      </header>

      <main className="chat-shell">
        {messages.length === 0 && (
          <h1 className="welcome">What's on your mind today?</h1>
        )}

        <div className="chat-box" id="chat-box" ref={chatBoxRef}>
          {messages.map((msg, idx) => (
            <ChatMessage key={idx} type={msg.type} text={msg.text} />
          ))}
        </div>

        <div className="input-bar">
          <textarea
            rows="1"
            placeholder="Ask a question"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
          />
          <button className="send-btn" onClick={handleSend}>
            <svg viewBox="0 0 24 24" width="22" height="22">
              <path
                d="M4 12h14M13 5l7 7-7 7"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </main>
    </div>
  );
}
