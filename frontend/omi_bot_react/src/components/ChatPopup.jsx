import React, { useState, useRef, useEffect } from "react";
import ChatMessage from "./ChatMessage";

export default function ChatPopup({ onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [hasAsked, setHasAsked] = useState(false);
  const [email, setEmail] = useState(localStorage.getItem("userEmail") || "");
  const [emailSubmitted, setEmailSubmitted] = useState(!!localStorage.getItem("userEmail"));
  const chatBoxRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
    }
  }, [input]);

  const scrollToBottom = () => {
    chatBoxRef.current?.scrollTo({ top: chatBoxRef.current.scrollHeight, behavior: "smooth" });
  };

  const handleSend = async () => {
    const message = input.trim();
    if (!message) return;

    const newMessages = [...messages, { type: "user", text: message }];
    setMessages([...newMessages, { type: "bot", loading: true }]);
    setInput("");
    setHasAsked(true);

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      const data = await res.json();
      const finalAnswer = data.answer || "I don't know.";

      // Remove placeholder and add real response
      setMessages((prev) => {
        const withoutLoading = prev.filter((msg) => !msg.loading);
        return [...withoutLoading, { type: "bot", text: finalAnswer }];
      });
    } catch (err) {
      console.error("Fetch error:", err);
      setMessages((prev) => {
        const withoutLoading = prev.filter((msg) => !msg.loading);
        return [...withoutLoading, { type: "bot", text: `⚠️ Error: ${err.message}` }];
      });
    }
  };

  const handleEmailSubmit = async () => {
    const trimmed = email.trim();
    const isValid = trimmed.includes("@") && (trimmed.endsWith(".com") || trimmed.endsWith(".edu"));
    if (!isValid) return alert("Please enter a valid email");

    try {
      await fetch("http://localhost:8000/register-email", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: trimmed }),
      });
      localStorage.setItem("userEmail", trimmed);
      setEmailSubmitted(true);
    } catch (err) {
      console.error("Failed to register email:", err);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div id="chat-popup" className={isFullscreen ? "fullscreen" : ""}>
      <header className="chat-header">
        <div className="header-left">OmiBot | OMI Live</div>
        <div className="chat-header-right">
          <button className="new-chat-btn" onClick={() => window.location.reload()}>New Chat</button>
          <button className="fullscreen-btn" onClick={() => setIsFullscreen(!isFullscreen)}>⛶</button>
          <button className="close-btn" onClick={onClose}>❌</button>
        </div>
      </header>

      <main className="chat-shell">
        {!emailSubmitted ? (
          <div className="email-prompt">
            <p>Please enter your email to begin:</p>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="example@domain.com"
            />
            <br />
            <button onClick={handleEmailSubmit}>Submit</button>
          </div>
        ) : messages.length === 0 ? (
          <div className="chat-intro">
            <h1 className="welcome">What's on your mind today?</h1>
            <div className={`input-bar ${isFullscreen && !hasAsked ? "wide-input" : ""}`}>
              <textarea
                ref={textareaRef}
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
                  <path d="M4 12h14M13 5l7 7-7 7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="chat-box" ref={chatBoxRef}>
              {messages.map((msg, idx) => (
                <ChatMessage
                  key={idx}
                  type={msg.type}
                  text={msg.text}
                  loading={msg.loading}
                />
              ))}
            </div>
            <div className={`input-bar ${isFullscreen ? "wide-input" : ""}`}>
              <textarea
                ref={textareaRef}
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
                  <path d="M4 12h14M13 5l7 7-7 7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
