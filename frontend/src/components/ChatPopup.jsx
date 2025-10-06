// frontend/src/components/ChatPopup.jsx
import React, { useState, useRef, useEffect } from "react";
import ChatMessage from "./ChatMessage";

const BASE_URL = "https://omi-backend-355024965259.us-central1.run.app";

export default function ChatPopup({ onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [email, setEmail] = useState("");
  
  // --- MODIFIED: More specific state for email submission ---
  const [showEmailPrompt, setShowEmailPrompt] = useState(false);
  const [emailContext, setEmailContext] = useState("newsletter"); // 'newsletter' or 'workbook'

  const chatBoxRef = useRef(null);
  const textareaRef = useRef(null);
  const scrollIntervalRef = useRef(null);

  // Initial greeting
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          type: "bot",
          text: `Hi there! I'm Omi, your eco-friendly shopping companion! 🌱✨
I'm here to help you discover sustainable brands, learn eco tips, and master live shopping - whether you're a conscious shopper or a creator ready to go live!
Ask me about:
🛍️ Sustainable shopping & green living tips
📱 Live shopping experiences & authentic brand connections
🌿 Eco-friendly brands & sustainability insights
🎯 Creator resources - Get our free step-by-step live shopping workbook!

Ready to chat about conscious commerce? What can I help you with today? 🎉`,
        },
      ]);
    }
  }, []);
  
  // Automatically trigger the onboarding question after the greeting
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    if (
      messages.length === 1 &&
      lastMessage?.type === "bot" &&
      lastMessage.text.includes("What can I help you with today?")
    ) {
      handleSend("__GET_ONBOARDING__", true);
    }
  }, [messages]);

  // --- NEW: Logic to show or hide the email prompt UI ---
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    const emailAlreadySubmitted = 
        (emailContext === 'workbook' && localStorage.getItem('workbookSubmitted')) ||
        (emailContext === 'newsletter' && localStorage.getItem('newsletterSubmitted'));

    if (lastMessage?.type === 'bot' && lastMessage.text.includes("What's your email?") && !emailAlreadySubmitted) {
        setShowEmailPrompt(true);
        // Determine context for submission
        if (lastMessage.text.includes("workbook")) {
            setEmailContext('workbook');
        } else {
            setEmailContext('newsletter');
        }
    } else {
        setShowEmailPrompt(false);
    }
  }, [messages, emailContext]);


  // Auto-resize input
  useEffect(() => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = "auto";
    textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
  }, [input]);

  // Focus input
  useEffect(() => {
    if (textareaRef.current && !showOnboardingButtons && !showEmailPrompt) {
      textareaRef.current.focus();
    }
  }, [isLoading, messages, showOnboardingButtons, showEmailPrompt]);

  // Auto-scroll
  useEffect(() => {
    if (chatBoxRef.current) {
        chatBoxRef.current.scrollTo({
          top: chatBoxRef.current.scrollHeight,
          behavior: "smooth",
        });
      }
  }, [messages]);


  const handleSend = async (messageOverride, isSilent = false) => {
    const message =
      typeof messageOverride === "string" ? messageOverride : input.trim();
    if (!message || isLoading) return;

    setIsLoading(true);

    if (message !== "__GET_ONBOARDING__" && !isSilent) {
      setMessages((prev) => [...prev, { type: "user", text: message }]);
    }
    setInput("");

    setMessages((prev) => [
      ...prev,
      { type: "bot", text: "OmiBot is thinking...", loading: true },
    ]);

    try {
      const res = await fetch(`${BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
        credentials: "include",
      });
      
      const data = await res.json();
      const finalAnswer = data.answer || "I'm having a little trouble right now.";
      
      setMessages((prev) => {
        const updated = prev.filter(
          (msg) => msg.text !== "OmiBot is thinking..."
        );
        return [
          ...updated,
          { type: "bot", text: finalAnswer, loading: false },
        ];
      });

    } catch (err) {
      console.error("Fetch error:", err);
      setMessages((prev) => {
        const updated = prev.filter(
          (msg) => msg.text !== "OmiBot is thinking..."
        );
        return [
          ...updated,
          {
            type: "bot",
            text: `⚠️ Error: Could not connect to the server.`,
            loading: false,
          },
        ];
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleEmailSubmit = async () => {
    const trimmed = email.trim();
    const isValid = /\S+@\S+\.\S+/.test(trimmed);
    if (!isValid) return alert("Please enter a valid email");

    try {
      // Still register email in Firestore
      await fetch(`${BASE_URL}/register-email`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: trimmed }),
        credentials: "include",
      });

      // --- MODIFIED: Use specific flags in localStorage ---
      if (emailContext === 'workbook') {
        localStorage.setItem("workbookSubmitted", "true");
      } else {
        localStorage.setItem("newsletterSubmitted", "true");
      }
      
      setShowEmailPrompt(false);
      
      // Send the email to the chat brain so it can give the correct follow-up response
      handleSend(trimmed, true);

    } catch (err) {
      console.error("Failed to register email:", err);
    }
  };

  const handleEmailReject = () => {
    setShowEmailPrompt(false);
    
    // Silently send a "no thanks" message to the backend
    handleSend("no thanks", true);
  };

  const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;
  const showOnboardingButtons =
    lastMessage &&
    lastMessage.type === "bot" &&
    lastMessage.text.includes("personalize your experience");

  return (
    <div id="chat-popup">
      <header className="chat-header">
        <div className="header-left">OmiBot | Omi Live</div>
        <div className="chat-header-right">
          <button
            className="new-chat-btn"
            onClick={() => {
              // Clear all memory for a truly new chat
              localStorage.removeItem("workbookSubmitted");
              localStorage.removeItem("newsletterSubmitted");
              window.location.reload();
            }}
          >
            New Chat
          </button>
          <button className="close-btn" onClick={onClose}>
            ❌
          </button>
        </div>
      </header>

      <main className="chat-shell">
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

        {showEmailPrompt && (
            <div className="email-prompt">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="example@domain.com"
                disabled={isLoading}
              />
              <button onClick={handleEmailSubmit} disabled={isLoading}>
                Submit
              </button>
              <button
                onClick={handleEmailReject}
                className="reject-btn"
                disabled={isLoading}
              >
                No thanks
              </button>
            </div>
          )}

        {showOnboardingButtons ? (
          <div className="onboarding-buttons">
            <button onClick={() => handleSend("Eco Shopper", true)}>Eco Shopper</button>
            <button onClick={() => handleSend("Creator", true)}>Creator</button>
            <button onClick={() => handleSend("Brand Owner", true)}>Brand Owner</button>
          </div>
        ) : (
          <div className="input-bar">
            <textarea
              ref={textareaRef}
              rows="1"
              placeholder={
                isLoading ? "OmiBot is thinking..." : "Ask me something..."
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !isLoading) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              disabled={isLoading || showEmailPrompt} // Disable while email prompt is shown
            />
            <button
              className="send-btn"
              onClick={() => handleSend()}
              disabled={isLoading || !input.trim() || showEmailPrompt}
            >
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
        )}
      </main>
    </div>
  );
}