import React, { useState, useRef, useEffect } from "react";
import ChatMessage from "./ChatMessage";

const BASE_URL = "https://omi-backend-355024965259.us-central1.run.app";

// Generate or retrieve persistent session ID
const getSessionId = () => {
  let sessionId = localStorage.getItem("omiSessionId");
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem("omiSessionId", sessionId);
  }
  return sessionId;
};

export default function ChatPopup({ onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [email, setEmail] = useState(localStorage.getItem("userEmail") || "");
  const [emailSubmitted, setEmailSubmitted] = useState(
    !!localStorage.getItem("userEmail")
  );
  const [sessionDismissed, setSessionDismissed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);

  const chatBoxRef = useRef(null);
  const textareaRef = useRef(null);
  const scrollIntervalRef = useRef(null);

  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          type: "bot",
          text: `Hi there! I'm Omi, your eco-friendly shopping companion! 🌱✨ I'm here to help you discover sustainable brands, learn eco tips, and master live shopping - whether you're a conscious shopper or a creator ready to go live! Ask me about: 🛍️ Sustainable shopping & green living tips 📱 Live shopping experiences & authentic brand connections 🌿 Eco-friendly brands & sustainability insights 🎯 Creator resources - Get our free step-by-step live shopping workbook! Ready to chat about conscious commerce? What can I help you with today? 🎉`,
        },
        {
          type: "bot",
          text: "To personalize your experience, please let me know who you are.",
        },
      ]);
    }
  }, []);

  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    if (messages.length === 2 && lastMessage?.type === "bot") {
      handleSend("__GET_ONBOARDING__", true, { suppressThinking: true });
      setShowOnboarding(true);
    }
  }, [messages.length]);

  useEffect(() => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = "auto";
    textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
  }, [input]);

  useEffect(() => {
    if (textareaRef.current && !showOnboarding) {
      textareaRef.current.focus();
    }
  }, [isLoading, messages, showOnboarding]);

  const stopContinuousScrolling = () => {
    if (scrollIntervalRef.current) {
      clearInterval(scrollIntervalRef.current);
      scrollIntervalRef.current = null;
    }
  };

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTo({
        top: chatBoxRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
    return () => stopContinuousScrolling();
  }, [messages]);

  const handleSend = async (messageOverride, isSilent = false, opts = {}) => {
    const message =
      typeof messageOverride === "string" ? messageOverride : input.trim();
    if (!message || isLoading) return;

    const sessionId = getSessionId();
    const isOnboardingPing = message === "__GET_ONBOARDING__";

    setIsLoading(true);

    if (!isSilent && !isOnboardingPing) {
      setMessages((prev) => [...prev, { type: "user", text: message }]);
    }
    setInput("");

    if (!opts.suppressThinking && !isOnboardingPing) {
      setMessages((prev) => [
        ...prev,
        { type: "bot", text: "OmiBot is thinking...", loading: true },
      ]);
      scrollIntervalRef.current = setInterval(() => {
        if (chatBoxRef.current) {
          chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
        }
      }, 100);
    }

    try {
      const res = await fetch(`${BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId }),
        credentials: "include",
      });

      if (isOnboardingPing) {
        setShowOnboarding(true);
        setIsLoading(false);
        return;
      }
            if (!res.body) {
        const data = await res.json().catch(() => ({}));
        const finalAnswer =
          (data && (data.answer || data.data || data.response)) ||
          "I'm having a little trouble right now.";
        setMessages((prev) => {
          const updated = prev.filter(
            (msg) => msg.text !== "OmiBot is thinking..."
          );
          return [
            ...updated,
            { type: "bot", text: finalAnswer, loading: false, streaming: false },
          ];
        });
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let botMessage = "";

      setMessages((prev) => {
        const updated = prev.filter((msg) => msg.text !== "OmiBot is thinking...");
        return [
          ...updated,
          { type: "bot", text: "", loading: true, streaming: true },
        ];
      });

      let fullChunk = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        fullChunk += decoder.decode(value, { stream: true });

        try {
          const parsed = JSON.parse(fullChunk);
          if (parsed.answer) {
            botMessage = parsed.answer;
          } else if (parsed.data) {
            botMessage = parsed.data;
          } else {
            botMessage = fullChunk;
          }
        } catch (e) {
          botMessage = fullChunk;
        }

        setMessages((prev) => {
          const updated = [...prev];
          const lastIndex = updated.length - 1;
          if (
            lastIndex >= 0 &&
            updated[lastIndex].type === "bot" &&
            updated[lastIndex].streaming
          ) {
            updated[lastIndex] = {
              ...updated[lastIndex],
              text: botMessage,
              loading: true,
            };
          }
          return updated;
        });
      }
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
            streaming: false,
          },
        ];
      });
    } finally {
      stopContinuousScrolling();
      setMessages((prev) => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        if (lastIndex >= 0 && updated[lastIndex].type === "bot") {
          updated[lastIndex].loading = false;
          updated[lastIndex].streaming = false;
        }
        return updated;
      });
      setIsLoading(false);
    }
  };

  const handleEmailSubmit = async () => {
    const trimmed = email.trim();
    const isValid = /\S+@\S+\.\S+/.test(trimmed);
    if (!isValid) return alert("Please enter a valid email");

    try {
      await fetch(`${BASE_URL}/register-email`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: trimmed }),
        credentials: "include",
      });
      localStorage.setItem("userEmail", trimmed);
      setEmailSubmitted(true);
      handleSend(trimmed, true, { suppressThinking: false });
    } catch (err) {
      console.error("Failed to register email:", err);
    }
  };

  const handleEmailReject = () => {
    setSessionDismissed(true);
    handleSend("no thanks", true, { suppressThinking: false });
  };

  return (
    <div id="chat-popup">
      <header className="chat-header">
        <div className="header-left">OmiBot | Omi Live</div>
        <div className="chat-header-right">
          <button
            className="new-chat-btn"
            onClick={() => {
              localStorage.removeItem("userEmail");
              localStorage.removeItem("omiSessionId");
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

        {!emailSubmitted &&
          !sessionDismissed &&
          messages.some((m) =>
            /drop your email|send.*workbook|what'?s your email/i.test(m.text)
          ) && (
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

        {showOnboarding ? (
          <div className="onboarding-section">
            <div className="onboarding-buttons">
              <button
                onClick={() => {
                  handleSend("Eco Shopper", true, { suppressThinking: false });
                  setShowOnboarding(false);
                }}
              >
                Eco Shopper
              </button>
              <button
                onClick={() => {
                  handleSend("Creator", true, { suppressThinking: false });
                  setShowOnboarding(false);
                }}
              >
                Creator
              </button>
              <button
                onClick={() => {
                  handleSend("Brand Owner", true, { suppressThinking: false });
                  setShowOnboarding(false);
                }}
              >
                Brand Owner
              </button>
            </div>
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
              disabled={isLoading}
            />
            <button
              className="send-btn"
              onClick={() => handleSend()}
              disabled={isLoading || !input.trim()}
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
