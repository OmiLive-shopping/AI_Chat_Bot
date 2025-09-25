// frontend/src/components/ChatPopup.jsx
import React, { useState, useRef, useEffect } from "react";
import ChatMessage from "./ChatMessage";

const BASE_URL = "https://omi-backend-355024965259.us-central1.run.app";

export default function ChatPopup({ onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [email, setEmail] = useState(localStorage.getItem("userEmail") || "");
  const [emailSubmitted, setEmailSubmitted] = useState(
    !!localStorage.getItem("userEmail")
  );
  const [sessionDismissed, setSessionDismissed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [userRequestCount, setUserRequestCount] = useState(0); // track user requests

  const chatBoxRef = useRef(null);
  const textareaRef = useRef(null);
  const endRef = useRef(null);
  const scrollTimeoutRef = useRef(null);
  const scrollIntervalRef = useRef(null);

  // Greeting on mount
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

  // Auto-resize input
  useEffect(() => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = "auto";
    textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
  }, [input]);

  // Focus input on mount
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  // Auto-scroll function
  const scrollToBottom = () => {
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }
    scrollTimeoutRef.current = setTimeout(() => {
      if (chatBoxRef.current) {
        chatBoxRef.current.scrollTo({
          top: chatBoxRef.current.scrollHeight,
          behavior: "smooth",
        });
      }
    }, 50);
  };

  // Continuous scrolling during typewriter effect
  const startContinuousScrolling = () => {
    stopContinuousScrolling();
    scrollIntervalRef.current = setInterval(() => {
      if (chatBoxRef.current) {
        chatBoxRef.current.scrollTo({
          top: chatBoxRef.current.scrollHeight,
          behavior: "smooth",
        });
      }
    }, 100);
  };

  const stopContinuousScrolling = () => {
    if (scrollIntervalRef.current) {
      clearInterval(scrollIntervalRef.current);
      scrollIntervalRef.current = null;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
      if (scrollIntervalRef.current) {
        clearInterval(scrollIntervalRef.current);
      }
    };
  }, []);

  const handleSend = async () => {
    const message = input.trim();
    if (!message || isLoading) return;

    setIsLoading(true);
    const newRequestCount = userRequestCount + 1;
    setUserRequestCount(newRequestCount);

    // Show user message + placeholder bot message
    setMessages((prev) => [
      ...prev,
      { type: "user", text: message },
      { type: "bot", text: "OmiBot is thinking...", loading: true },
    ]);
    setInput("");

    // Resize input after sending
    requestAnimationFrame(() => {
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
        textareaRef.current.style.height =
          textareaRef.current.scrollHeight + "px";
      }
    });

    try {
      const res = await fetch(`${BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      if (!res.body) {
        const data = await res.json();
        const finalAnswer = data.answer || "I don't know.";

        setMessages((prev) => {
          const filtered = prev.filter(
            (msg) => msg.text !== "OmiBot is thinking..."
          );
          return [...filtered, { type: "bot", text: finalAnswer, loading: false }];
        });

        // Newsletter prompt after 3rd request
        if (newRequestCount === 3 && !emailSubmitted && !sessionDismissed) {
          setMessages((prev) => [
            ...prev,
            {
              type: "bot",
              text: `💫 We're totally vibing! I'd love to keep this going - want to join our exclusive newsletter? You'll get early access to sustainable brand deals, new eco finds, and connect with our conscious shopping community.  

And if you're a brand or creator, I've got a free detailed live shopping workbook I can send you too! What's your email? 🌱`,
              loading: false,
            },
          ]);
        }

        setIsLoading(false);
        return;
      }

      const reader = res.body.getReader();
      let botMessage = "";

      // Replace placeholder with empty bot message for streaming
      setMessages((prev) => {
        const withoutThinking = prev.filter(
          (m) => m.text !== "OmiBot is thinking..."
        );
        return [
          ...withoutThinking,
          { type: "bot", text: "", loading: true, streaming: true },
        ];
      });

      startContinuousScrolling();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = new TextDecoder("utf-8").decode(value);
        botMessage += chunk;

        try {
          const jsonChunk = JSON.parse(botMessage);
          if (jsonChunk.answer) botMessage = jsonChunk.answer;
        } catch (e) {
          // ignore partial JSON
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

        scrollToBottom();
      }

      stopContinuousScrolling();
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
            streaming: false,
            loading: false,
          };
        }
        return updated;
      });

      // Newsletter prompt after 3rd request (for streamed responses)
      if (newRequestCount === 3 && !emailSubmitted && !sessionDismissed) {
        setMessages((prev) => [
          ...prev,
          {
            type: "bot",
            text: `💫 We're totally vibing! I'd love to keep this going - want to join our exclusive newsletter? You'll get early access to sustainable brand deals, new eco finds, and connect with our conscious shopping community.  

And if you're a brand or creator, I've got a free detailed live shopping workbook I can send you too! What's your email? 🌱`,
            loading: false,
          },
        ]);
      }
    } catch (err) {
      console.error("Fetch error:", err);
      setMessages((prev) => {
        const filtered = prev.filter(
          (msg) => msg.text !== "OmiBot is thinking..."
        );
        return [
          ...filtered,
          { type: "bot", text: `⚠️ Error: ${err.message}`, loading: false },
        ];
      });
      stopContinuousScrolling();
    } finally {
      setIsLoading(false);
      textareaRef.current?.focus();
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
      });
      localStorage.setItem("userEmail", trimmed);
      setEmailSubmitted(true);

      setMessages((prev) => [
        ...prev,
        {
          type: "bot",
          text: "🎉 Thanks! You're now part of the Omi community. Let's keep going!",
          loading: false,
        },
      ]);

      setTimeout(() => {
        textareaRef.current?.focus();
      }, 0);
    } catch (err) {
      console.error("Failed to register email:", err);
    }
  };

  const handleEmailReject = () => {
    setSessionDismissed(true);
    setMessages((prev) => [
      ...prev,
      {
        type: "bot",
        text: "👍 No worries! We'll keep chatting here.",
        loading: false,
      },
    ]);
    textareaRef.current?.focus();
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
              localStorage.removeItem("newsletterPrompted");
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
          <div ref={endRef} />
        </div>

        {/* Email input box */}
        {!emailSubmitted &&
          !sessionDismissed &&
          messages.some((m) => m.text.includes("What's your email?")) && (
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

        <div className="input-bar">
          <textarea
            ref={textareaRef}
            rows="1"
            placeholder={isLoading ? "OmiBot is thinking..." : "Ask me something..."}
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
            onClick={handleSend}
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
      </main>
    </div>
  );
}
