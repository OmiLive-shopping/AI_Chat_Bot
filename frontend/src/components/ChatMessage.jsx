// frontend/src/components/ChatMessage.jsx
import React, { useEffect, useState } from "react";
import "./ChatMessage.css";

export default function ChatMessage({ type, text, loading }) {
  const [dots, setDots] = useState(".");

  const isThinking = type === "bot" && (loading || text === "OmiBot is thinking...");

  // Animate "..." when bot is thinking
  useEffect(() => {
    if (!isThinking) return;
    const interval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? "." : prev + "."));
    }, 500);
    return () => clearInterval(interval);
  }, [isThinking]);

  return (
    <div className={`msg-row ${type}`}>
      {type === "bot" && (
        <img
          src={isThinking ? "/omibot_thinking.jpg" : "/Omi_HeadShot.JPG"}
          alt="OmiBot"
          className="avatar"
        />
      )}

      <div className={`msg ${type}`}>
        {isThinking ? (
          <div className="thinking-container">
            <p className="thinking-text">
              OmiBot is thinking{dots}
            </p>
          </div>
        ) : (
          <span>{text}</span>
        )}
      </div>
    </div>
  );
}
