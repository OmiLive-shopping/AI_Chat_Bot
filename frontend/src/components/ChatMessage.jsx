// frontend/src/components/ChatMessage.jsx
import React, { useEffect, useState } from "react";
import ReactMarkdown from 'react-markdown'; // --- NEW: Import the markdown renderer
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
          // --- MODIFIED: Replaced <span> with <ReactMarkdown> ---
          // This will automatically convert markdown links into clickable HTML links
          <ReactMarkdown
            components={{
              // This makes links open in a new tab for a better user experience
              a: ({node, ...props}) => <a {...props} target="_blank" rel="noopener noreferrer" />
            }}
          >
            {text}
          </ReactMarkdown>
        )}
      </div>
    </div>
  );
}