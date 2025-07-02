// src/components/ChatMessage.jsx
import React from "react";

export default function ChatMessage({ type, text }) {
  return (
    <div className={`msg ${type}`}>
      {text}
    </div>
  );
}
