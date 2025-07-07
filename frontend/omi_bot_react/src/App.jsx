// src/App.jsx
import React, { useState } from "react";
import ChatPopup from "./components/ChatPopup";

export default function App() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {!isOpen && (
        <button
          id="chat-toggle"
          className="chat-toggle"
          onClick={() => setIsOpen(true)}
        >
          💬
        </button>
      )}

      {isOpen && <ChatPopup onClose={() => setIsOpen(false)} />}
    </>
  );
}
