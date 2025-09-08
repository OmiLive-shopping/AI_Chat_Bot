// src/App.jsx
import React, { useState } from "react";
import ChatPopup from "./components/ChatPopup";

function App() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {!isOpen && (
        <button
          id="chat-toggle"
          className="chat-toggle"
          onClick={() => setIsOpen(true)}
          aria-label="Open OmiBot Chat"
        >
          <img
            src="/omibot_thinking.jpg"
            alt="Open Chat"
            style={{
              width: "50px",
              height: "50px",
              borderRadius: "50%",
              objectFit: "cover",
              boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
            }}
          />
        </button>
      )}

      {isOpen && <ChatPopup onClose={() => setIsOpen(false)} />}
    </>
  );
}

export default App;

// Attach to window only if running in browser (for embedding)
if (typeof window !== "undefined") {
  window.ChatbotApp = App;
}
