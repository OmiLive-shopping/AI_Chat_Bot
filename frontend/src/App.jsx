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
        >
          💬
        </button>
      )}

      {isOpen && <ChatPopup onClose={() => setIsOpen(false)} />}
    </>
  );
}

// Critical fix: Export BOTH normally AND to window
export default App;

// This is what makes it available to your HTML file
if (typeof window !== 'undefined') {
  window.ChatbotApp = App; // Changed from YourRootComponent to App
}