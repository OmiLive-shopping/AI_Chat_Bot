import React, { useState, useEffect } from "react";
import ChatPopup from "./components/ChatPopup";

function App() {
  const [isOpen, setIsOpen] = useState(false);

  // Add this effect to sync with Wix
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.toggleChatExternal = setIsOpen;
    }
  }, []);

  return (
    <>
      {!isOpen && (
        <button
          id="chat-toggle"
          className="chat-toggle"
          onClick={() => {
            setIsOpen(true);
            if (window.toggleChat) window.toggleChat(true);
          }}
        >
          💬
        </button>
      )}

      {isOpen && (
        <ChatPopup 
          onClose={() => {
            setIsOpen(false);
            if (window.toggleChat) window.toggleChat(false);
          }} 
        />
      )}
    </>
  );
}

export default App;

if (typeof window !== 'undefined') {
  window.ChatbotApp = App;
}