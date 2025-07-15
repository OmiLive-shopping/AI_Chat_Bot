// src/ChatOnly.jsx
import React from "react";
import ChatPopup from "./components/ChatPopup";

export default function ChatOnly() {
  return (
    <div style={{ height: "100vh", overflow: "hidden" }}>
      <ChatPopup onClose={() => {}} disableToggle={true} />
    </div>
  );
}
