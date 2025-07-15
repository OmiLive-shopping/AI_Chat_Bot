// src/main.jsx
import React from "react";
import ReactDOM from "react-dom/client";
import ChatOnly from "./ChatOnly"; // 👈 replace App with ChatOnly
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ChatOnly />
  </React.StrictMode>
);
