import React, { useEffect, useState } from "react";

export default function ChatMessage({ type, text, loading }) {
  const [displayed, setDisplayed] = useState("");

  useEffect(() => {
    if (loading) {
      setDisplayed("...");
      return;
    }

    if (type !== "bot" || typeof text !== "string") {
      setDisplayed(text || "");
      return;
    }

    let index = 0;
    let buffer = ""; // collect characters safely

    const interval = setInterval(() => {
      buffer += text.charAt(index);
      setDisplayed(buffer);
      index++;
      if (index >= text.length) clearInterval(interval);
    }, 30);

    return () => clearInterval(interval);
  }, [text, type, loading]);

  return <div className={`msg ${type}`}>{displayed}</div>;
}
