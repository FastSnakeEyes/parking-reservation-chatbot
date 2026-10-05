"use strict";

function initializeChat() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message-input");
  const sendButton = document.getElementById("send-button");
  const resetButton = document.getElementById("reset-button");
  const messages = document.getElementById("messages");
  const statusElement = document.getElementById("status");
  const suggestions = document.querySelectorAll(".suggestion");

  const requiredElements = {
    form,
    input,
    sendButton,
    resetButton,
    messages,
    statusElement
  };

  const missingElements = Object.entries(requiredElements)
    .filter(([, element]) => !element)
    .map(([name]) => name);

  if (missingElements.length > 0) {
    console.error(
      "Chat initialization failed. Missing elements:",
      missingElements
    );
    return;
  }

  let busy = false;

  function setBusy(value) {
    busy = value;

    input.disabled = value;
    sendButton.disabled = value;
    resetButton.disabled = value;

    for (const button of suggestions) {
      button.disabled = value;
    }

    messages.setAttribute("aria-busy", String(value));
    statusElement.textContent = value
      ? "Assistant is processing your message…"
      : "";
  }

  function appendMessage(role, text, sources = []) {
    const article = document.createElement("article");
    article.className = `message ${role}`;

    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent =
      role === "user" ? "You" :
      role === "error" ? "Request error" :
      "Assistant";

    const body = document.createElement("div");
    body.className = "message-text";

    // Render model output as text, never as HTML.
    body.textContent = text;

    article.append(label, body);

    if (Array.isArray(sources) && sources.length > 0) {
      const details = document.createElement("details");
      details.className = "sources";

      const summary = document.createElement("summary");
      summary.textContent = "Public information sources";

      const list = document.createElement("ul");

      for (const source of sources) {
        const item = document.createElement("li");
        const chunk = source.chunk_index == null
          ? ""
          : ` · chunk ${source.chunk_index}`;

        item.textContent =
          `[${source.label}] ${source.source}${chunk}`;

        list.appendChild(item);
      }

      details.append(summary, list);
      article.appendChild(details);
    }

    messages.appendChild(article);
    messages.scrollTop = messages.scrollHeight;
  }

  async function requestChat(question) {
    const controller = new AbortController();
    const timeout = window.setTimeout(
      () => controller.abort(),
      135000
    );

    try {
      const response = await fetch("/chat", {
        method: "POST",
        credentials: "same-origin",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({question}),
        signal: controller.signal
      });

      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          typeof payload?.detail === "string"
            ? payload.detail
            : `Chat request failed (${response.status}).`
        );
      }

      if (!payload || typeof payload.answer !== "string") {
        throw new Error(
          "The server returned an invalid chat response."
        );
      }

      return payload;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function showError(error) {
    const text = error.name === "AbortError"
      ? (
        "The request timed out and may still have been processed. " +
        "If you were confirming a reviewed reservation, " +
        "send confirm again to retrieve or complete that request."
      )
      : error.message || "Could not reach the server.";

    appendMessage("error", text);
  }

  async function sendMessage(value) {
    const question = value.trim();

    if (busy || !question) {
      return;
    }

    if (question.length > 4000) {
      statusElement.textContent =
        "Please keep your message under 4,000 characters.";
      return;
    }

    appendMessage("user", question);
    input.value = "";
    setBusy(true);

    try {
      const result = await requestChat(question);
      appendMessage(
        "assistant",
        result.answer,
        result.sources
      );
    } catch (error) {
      showError(error);
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void sendMessage(input.value);
  });

  input.addEventListener("keydown", (event) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.isComposing
    ) {
      event.preventDefault();

      if (!busy) {
        form.requestSubmit();
      }
    }
  });

  for (const button of suggestions) {
    button.addEventListener("click", () => {
      void sendMessage(button.dataset.message || "");
    });
  }

  resetButton.addEventListener("click", async () => {
    if (busy) {
      return;
    }

    setBusy(true);

    try {
      const result = await requestChat("cancel");

      // Clear the transcript only after the backend clears its state.
      messages.replaceChildren();
      input.value = "";

      appendMessage(
        "assistant",
        result.answer + "\nHow can I help you?"
      );
    } catch (error) {
      showError(error);
    } finally {
      setBusy(false);
      input.focus();
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener(
    "DOMContentLoaded",
    initializeChat,
    {once: true}
  );
} else {
  initializeChat();
}