"use strict";

function initializeChat() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message-input");
  const sendButton = document.getElementById("send-button");
  const resetButton = document.getElementById("reset-button");
  const messages = document.getElementById("messages");
  const statusElement = document.getElementById("status");
  const suggestions = document.querySelectorAll(".suggestion");
  const decisionDialog = document.getElementById("decision-dialog");
  const decisionTitle = document.getElementById("decision-title");
  const decisionText = document.getElementById("decision-text");

  // How often the page asks whether the administrator has decided.
  const UPDATES_POLL_MS = 5000;

  const requiredElements = {
    form,
    input,
    sendButton,
    resetButton,
    messages,
    statusElement,
    decisionDialog,
    decisionTitle,
    decisionText
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

  // Confirm / Cancel under a reservation review. Any later message makes the
  // review stale (the backend drops its draft), so the buttons are removed then.
  function removeReviewActions() {
    for (const actions of messages.querySelectorAll(".review-actions")) {
      actions.remove();
    }
  }

  function appendReviewActions() {
    const article = messages.lastElementChild;

    const actions = document.createElement("div");
    actions.className = "review-actions";

    const confirmButton = document.createElement("button");
    confirmButton.type = "button";
    confirmButton.className = "primary-button";
    confirmButton.textContent = "Confirm";
    confirmButton.addEventListener("click", () => void sendMessage("confirm"));

    const cancelButton = document.createElement("button");
    cancelButton.type = "button";
    cancelButton.className = "secondary-button";
    cancelButton.textContent = "Cancel";
    cancelButton.addEventListener("click", () => void clearConversation());

    actions.append(confirmButton, cancelButton);
    article.appendChild(actions);
    messages.scrollTop = messages.scrollHeight;
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

  const DECISION_TITLES = {
    approved: "Reservation approved",
    rejected: "Reservation rejected"
  };

  function showDecision(notice, reservationStatus) {
    decisionDialog.classList.remove("approved", "rejected");

    if (reservationStatus === "approved") {
      decisionDialog.classList.add("approved");
    } else if (reservationStatus === "rejected") {
      decisionDialog.classList.add("rejected");
    }

    decisionTitle.textContent =
      DECISION_TITLES[reservationStatus] || "Reservation update";
    // Server text, rendered as text - never as HTML.
    decisionText.textContent = notice;

    if (!decisionDialog.open) {
      decisionDialog.showModal();
    }
  }

  let pollTimer = null;

  function schedulePoll() {
    if (pollTimer === null) {
      pollTimer = window.setTimeout(pollUpdates, UPDATES_POLL_MS);
    }
  }

  // While a reservation awaits approval, ask every few seconds whether the
  // administrator has decided; show the outcome as soon as it arrives.
  async function pollUpdates() {
    pollTimer = null;

    try {
      const response = await fetch("/chat/updates", {
        credentials: "same-origin",
        cache: "no-store"
      });

      if (!response.ok) {
        // e.g. no cookie yet - nothing to wait for.
        return;
      }

      const update = await response.json();

      if (typeof update.notice === "string") {
        appendMessage("assistant", update.notice);
        showDecision(update.notice, update.reservation_status);
        return;
      }

      if (update.status === "waiting_for_admin") {
        schedulePoll();
      }
    } catch {
      // Network hiccup: keep waiting rather than give up.
      schedulePoll();
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

    removeReviewActions();
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

      if (result.awaiting_confirmation) {
        appendReviewActions();
      }

      if (result.status === "waiting_for_admin") {
        schedulePoll();
      }
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

  // A decision may have arrived while the page was closed or reloading.
  void pollUpdates();

  async function clearConversation() {
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
  }

  resetButton.addEventListener("click", () => void clearConversation());
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