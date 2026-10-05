"use strict";

const ADMIN_KEY_STORAGE_KEY = "vilniuspark.adminKey";

function initializeAdmin() {
  const keyForm = document.getElementById("key-form");
  const keyInput = document.getElementById("admin-key");
  const refreshButton = document.getElementById("refresh-button");
  const statusElement = document.getElementById("status");
  const reservationsSection = document.getElementById("reservations");
  const template = document.getElementById("reservation-template");

  const requiredElements = {
    keyForm,
    keyInput,
    refreshButton,
    statusElement,
    reservationsSection,
    template
  };

  const missingElements = Object.entries(requiredElements)
    .filter(([, element]) => !element)
    .map(([name]) => name);

  if (missingElements.length > 0) {
    console.error(
      "Admin console initialization failed. Missing elements:",
      missingElements
    );
    return;
  }

  keyInput.value = window.localStorage.getItem(ADMIN_KEY_STORAGE_KEY) || "";

  function adminKey() {
    return keyInput.value.trim();
  }

  function setStatus(text) {
    statusElement.textContent = text;
  }

  async function apiRequest(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: {
        ...options.headers,
        "X-Admin-Key": adminKey()
      }
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => null);

      throw new Error(
        typeof payload?.detail === "string"
          ? payload.detail
          : `Request failed (${response.status}).`
      );
    }

    if (response.status === 204) {
      return null;
    }

    return response.json();
  }

  function renderReservations(reservations) {
    reservationsSection.replaceChildren();

    if (reservations.length === 0) {
      const empty = document.createElement("p");
      empty.className = "notice";
      empty.textContent = "No reservations are waiting for approval.";
      reservationsSection.appendChild(empty);
      return;
    }

    for (const reservation of reservations) {
      const fragment = template.content.cloneNode(true);
      const article = fragment.querySelector(".reservation");

      article.querySelector(".reservation-message").textContent =
        reservation.message || "(no drafted message yet)";

      article.querySelector(".reservation-meta").textContent =
        `${reservation.first_name} ${reservation.surname} · car ${reservation.car_number} · ` +
        `${reservation.starts_at} to ${reservation.ends_at} · ref ${reservation.id}`;

      const decide = async (decision, button) => {
        const buttons = article.querySelectorAll("button");

        for (const element of buttons) {
          element.disabled = true;
        }

        setStatus(`Submitting ${decision} decision…`);

        try {
          await apiRequest(`/admin/reservations/${reservation.id}/decision`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({decision})
          });

          setStatus(`Reservation ${decision}.`);
          await loadReservations();
        } catch (error) {
          setStatus(error.message || "Could not submit the decision.");

          for (const element of buttons) {
            element.disabled = false;
          }
        }
      };

      article
        .querySelector(".approve-button")
        .addEventListener("click", (event) => void decide("approved", event.target));

      article
        .querySelector(".reject-button")
        .addEventListener("click", (event) => void decide("rejected", event.target));

      reservationsSection.appendChild(fragment);
    }
  }

  async function loadReservations() {
    if (!adminKey()) {
      setStatus("Enter the admin key to load pending reservations.");
      reservationsSection.replaceChildren();
      return;
    }

    setStatus("Loading…");

    try {
      const reservations = await apiRequest("/admin/reservations");
      renderReservations(reservations);
      setStatus(`${reservations.length} pending.`);
    } catch (error) {
      setStatus(error.message || "Could not load pending reservations.");
    }
  }

  keyForm.addEventListener("submit", (event) => {
    event.preventDefault();
    window.localStorage.setItem(ADMIN_KEY_STORAGE_KEY, adminKey());
    void loadReservations();
  });

  refreshButton.addEventListener("click", () => void loadReservations());

  void loadReservations();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeAdmin, {once: true});
} else {
  initializeAdmin();
}
