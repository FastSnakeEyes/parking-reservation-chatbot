"""Stage 2: the administrator's console for approving/rejecting reservations.

Run with: uv run streamlit run ui/admin_app.py
Talks only to the app's /admin REST API (app/api/routers/admin.py); it
holds no direct DB connection of its own.
"""

import os
from datetime import datetime

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

st.set_page_config(page_title="Parking admin console", page_icon="🅿️")
st.title("Pending reservation requests")


def fetch_pending() -> list[dict]:
    response = httpx.get(
        f"{API_BASE_URL}/admin/reservations",
        headers={"X-Admin-Key": ADMIN_API_KEY},
        timeout=10,
    )
    response.raise_for_status()

    return response.json()


def send_decision(reservation_id: str, decision: str) -> None:
    response = httpx.post(
        f"{API_BASE_URL}/admin/reservations/{reservation_id}/decision",
        headers={"X-Admin-Key": ADMIN_API_KEY},
        json={"decision": decision},
        timeout=10,
    )
    response.raise_for_status()


if st.button("Refresh"):
    st.rerun()

if not ADMIN_API_KEY:
    st.error("Set ADMIN_API_KEY before running this console.")
    st.stop()

try:
    pending = fetch_pending()
except httpx.HTTPStatusError as exc:
    st.error(f"Could not load pending reservations: {exc.response.status_code} {exc.response.text}")
    st.stop()
except httpx.HTTPError as exc:
    st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
    st.stop()

st.caption(f"Last updated {datetime.now().strftime('%H:%M:%S')} - {len(pending)} pending")

if not pending:
    st.info("No reservations are waiting for approval.")

for reservation in pending:
    with st.container(border=True):
        st.markdown(reservation.get("message") or "*(no drafted message yet)*")

        st.caption(
            f"Reservation {reservation['id']} - {reservation['first_name']} {reservation['surname']} - "
            f"car {reservation['car_number']} - {reservation['starts_at']} to {reservation['ends_at']}"
        )

        approve_col, reject_col = st.columns(2)

        if approve_col.button("Approve", key=f"approve-{reservation['id']}"):
            send_decision(reservation["id"], "approved")
            st.rerun()

        if reject_col.button("Reject", key=f"reject-{reservation['id']}"):
            send_decision(reservation["id"], "rejected")
            st.rerun()
