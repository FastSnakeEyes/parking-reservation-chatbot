"""app/ai/middleware.py's date stamp for the user agent's prompt."""

from datetime import datetime
from zoneinfo import ZoneInfo

from app.ai.middleware import current_time_text


def test_the_prompt_spells_out_the_coming_week():
    saturday = datetime(2026, 10, 17, 9, 30, tzinfo=ZoneInfo("Europe/Vilnius"))

    text = current_time_text(saturday)

    assert "Saturday 2026-10-17 09:30" in text
    # "Wednesday" is looked up, not computed: the next one is the 21st.
    assert "Wednesday 2026-10-21" in text
    assert "Sunday 2026-10-18" in text and "Saturday 2026-10-24" in text
