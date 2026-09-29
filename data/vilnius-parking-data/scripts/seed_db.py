"""Create and fill the VilniusPark SQLite database.

Usage:  uv run python scripts/seed_db.py [--db data/parking.db] [--days 14]

Loads dynamic/schema.sql and dynamic/seed.sql, then generates random but
deterministic reservations from today for --days days.
"""
from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FMT = "%Y-%m-%d %H:%M"

FIRST_NAMES = ["Jonas", "Tomas", "Lukas", "Mantas", "Paulius", "Andrius", "Rūta",
               "Eglė", "Ieva", "Gabija", "Austėja", "Laura", "Anna", "Mark", "Olga"]
SURNAMES = ["Kazlauskas", "Jankauskas", "Petrauskas", "Stankevičius", "Vasiliauskas",
            "Žukauskas", "Butkus", "Paulauskas", "Urbonas", "Kavaliauskas",
            "Smith", "Nowak", "Ivanova"]
PLATE_LETTERS = "ABCDEFGHJKLMNPRSTUVZ"

# Rough busyness: reservations per day per parking
DAILY_LOAD = {"old-town": (8, 20), "station": (5, 12), "business-center": (15, 35),
              "airport": (6, 14), "uzupis": (2, 6)}


def plate(rng: random.Random) -> str:
    return "".join(rng.choices(PLATE_LETTERS, k=3)) + " " + f"{rng.randint(0, 999):03d}"


def hours_for(conn: sqlite3.Connection, parking_id: str, d: date):
    """Return (open, close) datetimes for a date, or None if closed."""
    row = conn.execute(
        "SELECT open_time, close_time, is_closed FROM special_hours WHERE parking_id=? AND date=?",
        (parking_id, d.isoformat())).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT open_time, close_time, is_closed FROM working_hours WHERE parking_id=? AND day_of_week=?",
            (parking_id, d.isoweekday())).fetchone()
    open_t, close_t, closed = row
    if closed:
        return None

    def to_dt(t: str) -> datetime:
        h, m = map(int, t.split(":"))
        return datetime.combine(d, datetime.min.time()) + timedelta(hours=h, minutes=m)

    return to_dt(open_t), to_dt(close_t)


def generate_reservations(conn: sqlite3.Connection, days: int, seed: int = 42) -> int:
    rng = random.Random(seed)
    today = date.today()
    rows = []
    for pid, (lo, hi) in DAILY_LOAD.items():
        for offset in range(-2, days):  # a couple of past days for history
            d = today + timedelta(days=offset)
            hrs = hours_for(conn, pid, d)
            if hrs is None:
                continue
            open_dt, close_dt = hrs
            span_slots = int((close_dt - open_dt).total_seconds() // 1800)
            for _ in range(rng.randint(lo, hi)):
                if pid == "airport":
                    start = open_dt + timedelta(minutes=30 * rng.randrange(span_slots))
                    end = start + timedelta(days=rng.randint(1, 10))
                else:
                    dur = rng.choice([2, 2, 3, 4, 4, 6, 8, 10])  # in 30-min slots
                    if span_slots <= dur:
                        continue
                    start = open_dt + timedelta(minutes=30 * rng.randrange(span_slots - dur))
                    end = start + timedelta(minutes=30 * dur)
                if offset < 0:
                    status = rng.choices(["approved", "cancelled", "rejected"], [85, 10, 5])[0]
                else:
                    status = rng.choices(["approved", "pending", "cancelled", "rejected"],
                                         [70, 15, 10, 5])[0]
                name, surname = rng.choice(FIRST_NAMES), rng.choice(SURNAMES)
                comment = "No free reservable spaces" if status == "rejected" else None
                created = start - timedelta(hours=rng.randint(2, 240))
                rows.append((pid, name, surname, plate(rng),
                             f"{name.lower()}.{surname.lower()}@example.com",
                             start.strftime(FMT), end.strftime(FMT), status, comment,
                             created.strftime("%Y-%m-%d %H:%M:%S")))
    conn.executemany(
        """INSERT INTO reservations (parking_id, customer_name, customer_surname, car_plate,
               contact, start_time, end_time, status, admin_comment, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""", rows)
    return len(rows)


def free_spaces(conn: sqlite3.Connection, parking_id: str, start: str, end: str) -> int:
    """Minimum number of free reservable spaces over [start, end), checked per 30-min slot."""
    bookable = conn.execute("SELECT bookable_spaces FROM parkings WHERE id=?",
                            (parking_id,)).fetchone()[0]
    res = conn.execute(
        """SELECT start_time, end_time FROM reservations
           WHERE parking_id=? AND status='approved' AND start_time < ? AND end_time > ?""",
        (parking_id, end, start)).fetchall()
    t, t_end = datetime.strptime(start, FMT), datetime.strptime(end, FMT)
    peak = 0
    while t < t_end:
        s = t.strftime(FMT)
        peak = max(peak, sum(1 for a, b in res if a <= s < b))
        t += timedelta(minutes=30)
    return bookable - peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "parking.db"))
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "dynamic" / "schema.sql").read_text(encoding="utf-8"))
        conn.executescript((ROOT / "dynamic" / "seed.sql").read_text(encoding="utf-8"))
        n = generate_reservations(conn, args.days)
        conn.commit()
        print(f"Created {db_path} with {n} reservations")
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        for (pid,) in conn.execute("SELECT id FROM parkings"):
            print(f"  {pid:16} free tomorrow 09:00-12:00: "
                  f"{free_spaces(conn, pid, tomorrow + ' 09:00', tomorrow + ' 12:00')}")


if __name__ == "__main__":
    main()
