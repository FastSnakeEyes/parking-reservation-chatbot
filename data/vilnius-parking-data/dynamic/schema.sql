-- VilniusPark dynamic data (SQLite; also works in Postgres with minor type changes)

CREATE TABLE IF NOT EXISTS parkings (
    id               TEXT PRIMARY KEY,          -- same id as `parking_id` in the static docs
    name             TEXT NOT NULL,
    district         TEXT NOT NULL,
    latitude         REAL NOT NULL,
    longitude        REAL NOT NULL,
    total_spaces     INTEGER NOT NULL,
    bookable_spaces  INTEGER NOT NULL,          -- spaces that can be reserved; the rest are drive-in only
    CHECK (bookable_spaces <= total_spaces)
);

-- Regular weekly hours. day_of_week: 1 = Monday ... 7 = Sunday (ISO).
-- 24h operation is stored as 00:00-24:00.
CREATE TABLE IF NOT EXISTS working_hours (
    parking_id   TEXT NOT NULL REFERENCES parkings(id),
    day_of_week  INTEGER NOT NULL CHECK (day_of_week BETWEEN 1 AND 7),
    open_time    TEXT,                          -- 'HH:MM', NULL when closed
    close_time   TEXT,
    is_closed    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (parking_id, day_of_week)
);

-- Exceptions (public holidays, maintenance). Overrides working_hours for that date.
CREATE TABLE IF NOT EXISTS special_hours (
    parking_id  TEXT NOT NULL REFERENCES parkings(id),
    date        TEXT NOT NULL,                  -- 'YYYY-MM-DD'
    open_time   TEXT,
    close_time  TEXT,
    is_closed   INTEGER NOT NULL DEFAULT 0,
    note        TEXT,
    PRIMARY KEY (parking_id, date)
);

CREATE TABLE IF NOT EXISTS prices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    parking_id      TEXT NOT NULL REFERENCES parkings(id),
    day_type        TEXT NOT NULL CHECK (day_type IN ('weekday', 'weekend')),
    price_per_hour  REAL NOT NULL,
    daily_max       REAL NOT NULL,
    currency        TEXT NOT NULL DEFAULT 'EUR',
    valid_from      TEXT NOT NULL,              -- 'YYYY-MM-DD'
    valid_to        TEXT                        -- NULL = still valid
);

CREATE TABLE IF NOT EXISTS reservations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    parking_id        TEXT NOT NULL REFERENCES parkings(id),
    customer_name     TEXT NOT NULL,
    customer_surname  TEXT NOT NULL,
    car_plate         TEXT NOT NULL,
    contact           TEXT,
    start_time        TEXT NOT NULL,            -- 'YYYY-MM-DD HH:MM'
    end_time          TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending', 'approved', 'rejected', 'cancelled')),
    admin_comment     TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (end_time > start_time)
);

CREATE INDEX IF NOT EXISTS idx_res_parking_time
    ON reservations (parking_id, start_time, end_time);

-- Reservable spaces free right now
CREATE VIEW IF NOT EXISTS current_availability AS
SELECT p.id AS parking_id,
       p.name,
       p.bookable_spaces,
       p.bookable_spaces - COUNT(r.id) AS free_bookable_spaces
FROM parkings p
LEFT JOIN reservations r
       ON r.parking_id = p.id
      AND r.status = 'approved'
      AND r.start_time <= strftime('%Y-%m-%d %H:%M', 'now', 'localtime')
      AND r.end_time   >  strftime('%Y-%m-%d %H:%M', 'now', 'localtime')
GROUP BY p.id;
