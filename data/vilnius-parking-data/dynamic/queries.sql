-- Parameterised queries for the chatbot's SQL tools (use :named params).

-- 1. Free reservable spaces for a period.
--    Counts approved reservations that overlap [:start, :end).
SELECT p.id, p.name,
       p.bookable_spaces - COUNT(r.id) AS free_bookable_spaces
FROM parkings p
LEFT JOIN reservations r
       ON r.parking_id = p.id
      AND r.status = 'approved'
      AND r.start_time < :end
      AND r.end_time   > :start
WHERE p.id = :parking_id
GROUP BY p.id;
-- Note: this counts every overlapping reservation, which slightly underestimates
-- free spaces for long periods. The exact per-slot peak is computed in
-- scripts/seed_db.py -> free_spaces().

-- 2. Opening hours for a date (special hours override the weekly schedule).
--    :dow = ISO day of week of :date (1 = Monday).
SELECT COALESCE(s.open_time,  CASE WHEN s.parking_id IS NULL THEN w.open_time  END) AS open_time,
       COALESCE(s.close_time, CASE WHEN s.parking_id IS NULL THEN w.close_time END) AS close_time,
       COALESCE(s.is_closed, w.is_closed) AS is_closed,
       s.note
FROM working_hours w
LEFT JOIN special_hours s ON s.parking_id = w.parking_id AND s.date = :date
WHERE w.parking_id = :parking_id AND w.day_of_week = :dow;

-- 3. Current price for a date (:day_type = 'weekday' | 'weekend').
SELECT price_per_hour, daily_max, currency
FROM prices
WHERE parking_id = :parking_id
  AND day_type   = :day_type
  AND valid_from <= :date
  AND (valid_to IS NULL OR valid_to >= :date)
ORDER BY valid_from DESC
LIMIT 1;

-- 4. Pending requests for the admin (stage 2).
SELECT * FROM reservations WHERE status = 'pending' ORDER BY created_at;
