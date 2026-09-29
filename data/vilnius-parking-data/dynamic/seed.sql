-- Coordinates are approximate points in the named Vilnius districts.
-- VilniusPark and its car parks are fictional.
INSERT INTO parkings (id, name, district, latitude, longitude, total_spaces, bookable_spaces) VALUES
 ('old-town',        'VilniusPark Old Town',          'Senamiestis', 54.6840, 25.2860, 180,  60),
 ('station',         'VilniusPark Station',           'Naujamiestis',54.6705, 25.2835, 120,  40),
 ('business-center', 'VilniusPark Business Center',   'Šnipiškės',   54.6960, 25.2770, 350, 120),
 ('airport',         'VilniusPark Airport Long-Stay', 'Kirtimai',    54.6385, 25.2850, 400, 200),
 ('uzupis',          'VilniusPark Užupis',            'Užupis',      54.6805, 25.2960,  40,  15);

-- Weekly hours (1=Mon .. 7=Sun)
INSERT INTO working_hours (parking_id, day_of_week, open_time, close_time, is_closed) VALUES
 ('old-town',1,'06:00','24:00',0),('old-town',2,'06:00','24:00',0),('old-town',3,'06:00','24:00',0),
 ('old-town',4,'06:00','24:00',0),('old-town',5,'06:00','24:00',0),('old-town',6,'06:00','24:00',0),
 ('old-town',7,'06:00','24:00',0),
 ('station',1,'00:00','24:00',0),('station',2,'00:00','24:00',0),('station',3,'00:00','24:00',0),
 ('station',4,'00:00','24:00',0),('station',5,'00:00','24:00',0),('station',6,'00:00','24:00',0),
 ('station',7,'00:00','24:00',0),
 ('business-center',1,'06:00','22:00',0),('business-center',2,'06:00','22:00',0),
 ('business-center',3,'06:00','22:00',0),('business-center',4,'06:00','22:00',0),
 ('business-center',5,'06:00','22:00',0),('business-center',6,'08:00','20:00',0),
 ('business-center',7,NULL,NULL,1),
 ('airport',1,'00:00','24:00',0),('airport',2,'00:00','24:00',0),('airport',3,'00:00','24:00',0),
 ('airport',4,'00:00','24:00',0),('airport',5,'00:00','24:00',0),('airport',6,'00:00','24:00',0),
 ('airport',7,'00:00','24:00',0),
 ('uzupis',1,'07:00','21:00',0),('uzupis',2,'07:00','21:00',0),('uzupis',3,'07:00','21:00',0),
 ('uzupis',4,'07:00','21:00',0),('uzupis',5,'07:00','21:00',0),('uzupis',6,'08:00','21:00',0),
 ('uzupis',7,NULL,NULL,1);

-- Lithuanian public holidays and maintenance
INSERT INTO special_hours (parking_id, date, open_time, close_time, is_closed, note) VALUES
 ('business-center','2026-11-01',NULL,NULL,1,'All Saints'' Day'),
 ('uzupis',         '2026-11-01',NULL,NULL,1,'All Saints'' Day'),
 ('old-town',       '2026-11-01','08:00','20:00',0,'All Saints'' Day – reduced hours'),
 ('old-town',       '2026-10-20','06:00','14:00',0,'Level -3 floor maintenance'),
 ('old-town',       '2026-12-24','06:00','18:00',0,'Christmas Eve – early closing'),
 ('business-center','2026-12-24',NULL,NULL,1,'Christmas Eve'),
 ('business-center','2026-12-25',NULL,NULL,1,'Christmas Day'),
 ('business-center','2026-12-26',NULL,NULL,1,'Second day of Christmas'),
 ('uzupis',         '2026-12-24','08:00','15:00',0,'Christmas Eve – early closing'),
 ('uzupis',         '2026-12-25',NULL,NULL,1,'Christmas Day'),
 ('uzupis',         '2026-12-26',NULL,NULL,1,'Second day of Christmas'),
 ('business-center','2027-01-01',NULL,NULL,1,'New Year''s Day'),
 ('uzupis',         '2027-01-01',NULL,NULL,1,'New Year''s Day');

-- Prices. Includes one expired row so "current price" queries must filter on dates.
INSERT INTO prices (parking_id, day_type, price_per_hour, daily_max, currency, valid_from, valid_to) VALUES
 ('old-town','weekday',2.80,24.00,'EUR','2026-01-01','2026-06-30'),
 ('old-town','weekday',3.00,25.00,'EUR','2026-07-01',NULL),
 ('old-town','weekend',2.50,20.00,'EUR','2026-01-01',NULL),
 ('station','weekday',2.00,15.00,'EUR','2026-01-01',NULL),
 ('station','weekend',2.00,15.00,'EUR','2026-01-01',NULL),
 ('business-center','weekday',2.50,18.00,'EUR','2026-01-01',NULL),
 ('business-center','weekend',1.50,10.00,'EUR','2026-01-01',NULL),
 ('airport','weekday',1.50,12.00,'EUR','2026-01-01',NULL),
 ('airport','weekend',1.50,12.00,'EUR','2026-01-01',NULL),
 ('uzupis','weekday',1.80,12.00,'EUR','2026-01-01',NULL),
 ('uzupis','weekend',1.50,10.00,'EUR','2026-01-01',NULL);
