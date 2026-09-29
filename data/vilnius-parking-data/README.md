# VilniusPark – sample data for the parking reservation chatbot

A fictional parking operator with 5 car parks in Vilnius. Districts, streets and
landmarks are real, but the car parks, addresses and prices are made up, and
coordinates are approximate.

## Layout
```
static/                  -> embed into Milvus (RAG)
  general_info.md        company info, contacts, payment, FAQ
  booking_process.md     how to reserve, limits, cancellation
  rules.md               parking rules
  parkings/*.md          one doc per car park (location, access, facilities)
dynamic/                 -> SQL database
  schema.sql             tables: parkings, working_hours, special_hours, prices, reservations
  seed.sql               5 car parks, weekly hours, holidays, prices
  queries.sql            parameterised queries for the chatbot's SQL tools
scripts/seed_db.py       builds data/parking.db and generates ~900 reservations
```

## Linking the two stores
Each Markdown file has front matter with `parking_id` (`old-town`, `station`,
`business-center`, `airport`, `uzupis`, or `all`). Store it as chunk metadata in Milvus.
It matches `parkings.id` in SQL, so after retrieving a chunk you can look up
live hours, prices and availability for the same car park.

Static docs deliberately contain **no prices, hours or availability**. They
point the user to the assistant for those, so answers always come from SQL.

## Usage
```bash
uv run python scripts/seed_db.py            # -> data/parking.db
uv run python scripts/seed_db.py --days 30  # longer horizon
```
Reservations are generated relative to today with a fixed random seed.
Statuses are `pending | approved | rejected | cancelled`, which the admin agent (stage 2)
and the MCP server (stage 3) can use directly.
