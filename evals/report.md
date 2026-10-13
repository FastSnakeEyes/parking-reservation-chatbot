# RAG evaluation report

Golden set size: 15 questions, K=4

## Retrieval accuracy

- Mean Precision@4: 0.58
- Mean Recall@4: 0.93

## Retrieval latency

- Mean: 598.0 ms
- p95: 807.5 ms

## End-to-end /chat latency

- Samples: 5
- Mean: 3660.3 ms
- p95: 6840.0 ms

## Method

- Relevance is labelled per source document (`golden_qa.json`); the static docs are
  split into chunks, so one document can fill several of the K results.
- **Precision@4**: share of the K retrieved chunks that come from a relevant document.
- **Recall@4**: share of the relevant documents with at least one chunk in the top K.
- Retrieval latency includes embedding the question through the OpenAI embeddings API,
  which dominates it; the vector search itself is a few milliseconds.
- End-to-end /chat latency is a real chat turn (agent, tools, model calls), spaced to stay
  under the per-session rate limit.

## Per-question detail

| Question | Precision | Recall | Latency (ms) | Retrieved sources |
| --- | --- | --- | --- | --- |
| What payment methods are accepted at VilniusPark car parks? | 0.50 | 1.00 | 781.7 | general_info.md, general_info.md, booking_process.md, rules.md |
| Can I park a motorcycle at a VilniusPark car park? | 0.50 | 1.00 | 592.9 | rules.md, general_info.md, booking_process.md, general_info.md |
| Do disability parking card holders get free parking? | 0.50 | 1.00 | 597.8 | parkings/old_town.md, parkings/uzupis.md, general_info.md, general_info.md |
| What are the steps to reserve a parking space? | 0.75 | 1.00 | 398.8 | booking_process.md, booking_process.md, booking_process.md, general_info.md |
| What is the minimum and maximum length of a reservation? | 0.75 | 1.00 | 593.6 | booking_process.md, parkings/airport.md, booking_process.md, booking_process.md |
| Can I cancel my parking reservation for free? | 0.75 | 1.00 | 404.4 | booking_process.md, booking_process.md, general_info.md, booking_process.md |
| Why might a parking reservation be rejected? | 1.00 | 1.00 | 803.2 | booking_process.md, booking_process.md, booking_process.md, booking_process.md |
| What is the maximum driving speed inside the car parks? | 0.25 | 1.00 | 601.9 | rules.md, booking_process.md, parkings/business_center.md, general_info.md |
| What happens if I lose my entry ticket? | 0.00 | 0.00 | 597.7 | booking_process.md, booking_process.md, booking_process.md, parkings/old_town.md |
| Are LPG vehicles allowed in the Old Town car park? | 1.00 | 1.00 | 601.5 | rules.md, parkings/old_town.md, parkings/old_town.md, parkings/old_town.md |
| What is the height limit at the Old Town car park? | 0.75 | 1.00 | 593.8 | rules.md, parkings/old_town.md, parkings/old_town.md, parkings/old_town.md |
| How many EV charging bays does the Old Town car park have? | 0.25 | 1.00 | 605.8 | parkings/airport.md, parkings/business_center.md, parkings/uzupis.md, parkings/old_town.md |
| What facilities does the Business Center car park have? | 0.75 | 1.00 | 599.9 | parkings/business_center.md, parkings/business_center.md, general_info.md, parkings/business_center.md |
| Where is the Užupis car park located and how big is it? | 0.50 | 1.00 | 598.8 | parkings/uzupis.md, general_info.md, parkings/uzupis.md, general_info.md |
| Is there a shuttle between the Airport car park and the terminal? | 0.50 | 1.00 | 598.2 | parkings/airport.md, parkings/airport.md, general_info.md, booking_process.md |
