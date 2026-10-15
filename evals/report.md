# RAG evaluation report

Golden set size: 15 questions, K=3

## Retrieval accuracy

- Mean Precision@3: 0.64
- Mean Recall@3: 1.00
- MRR@3: 0.92 (1.00 = the relevant doc is always ranked first)
- Precision@1: 0.87 (share of questions whose top chunk is relevant)

## Choosing K

| K | Precision@K | Recall@K | MRR@K | Precision@1 | Mean latency (ms) |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.73 | 0.93 | 0.90 | 0.87 | 546.5 |
| 3 (current, recommended) | 0.64 | 1.00 | 0.92 | 0.87 | 573.9 |
| 4 | 0.57 | 1.00 | 0.92 | 0.87 | 599.8 |
| 5 | 0.57 | 1.00 | 0.92 | 0.87 | 573.1 |

Recommended K=3: the smallest K with the best recall. A larger K only adds chunks the model has to read past (Precision@K falls), and a smaller one misses answers. RETRIEVER_K already uses it.

## Retrieval latency

- Mean: 611.2 ms
- p95: 1268.8 ms

## Method

- Relevance is labelled per source document (`golden_qa.json`); the static docs are
  split into chunks, so one document can fill several of the K results.
- **Precision@3**: share of the K retrieved chunks that come from a relevant document.
- **Recall@3**: share of the relevant documents with at least one chunk in the top K.
- **MRR** (mean reciprocal rank): 1 / the rank of the first relevant chunk, averaged; it shows whether the answer comes first. **Precision@1** is the share of questions where it does.
- Precision@K is capped below 1: most answers sit in one chunk, but K chunks are always returned, and small docs (rules.md has 2 chunks) can't fill K slots. Recall and MRR are the better signals for answer quality.
- Retrieval latency includes embedding the question through the OpenAI embeddings API,
  which dominates it; the vector search itself is a few milliseconds.
- End-to-end /chat latency is a real chat turn (agent, tools, model calls), spaced to stay
  under the per-session rate limit.

## Per-question detail

| Question | Precision | Recall | Latency (ms) | Retrieved sources |
| --- | --- | --- | --- | --- |
| What payment methods are accepted at VilniusPark car parks? | 0.67 | 1.00 | 1188.6 | general_info.md, general_info.md, parkings/station.md |
| Can I park a motorcycle at a VilniusPark car park? | 1.00 | 1.00 | 787.5 | general_info.md, general_info.md, general_info.md |
| Do disability parking card holders get free parking? | 0.33 | 1.00 | 393.7 | parkings/uzupis.md, parkings/old_town.md, general_info.md |
| What are the steps to reserve a parking space? | 1.00 | 1.00 | 603.7 | booking_process.md, booking_process.md, booking_process.md |
| What is the minimum and maximum length of a reservation? | 0.67 | 1.00 | 600.6 | booking_process.md, parkings/airport.md, booking_process.md |
| Can I cancel my parking reservation for free? | 0.67 | 1.00 | 393.7 | booking_process.md, booking_process.md, general_info.md |
| Why might a parking reservation be rejected? | 1.00 | 1.00 | 601.9 | booking_process.md, booking_process.md, booking_process.md |
| What is the maximum driving speed inside the car parks? | 0.33 | 1.00 | 603.2 | rules.md, general_info.md, booking_process.md |
| What happens if I lose my entry ticket? | 0.33 | 1.00 | 597.9 | rules.md, general_info.md, booking_process.md |
| Are LPG vehicles allowed in the Old Town car park? | 0.67 | 1.00 | 601.4 | rules.md, parkings/old_town.md, general_info.md |
| What is the height limit at the Old Town car park? | 0.67 | 1.00 | 403.5 | rules.md, parkings/old_town.md, parkings/old_town.md |
| How many EV charging bays does the Old Town car park have? | 0.33 | 1.00 | 597.1 | parkings/old_town.md, rules.md, parkings/business_center.md |
| What facilities does the Business Center car park have? | 0.67 | 1.00 | 600.2 | parkings/business_center.md, parkings/business_center.md, general_info.md |
| Where is the Užupis car park located and how big is it? | 0.67 | 1.00 | 599.8 | parkings/uzupis.md, general_info.md, parkings/uzupis.md |
| Is there a shuttle between the Airport car park and the terminal? | 0.67 | 1.00 | 595.4 | parkings/airport.md, parkings/airport.md, general_info.md |

## Evidently AI: LLM-as-judge answer quality

15 golden questions, real chatbot answers, judged by `gpt-4o-mini` with Evidently descriptors - full interactive report: [`evidently_report.html`](evidently_report.html).

| Check | Passing | Pass means |
| --- | --- | --- |
| Context quality | 100% | the top-3 retrieved chunks are enough to answer |
| Faithfulness | 93% | the answer is grounded in the retrieved context |
| Completeness | 93% | the answer gives every fact and condition the question asks for |
| Declined | 93% | the chatbot answered instead of refusing |
| PII in answer | 100% | no actual personal data (a name, plate, personal contact) in the answer |

How to read it - check the per-row reasoning in `evidently_rows.csv` before trusting a number:

- The judges compare each answer with the top-3 chunks retrieved for the *question as asked*. The chatbot searches with its own wording and also answers from the database tools (prices, hours, availability), so a correct answer can be marked unfaithful or incomplete.
- LLM judges are noisy: a plain "no" can be read as a refusal, and verdicts vary a little between runs.
- Completeness and PII use custom judges (`COMPLETENESS_CRITERIA`, `PII_CRITERIA`). Evidently's built-in Completeness wants the answer to repeat every retrieved chunk (13% here, failing "10 km/h" for not listing EV charger penalties); its PII judge flagged an answer for saying a booking needs a name and car number.
