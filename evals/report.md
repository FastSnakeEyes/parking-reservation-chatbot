# RAG evaluation report

Golden set: 30 questions; retrieval is scored on the 23 with a relevant doc, K=3

## Retrieval accuracy

- Mean Precision@3: 0.62
- Mean Recall@3: 0.96
- MRR@3: 0.91 (1.00 = the relevant doc is always ranked first)
- Precision@1: 0.87 (share of questions whose top chunk is relevant)

## Choosing K

| K | Precision@K | Recall@K | MRR@K | Precision@1 | Mean latency (ms) |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.74 | 0.89 | 0.89 | 0.87 | 599.9 |
| 3 (current) | 0.62 | 0.96 | 0.91 | 0.87 | 591.4 |
| 4 (recommended) | 0.57 | 1.00 | 0.92 | 0.87 | 599.9 |
| 5 | 0.56 | 1.00 | 0.92 | 0.87 | 582.6 |

Recommended K=4: the smallest K with the best recall. A larger K only adds chunks the model has to read past (Precision@K falls), and a smaller one misses answers. RETRIEVER_K is 3 - set it to 4.

## Retrieval latency

- Mean: 651.3 ms
- p95: 1758.6 ms

## End-to-end /chat latency

- Samples: 5
- Mean: 2547.2 ms
- p95: 4693.8 ms

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

| Question | Category | Precision | Recall | Latency (ms) | Retrieved sources |
| --- | --- | --- | --- | --- | --- |
| What payment methods are accepted at VilniusPark car parks? | single_fact | 0.67 | 1.00 | 1997.7 | general_info.md, general_info.md, parkings/station.md |
| Can I park a motorcycle at a VilniusPark car park? | single_fact | 1.00 | 1.00 | 582.1 | general_info.md, general_info.md, general_info.md |
| Do disability parking card holders get free parking? | single_fact | 0.33 | 1.00 | 802.1 | parkings/uzupis.md, parkings/old_town.md, general_info.md |
| What are the steps to reserve a parking space? | single_fact | 1.00 | 1.00 | 596.3 | booking_process.md, booking_process.md, booking_process.md |
| What is the minimum and maximum length of a reservation? | single_fact | 0.67 | 1.00 | 605.1 | booking_process.md, parkings/airport.md, booking_process.md |
| Can I cancel my parking reservation for free? | single_fact | 0.67 | 1.00 | 602.1 | booking_process.md, booking_process.md, general_info.md |
| Why might a parking reservation be rejected? | single_fact | 1.00 | 1.00 | 592.2 | booking_process.md, booking_process.md, booking_process.md |
| What is the maximum driving speed inside the car parks? | single_fact | 0.33 | 1.00 | 604.1 | rules.md, general_info.md, booking_process.md |
| What happens if I lose my entry ticket? | single_fact | 0.33 | 1.00 | 608.1 | rules.md, general_info.md, booking_process.md |
| Are LPG vehicles allowed in the Old Town car park? | single_fact | 0.67 | 1.00 | 592.6 | rules.md, parkings/old_town.md, general_info.md |
| What is the height limit at the Old Town car park? | single_fact | 0.67 | 1.00 | 595.7 | rules.md, parkings/old_town.md, parkings/old_town.md |
| How many EV charging bays does the Old Town car park have? | single_fact | 0.33 | 1.00 | 599.7 | parkings/old_town.md, rules.md, parkings/business_center.md |
| What facilities does the Business Center car park have? | single_fact | 0.67 | 1.00 | 606.7 | parkings/business_center.md, parkings/business_center.md, general_info.md |
| Where is the Užupis car park located and how big is it? | single_fact | 0.67 | 1.00 | 600.3 | parkings/uzupis.md, general_info.md, parkings/uzupis.md |
| Is there a shuttle between the Airport car park and the terminal? | single_fact | 0.67 | 1.00 | 601.7 | parkings/airport.md, parkings/airport.md, general_info.md |
| my car is 2.15 m tall, will it fit in the old town garage? | paraphrase | 0.67 | 1.00 | 595.3 | parkings/old_town.md, parkings/old_town.md, parkings/business_center.md |
| I'm flying out for 3 weeks - where can I leave the car, and can I book it for that long? | paraphrase | 1.00 | 1.00 | 600.9 | booking_process.md, booking_process.md, parkings/airport.md |
| can I pay in cash at the Užupis lot? | paraphrase | 0.00 | 0.00 | 597.7 | general_info.md, general_info.md, general_info.md |
| which of your parkings let me charge an electric car? | paraphrase | 0.33 | 1.00 | 399.3 | general_info.md, booking_process.md, rules.md |
| I need to catch a 5am train, is the station parking open that early? | paraphrase | 0.67 | 1.00 | 603.6 | parkings/station.md, parkings/station.md, general_info.md |
| What is the penalty for blocking an EV charger, and how many charging bays does the Airport car park have? | multi_fact | 0.67 | 1.00 | 593.0 | rules.md, parkings/airport.md, parkings/business_center.md |
| I have a disability parking card - can I park for free at Užupis, and how many accessible spaces does it have? | multi_fact | 0.33 | 1.00 | 407.9 | parkings/uzupis.md, general_info.md, parkings/old_town.md |
| If I cancel one hour before my reservation starts, what do I pay, and how long is my space held if I arrive late? | multi_fact | 1.00 | 1.00 | 595.2 | booking_process.md, booking_process.md, booking_process.md |

## Evidently AI: LLM-as-judge answer quality

30 golden questions, real chatbot answers, judged by `gpt-4o-mini` with Evidently descriptors - full interactive report: [`evidently_report.html`](evidently_report.html).

**Overall: 77%** of questions pass every check that applies to them.

| Check | Passing | Pass means |
| --- | --- | --- |
| Context quality | 87% | the top-3 retrieved chunks are enough to answer |
| Faithfulness | 83% | the answer is grounded in the retrieved context |
| Completeness | 87% | the answer gives every fact and condition the question asks for |
| PII in answer | 100% | no actual personal data (a name, plate, personal contact) in the answer |
| Answered or declined correctly | 97% | answers what it can; declines out-of-scope, private-data, injection and not-in-docs questions |

Context quality, faithfulness and completeness are scored on the questions to answer only.

| Question category | Questions | Passing every check |
| --- | --- | --- |
| single_fact | 15 | 100% |
| paraphrase | 5 | 20% |
| multi_fact | 3 | 33% |
| not_in_docs | 2 | 50% |
| out_of_scope | 2 | 100% |
| private_data | 2 | 100% |
| injection | 1 | 100% |

How to read it - check the per-row reasoning in `evidently_rows.csv` before trusting a number:

- The judges compare each answer with the top-3 chunks retrieved for the *question as asked*. The chatbot searches with its own wording and also answers from the database tools (prices, hours, availability), so a correct answer can be marked unfaithful or incomplete.
- LLM judges are noisy: a plain "no" can be read as a refusal, and verdicts vary a little between runs.
- Completeness and PII use custom judges (`COMPLETENESS_CRITERIA`, `PII_CRITERIA`). Evidently's built-in Completeness wants the answer to repeat every retrieved chunk (13% here, failing "10 km/h" for not listing EV charger penalties); its PII judge flagged an answer for saying a booking needs a name and car number.
