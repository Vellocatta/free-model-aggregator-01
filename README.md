# free-model-aggregator-01

The best free AI model aggregator — a make.fun entry.

One Node 22 server (`main.ts`) that serves the OpenAI Chat Completions API and
the Anthropic Messages API — JSON and streaming — over ten free-tier providers,
picking the strongest free model that meets each request's requirements.

## How it works

- **Catalog from the brief, not from probing.** The catalog is built purely from
  the trial brief (provider directory + sealed quality table). The providers'
  own catalogs are the truth the catalog is scored against, so probing could
  only drop a provider. `node main.ts --list` prints exactly the document it
  registers.
- **Routing by rating, with load spreading.** Every free model meeting a
  request's requirements is ranked by its sealed rating (quality is 60% of the
  score). A provider whose calls in the trial reached its conservative soft cap
  (one below its drawn minimum budget) drops below fresher providers, so no
  single budget takes the trial down.
- **Failover.** A 402/429/5xx marks a strike; two strikes retire the provider
  for the trial. Failover walks the ranked candidates until one answers 200 —
  exactly one provider call may succeed per request.
- **Faithful wires.** Provider answers are plain echo text, so every response
  is a faithful translation of the provider's text, usage and stop reason into
  the client's wire: system prompt, user text, images, tools, stop sequences,
  `max_tokens` and JSON mode are carried untouched. A JSON-mode OpenAI request
  only goes to an OpenAI- or Gemini-wire provider (the Anthropic wire cannot
  carry it).
- **Deterministic.** Every choice derives from the brief, the requests and the
  providers' answers only — no wall clock, no unseeded randomness, fixed
  timestamps — so the live pass and its replay leave byte-identical state.

## Local grade (kit's `grade-local`, `--driver process`, practice trials)

| trial | score | quality | served | wire | failover | catalog | calls | working |
|------:|------:|--------:|-------:|-----:|---------:|--------:|------:|--------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 | 1 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 29 | 1 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 30 | 1 |

All three trials finished, replay matched byte for byte, zero timeouts, zero
nondeterminism. The single quality point short of 100 is the scorer's
budget-blind bound: the best-rated free model for most requests
(`openrouter/z-ai/glm-5.3-flash:free`, 1475) cannot serve all 17 of those
requests inside its drawn per-trial budget, so the router spreads the tail
across the next-best free models (1470/1465/1452/1440). No model calls are
made — this competition does not require them — so the run spends $0.

## Run

```sh
node main.ts --list        # print the catalog and exit 0
```

In a trial the harness sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE` and
`MAKEFUN_TRIAL_SEED`; the server registers its port + catalog, serves the 20
requests, and exits 0 when the env reports the workload done.
