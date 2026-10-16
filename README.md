# Equi — Agentic GTM: Demand Engine + Lead Pipeline

Two deliverables, one thesis: **the same signal warehouse should produce both the content and the outreach.** Treating them as separate programs is why most alts distribution teams run a content calendar that nobody reads and a sequencer that nobody answers.

- **Part 1** — demand plan: three experiments, instrumented, with explicit kill points.
- **Part 2** — a working lead pipeline: clean → resolve → score → route → personalize, with a ranked shortlist you can defend line by line.

```bash
pip install -r requirements.txt
python -m src.pipeline --offline                    # no API key needed, fully reproducible
GEMINI_API_KEY=... python -m src.pipeline           # live LLM openers (gemini-3.6-flash)
pytest -q                                           # 39 assertions
```

Outputs land in `out/`: `shortlist.csv`, `scored_leads.csv`, `merge_audit.csv`, `enrichment_queue.csv`, `run_report.json`.

---

# Part 1 — The Demand Problem

## The diagnosis

Equi is not short of content. It is short of **surface area that compounds**. A founder webinar is a one-shot asset with a decaying audience; a personalized read on a prospect's filings is a high-quality asset with no distribution leverage — it costs a human an hour and reaches exactly one firm.

The current outbound motion (a bespoke read of a prospect's alts positioning from public filings) is the right instinct built on the wrong economics. It's a **manual artifact with a marginal cost that doesn't fall**. The fix is not to send more of them. It is to turn that artifact into a **computed output of a data spine**, then expose the same spine through three surfaces with different costs and different latencies.

```
                 ┌─────────────────────────────────────────┐
                 │  SIGNAL SPINE                           │
                 │  Form ADV Part 1/2 deltas · brand &      │
                 │  domain changes · site copy diffs ·      │
                 │  job posts · custodian · personnel moves │
                 └────────────────┬────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
   EXP 1 Rebrand Radar      EXP 2 Readiness Index    EXP 3 IC-Memo Generator
   1:1, event-triggered     public, always-on        self-serve, intent-captured
   highest intensity        highest compounding      highest qualification
```

One warehouse. Three motions. Every artifact one motion produces is reusable by the other two — the per-firm scorecard that powers the public index *is* the payload of the cold email, and the memo an advisor generates themselves *is* the proof the champion needs in their IC.

---

## Experiment 1 (flagship) — "Rebrand Radar": signal-based audio outreach

**Hypothesis.** An independent RIA mid-transition to a multi-family office has a live, time-boxed infrastructure problem: it is promising family-office-grade service to $10M+ clients before it has family-office-grade alternatives capability. A 90-second audio brief that names that gap, in the voice of a peer allocator and addressed to the one person who can act on it, will out-convert any written cold email — because it arrives at the moment the gap is on the principal's whiteboard.

**Why audio specifically.** Three reasons, in order of weight:

1. **Channel arbitrage.** The RIA principal inbox is saturated with text from wholesalers. A 90-second listen is a different cognitive slot — consumed in the car, between client meetings, where it competes with podcasts rather than with 40 other emails.
2. **It is non-forwardable-looking but highly forwardable.** A principal who finds it useful sends it to their ops lead. Text gets skimmed and deleted; a brief about *their own firm* gets played.
3. **Listen-through is a high-frequency metric.** This matters more than it sounds — see the power analysis below. Reply rate on a 1,500-firm universe is statistically unreadable within a few weeks. Listen-through percentage gives a measurable signal in days.

**The signal.** RIA→MFO transition is not a field in any database, so it is scored as a conjunction of weak signals:

| Source | Signal | Weight |
|---|---|---|
| Form ADV Part 1 | HNW client count rising while total client count falls | strong |
| Form ADV Part 1 | Item 5 discretionary AUM per client crossing ~$10M | strong |
| Brand / domain | new domain registration, `*familyoffice*` / `*privatewealth*` TLD acquisition | strong |
| Site copy diff | "wealth management" → "family office" / "multi-family office" in H1 and nav | strong |
| Job posts | first hire titled *Director of Research*, *Head of Alternatives*, *Family Office Services* | medium |
| ADV Part 2 brochure | new language on alternative investments, QP suitability, evergreen structures | medium |
| Personnel | CIO hire from an MFO or an allocator seat | medium |

`src/score.py::detect_mfo_transition()` implements the list-data-only proxy of this (name semantics × client size × scale). The production version reads the ADV quarterly dump and diffs it against the prior snapshot — that is the only part of this motion that requires real engineering, and it is a single scheduled job over a public dataset.

### Architecture

```
┌──────────────┐   weekly    ┌──────────────┐          ┌─────────────────┐
│ SEC IAPD /   │────────────▶│ ADV snapshot │─────────▶│ delta detector  │
│ Form ADV ZIP │   cron      │ (BigQuery)   │  diff    │ (SQL + rules)   │
└──────────────┘             └──────────────┘          └────────┬────────┘
┌──────────────┐   daily     ┌──────────────┐                   │
│ site copy /  │────────────▶│ page hashes  │───────────────────┤
│ domain / ATS │  scraper    │ (GCS)        │                   │
└──────────────┘             └──────────────┘                   ▼
                                                       ┌─────────────────┐
                                                       │ ICP scorer      │
                                                       │ (Part 2 engine) │
                                                       └────────┬────────┘
                                                      score ≥ Tier A + signal
                                                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│ BRIEF COMPOSER (LLM)                                                   │
│  in:  ADV deltas, sleeve mix, client band, custodian, decision topology│
│  out: 90-sec script (JSON) + written one-pager, both firm-specific     │
│  guardrails: no invented AUM/performance/personnel; claim-check pass   │
│              against source fields; fail closed → human queue          │
└───────────────────────────┬────────────────────────────────────────────┘
                            ▼
              ┌──────────────────────────┐
              │ ElevenLabs TTS           │   founder-cloned voice, consented
              │ → mp3 → GCS → signed URL │   per-firm landing page w/ player
              └────────────┬─────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │ Sequencer (Smartlead/Instantly) + CRM    │
        │ email body = 2-sentence opener + link    │
        │ events: open, click, play, 25/50/75/100% │
        │         listen, reply, meeting booked    │
        └──────────────────┬───────────────────────┘
                           ▼
                 ┌────────────────────┐
                 │ BigQuery event     │  → experiment readout
                 │ warehouse          │     (variant, firm, stage)
                 └────────────────────┘
```

**Build order, and what is already built.** The scorer, the signal proxy, the LLM personalization layer with claim guardrails, the deterministic A/B assignment and the event schema are in this repo (Part 2). What remains for the flagship is the ADV delta job, the TTS call, and the per-firm player page — roughly a day of work, because the hard part was never the audio.

**Hosted player, not an attachment.** The mp3 lives on a per-firm page, not in the email. Attachments wreck deliverability, and a hosted page gives granular listen telemetry, which is the metric the whole experiment is powered on.

### A/B design

The honest constraint first. The US universe of independent RIAs and MFOs in the $1–30B band is on the order of **1,000–2,000 firms** (verify against ADV before committing budget). Detecting a lift from a 4% to an 8% reply rate at 80% power, α=0.05, requires **~550 firms per arm**. That means a classic two-arm reply-rate test would consume *the entire addressable market* for a single readout — and you'd only get one.

So the test design inverts: **A/B on the high-frequency upstream metric, gate on the low-frequency downstream one.**

| Layer | What varies | Metric tested | N needed | Method |
|---|---|---|---|---|
| Primary A/B | audio brief vs. written brief, same content | listen/read-through ≥ 50% | ~120/arm | fixed-horizon, 2 weeks |
| Secondary A/B | opener frame: filings-anchored vs. cohort-benchmark | click-to-asset rate | ~200/arm | implemented in `personalize.py` |
| Gate (not a test) | whole motion | reply rate, meetings booked | all | sequential monitoring, pre-registered thresholds |

Assignment is a deterministic hash of the firm key (`personalize.py::assign_variant`), so a re-run never reshuffles a firm between arms — the single most common way these experiments get quietly corrupted.

**Guardrails that stop the test regardless of performance:** spam complaint rate > 0.1%, bounce rate > 3%, or any domain reputation drop. Deliverability is a shared, non-recoverable asset; no experiment outranks it.

### Instrumentation and unit economics

Modeled costs — these are planning assumptions to be replaced by measured values in week one, not claims:

| Line | Assumption | Per firm |
|---|---|---|
| Signal detection (amortized infra) | ~$400/mo over ~800 monitored firms | ~$0.50 |
| Script generation (LLM) | ~2.5k in / 400 out tokens, mid-tier model | ~$0.01 |
| TTS (ElevenLabs) | ~220 words ≈ 1.4k characters | ~$0.15–0.30 |
| Human QC (Tier A only) | 4 min @ fully-loaded AE cost | ~$4.00 |
| **Total, Tier A** | | **~$5 / firm** |

At 300 Tier-A firms per quarter that is ~$1,500 of variable cost. Against a single white-label mandate, the breakeven is measured in basis points of one firm's allocation. **The binding constraint on this motion is not cost — it is signal quality and sender reputation.** Scale the list and you destroy both. This is deliberately a low-volume, high-intensity motion; anyone proposing to run it at 5,000 firms/month has misunderstood it.

### Decision gates

| When | Check | Action |
|---|---|---|
| Day 3 | pipeline integrity: briefs generated, claim-check pass rate ≥ 95%, zero hallucinated facts in a 20-brief manual audit | fix before any send; a wrong AUM in a peer-to-peer brief is unrecoverable with that firm |
| Week 2 | listen-through ≥ 50% on ≥ 120 sends | below 35% → the *content* is wrong, not the channel: rewrite the script, re-run once |
| Week 3 | positive reply rate ≥ 2× the written control | below parity → audio adds cost without lift; fold the signal detection into Exp 2 and kill the audio layer |
| Week 6 | ≥ 3 qualified meetings from ≤ 150 Tier-A sends, ≥ 1 advancing to materials | met → productionize and expand the signal set |

**Kill point.** Terminate at week 6 if either: (a) positive reply rate is below 2% on ≥ 150 sends with listen-through above 50% — meaning people listen and still don't care, which falsifies the offer, not the channel; or (b) fewer than 2 qualified meetings. Terminate immediately at any point on a deliverability guardrail breach.

**What survives a kill.** The signal spine and the scorer. That is the deliberate design choice: the expensive, reusable asset is the data layer, and the audio is a cheap, disposable surface on top of it. Killing Experiment 1 costs the TTS integration and nothing else.

---

## Experiment 2 — "Evergreen Alts Readiness Index": a public data product

**Hypothesis.** The fastest route to becoming the recognizable voice in a channel is not publishing more opinions — it is publishing the **benchmark the channel measures itself against**. A firm that is scored by your index has to engage with your framing to dispute it.

**What it is.** A public, quarterly index built from Form ADV and public filings: for each independent RIA and MFO in the band, a readiness score on evergreen alternatives (client QP density, current sleeve mix, liquidity profile of existing allocations, infrastructure signals). Firm-level pages, cohort pages by AUM band and region, and a downloadable methodology.

**Why it sits where content and outreach overlap.** Every firm page is simultaneously (a) an SEO/AEO asset indexed against the exact queries advisors and their clients search, (b) the cohort comparison the Part 2 openers already reference, and (c) a legitimate, non-pitch reason to contact a firm: *here is your score, here is the methodology, tell us what we got wrong.* That last one is the highest-reply-rate cold email in existence, because it offers correction rights over a public claim about them.

**AEO is the real prize.** Advisors researching evergreen structures increasingly ask an LLM before they ask a wholesaler. Answer engines cite structured, cited, machine-parseable sources. A quarterly index with a published methodology, stable URLs, schema markup and a citable dataset is exactly the shape of thing they cite. Opinion blog posts are not.

**Instrumentation.** Indexed pages, organic impressions on commercial-intent queries, **citation share** (tracked by running a fixed panel of ~50 advisor-intent prompts weekly across ChatGPT/Perplexity/Google AI Overviews and logging whether Equi is cited), inbound "fix my score" requests, assisted pipeline.

**Gates.** Week 4: ≥ 300 firm pages live and indexed, ≥ 5 inbound correction requests. Week 8: citation share > 10% on the prompt panel, or ≥ 10 inbound conversations. **Kill point:** at week 12, if organic + inbound has produced fewer than 3 qualified conversations *and* citation share is under 5%, stop publishing new cohorts and retain only the data layer, which Experiment 1 needs anyway. Note the asymmetry: this is the slowest experiment to read and the only one that compounds, so it gets the longest leash and the strictest leading indicators.

---

## Experiment 3 — "IC-Memo Generator": product-led demand for committee-led firms

**Hypothesis.** Equi's own ICP note says committee-led firms are the slow half and that the champion must be *equipped*, not closed. So build the equipment and give it away: a self-serve tool that generates an investment-committee-ready memo on an evergreen alternatives allocation — structure, liquidity terms, risk framing, suitability, the objections an IC will raise and how to answer them — exportable under the advisor's own brand.

**Why this and not more content.** It inverts the sales motion. Instead of Equi arguing its framing into a committee it will never attend, the champion walks in carrying a document built on Equi's framing, with Equi's analytical structure, which they will defend as their own work. It is also the most honest qualification mechanism available: a director of research who spends 15 minutes generating an IC memo has disclosed more intent than any form fill or webinar attendance.

**Mechanics.** Gated by work, not by email — the user configures a real memo before being asked who they are. Every generation is a scored intent event routed into the same pipeline in Part 2, arriving pre-qualified with the firm's own stated parameters attached. It also doubles as the Tier-A follow-up asset in the `A_COMMITTEE` sequence.

**Gates.** Week 3: ≥ 40 memo generations. Week 6: ≥ 25% of generations from firms inside the ICP band, ≥ 5 sales conversations originated. **Kill point:** week 8, if ICP-band generations are under 15% of volume (it's attracting the wrong audience) or if generations haven't produced 3 conversations (it's a nice tool, not a demand channel). Strip it to a static template and retain it as sales collateral.

## Sequencing and why

Run **1 and 2 together from day one**, 3 from week three. They share the spine, so the marginal cost of the second is small, and they fail for independent reasons — if the audio motion dies, the index still runs. Experiment 1 produces a readable signal fastest and funds attention for the other two; Experiment 2 is the only one that still pays out in year two. Experiment 3 waits because it needs the index's firm pages to have somewhere to send people.

---

# Part 2 — The Lead Pipeline

## Run it

```bash
python -m src.pipeline --offline                       # deterministic, no credentials
GEMINI_API_KEY=... python -m src.pipeline              # defaults to gemini-3.6-flash
GEMINI_API_KEY=... python -m src.pipeline --threshold 70 --max-personalize 10
```

## Results on the supplied 50-row list

| | |
|---|---|
| Input rows | 50 |
| After entity resolution | 45 firms (5 duplicate clusters collapsed) |
| Structurally disqualified | 10 |
| **Tier A** (founder-direct + audio) | **7** |
| **Tier B** (automated 4-touch) | **12** |
| Tier C (nurture) | 7 |
| Blocked on enrichment | 3 |
| Openers generated | 19 |

Top of the shortlist, with the reason each firm is there:

| # | Firm | AUM | Score | Why |
|---|---|---|---|---|
| 1 | Bramblewick Capital Advisers | $9.6B | 111 | RIA at $9.6B serving $11M clients — MFO economics, RIA branding |
| 2 | Quillmoor Advisors | $7.7B | 111 | same transition signature, two sleeves, CIO-led |
| 3 | Solstice Grove Advisors | $8.9B | 109 | transition signal, but committee-led → champion-enablement sequence |
| 4 | Aldermoor Capital Group | $6.1B | 102 | MFO, bullseye band, $19M clients, moderate alts |
| 5 | Sunridge Collective | $3.05B | 102 | transition signal, never contacted — clean first touch |
| 6 | Marisol & Hart Private Wealth | $3.1B | 101 | warm intro on record; open with the relationship, not a cold line |
| 7 | Fenwick Larkspur Capital | $12.3B | 101 | large, alts-native — strong fit, longer displacement cycle |

## Architecture

```
load ──▶ normalize ──▶ resolve entities ──▶ score ──▶ route ──▶ personalize (gated) ──▶ emit
         per-field      union-find over      2-stage   tier ×     LLM, cached,
         + parse flags  name & domain        model     topology   validated
```

Five modules, each independently testable:

| Module | Responsibility |
|---|---|
| `normalize.py` | total, pure field parsers; never raises, records `_parse_flags` instead |
| `dedupe.py` | entity resolution + contact election; emits a merge audit |
| `score.py` | eligibility (boolean) then fit (continuous), with per-component rationale |
| `route.py` | tier × decision topology → sequence, owner, QC policy |
| `personalize.py` | gated LLM openers, deterministic A/B arms, validation, cost accounting |

## The decisions worth defending

**1. Eligibility is a gate, not a penalty.** A $340B wirehouse should not appear on a ranked list at position 30 with a respectable score. Firm type, the $30B ceiling, the sub-scale floor and US domicile are boolean. Ten firms are removed outright, each with a stated reason in `disqualify_reason`. This is the single highest-leverage decision in the model: weighted scoring alone would have let large, well-staffed, structurally unsellable firms crowd the top of the list.

**2. Two-stage over a classifier.** There is no outcome data yet. An additive model with explicit weights is auditable, arguable in a GTM meeting, and re-weightable the moment reply data exists. A model trained on 50 rows would be fitting noise and would be unable to answer "why is this firm third". The first ~200 replies are the training set; `SCORE_MODEL_VERSION` is stamped on every row so that re-weighting can be evaluated against cohorts scored under the old weights.

**3. Weights, and the reasoning behind the ranking.** Equi deliberately withheld these. Mine:

| Dimension | Max | Why that rank |
|---|---|---|
| AUM fit | 30 | the band is a hard economic constraint at both ends; the brief states the ceiling exists because the model breaks above it |
| Firm type fit | 25 | stated explicitly to matter more than size; MFO > RIA > trust > SFO |
| Client qualification | 20 | QP density *gates eligibility for the fund structures* — a legal constraint wearing a marketing costume |
| Alts posture | 15 | non-monotonic: the middle scores highest, "none" and "everything" both score low for opposite reasons |
| Decision velocity | 10 | the stated best predictor of speed, but it affects *how fast* a deal moves, not *whether* the firm can buy — so it is a tiebreaker, not a driver |
| MFO transition | +12 | the strongest stated signal, and the only one that is time-sensitive; it must be able to move a firm a full tier |

Penalties: captive custodian (-8, LPL/in-house gate third-party alts), competitor recently installed (-10, displacement cycle), stale relationship (-3).

**4. Alts posture is non-monotonic, and that is the point.** Most scoring models treat "more alternatives experience" as monotonically better. The brief says the opposite: a firm doing nothing needs Equi most but carries a long education curve, and an alts-native firm already has its own infrastructure. Two sleeves scores 15; "everything" scores 5.

**5. Two policy caps, applied to tier not to score.** A firm with undisclosed AUM cannot reach Tier A, and a sub-$1B firm cannot reach a 1:1 send tier, regardless of how well it scores elsewhere. Cairnwell Private Capital ($850M MFO, $9M clients) scores well and is still held at Tier C — band discipline is policy, not a weight to be out-argued by a good contact. The raw score stays visible so a rep can see the firm is otherwise excellent.

**6. Entity resolution elects a contact, it doesn't just merge rows.** Thornbury Wealth Advisors appears twice with two different humans: a Director of Manager Research and an EA to the CIO. That is not a duplicate record, it is a contact decision. Union-find over name slug and email domain collapses the firm; the surviving contact is chosen by decision velocity, and the EA is retained in `alternate_contacts` rather than discarded. Five clusters collapse on this list, every one logged in `merge_audit.csv` — a GTM team should not have to trust a dedupe it can't inspect.

**7. AUM scale inference is explicit and flagged.** `11500` means $11.5B, `6800000000` means $6.8B, `4.2` would mean $4.2B. Resolution is by order of magnitude, and every inferred row carries `aum_scale_inferred:millions` in `_parse_flags`. A silent coercion here is a mis-scored lead nobody audits.

**8. Fail closed on LLM output.** Openers are requested as JSON, validated against a banned-phrase list, sentence count, length and unfilled placeholders, then routed to `needs_review` rather than to a sender if anything trips. The failure mode this prevents — "Dear [FIRM]" landing in a $9B CIO's inbox — costs more than the lead is worth.

**9. LLM spend is gated by score and cached by content hash.** Only leads above the threshold are personalized (19 of 45 here). Re-running an unchanged row costs $0. Modeled cost on the live path with `gemini-3.6-flash` is ~$0.0010/lead at ~700 in / ~120 out tokens — about two cents for the full 50-row run — and the measured figure is written to `run_report.json` per run (`llm_cost_usd`, `cost_per_shortlisted_lead_usd`). Trivial at this volume, which is exactly why it is worth instrumenting now, before the list is 50,000 rows and nobody can explain the bill.

**10. Routing is tier × topology, not score alone.** Principal-led Tier A firms get a founder-direct sequence; committee-led Tier A firms get champion enablement, because the brief is explicit that these people must defend the pick internally. Any firm missing a routable contact or a verified AUM goes to the enrichment queue regardless of score — sending into a gatekeeper inbox burns both the domain and the firm.

## A/B testing and measurement

**Implemented:** deterministic variant assignment (`assign_variant`) splitting openers between a filings-anchored frame and a cohort-benchmark frame. Same firm, same arm, every run. `ab_variant` and `score_model_version` are stamped on every row so any readout can be sliced by both.

**What I would measure, in order of how early it reads:**

| Metric | Reads by | What it tells you |
|---|---|---|
| Deliverability (bounce, complaint, inbox placement) | day 1 | whether anything else is measurable at all |
| Open / listen-through | week 1 | subject line and channel fit |
| Positive reply rate by variant | week 2–3 | opener frame quality |
| Positive reply rate by **tier** | week 3–4 | **whether the scoring model is real** |
| Meetings booked per 100 sends by tier | week 6 | the only number that matters |
| Score decile vs. reply rate correlation | week 8 | the input to re-weighting |

The fourth row is the one most teams skip and the one that validates this entire build. If Tier A does not out-reply Tier B by a clear margin, the weights are wrong and should be refit on observed outcomes — that is precisely why every component score is persisted per row rather than just the total.

**Sequential, not fixed-horizon, for the gate metrics.** With a ~1,500-firm universe you cannot afford a properly powered fixed-horizon reply-rate test (see Part 1). Monitor cumulatively against pre-registered thresholds and accept the inflated false-positive rate as the price of not burning the market on a single readout.

## Things I would fold in to actually land the meeting

- **Deliverability as a first-class constraint**: separate sending domains per motion, warmed, with per-domain volume caps enforced in the router. The best-scored list in the world is worth nothing from a burned domain.
- **Enrichment before send, not after**: the 3 firms in `enrichment_queue.csv` are blocked deliberately. ADV Part 1 resolves AUM and client mix for all of them in one API call.
- **Trigger on the champion, not just the firm**: a Director of Research changing firms takes the relationship with them. Personnel moves inside scored firms should re-open a closed-lost record.
- **Reply classification back into the score**: route replies through a classifier (interested / not now / wrong person / never) and feed "wrong person" back into the persona model. That is a scoring bug reported for free.
- **Referral path over cold path for warm records**: Marisol & Hart has a warm intro on file and still scored into the cold sequence by default. Any record with warm context should be routed to a human for a referral ask, not to a sequencer. Implemented as a signal; the routing override is a one-line change I'd want to make with the AE in the room.

## Tradeoffs and known limits

- **The MFO transition detector is a proxy.** On list data alone it can only infer the signal from client size × scale × name semantics. It fires on 5 firms here, and I would expect a meaningful false-positive rate against real ADV deltas. The production detector is described in Part 1; this is honest scaffolding, not the real thing.
- **Tier thresholds are calibrated to this sample.** A tier boundary is a capacity decision — how many firms one AE can work in a fortnight — not a statistical one. They will need re-cutting against the real universe.
- **No persistence layer.** CSV in, CSV out. With volume this becomes BigQuery with the scored table partitioned by run date, which is both where Equi already works and what the time-series analysis in the measurement section requires.
- **The offline opener is a template and reads like one.** It exists so a reviewer can run the whole pipeline with no credentials. The live path with the real prompt is where the copy quality lives.
- **Single-threaded LLM calls.** Fine at 19 leads, wrong at 19,000; that is an `asyncio.gather` with a semaphore, deliberately not built here.

## Repo layout

```
src/config.py        ICP policy: weights, bands, thresholds, taxonomies — all tunable in one file
src/normalize.py     total field parsers + parse-flag recording
src/dedupe.py        union-find entity resolution + contact election
src/score.py         eligibility gate + weighted fit model + rationale
src/route.py         sequences, owners, QC policy
src/personalize.py   LLM openers, A/B arms, validation, caching, cost accounting
src/pipeline.py      orchestration + CLI
tests/               39 assertions over parsing, resolution, scoring policy, validation
out/                 generated: shortlist, scored leads, merge audit, enrichment queue, run report
```
