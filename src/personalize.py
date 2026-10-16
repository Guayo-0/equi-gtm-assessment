"""
LLM personalization layer.

Design constraints that drove this file:

  1. Spend is gated by score. We only pay for tokens on leads the model already
     believes in (>= PERSONALIZE_MIN). Personalizing a disqualified wirehouse is
     a rounding error per row and a real number at 50k rows.
  2. Output is structured and validated. A free-text completion dropped into a
     sequencer is how you ship "Dear [FIRM]" to a $9B CIO. We request JSON,
     validate it, and fail the row closed (status=needs_review) rather than
     emitting something unsendable.
  3. Deterministic A/B arms. Variant assignment is a hash of the firm key, not
     a coin flip, so a re-run never reassigns a firm mid-experiment.
  4. Idempotent + cached. Re-running the pipeline on an unchanged row costs $0.
  5. Offline mode. The whole pipeline runs with no API key via a deterministic
     template renderer, so the repo is reproducible by a reviewer who doesn't
     want to burn credits.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass

import pandas as pd

from . import config as C

# --- Prompting ---------------------------------------------------------------

# Bumped whenever the prompt changes: it is part of the cache key, so a new
# version invalidates cached openers instead of serving copy written under the
# old rules.
PROMPT_VERSION = "opener-v3-particular"

SYSTEM_PROMPT = """You write first-touch outreach for Equi, which builds \
institutional-grade liquid alternative strategies with managed risk and \
downside protection, structured as evergreen vehicles rather than lock-ups. \
Equi sells to independent RIAs and multi-family offices as a partner, not a \
vendor: it builds the advisor's client-facing materials so the advisor can \
sell to their own clients.

You are one allocator writing to one other allocator. The single test your \
message must pass: IT MUST BE IMPOSSIBLE TO SEND TO ANY OTHER FIRM. If the \
sentence would still be true and sensible with a different firm's name on it, \
you have failed, no matter how sharp it sounds.

THE FIRST SENTENCE — what it must NOT be:
- Not a generalisation about a category. Banned openings: "Most", "Many", \
"Firms that", "Advisors who", "RIAs", "Multi-family offices", "In our \
experience", "Based on my experience". A claim true of a hundred firms is \
not an opener, it is a billboard.
- Not a recital of their own figures. Do not begin with a gerund or \
participle ("Managing", "Running", "Given", "With", "As a", "Having"), and \
do not make their AUM or client size the subject.
- It must NOT mention Equi, "we", or any product. The first sentence belongs \
entirely to the reader's world.

THE FIRST SENTENCE — what it must be:
- Built on the most PARTICULAR fact available about this firm. Rank them: \
the internal CRM note first (a meeting, an introduction, a competitor they \
just installed, a committee that moves slowly), then their specific sleeve \
mix or custodian, then their position. Use the CRM note whenever one exists, \
without quoting it word for word and without sounding like you read a file.
- It should carry an observation or a consequence they are probably already \
living with, stated flatly enough that they could disagree with it.

THE SECOND SENTENCE:
- Do not begin with "I can send", "I can share", "I have", "We built", \
"We recently", "I put together", "I mapped", "Equi".
- Mention what Equi does ONLY as a half-clause, in passing, in the reader's \
terms. Never as a product description, never as a list of features, never \
with the company name leading the sentence.
- Do not ask for time, a call, or fifteen minutes.

GENERAL:
- Exactly two sentences. Under 55 words.
- Never invent AUM, performance, personnel, or events. Only the context given.
- Banned anywhere: "I hope this finds you well", "reaching out", "quick \
question", "circle back", "touch base", "solutions", "synergies", \
"game-changing", "I'd love to", "liquidity trap", any superlative about Equi.
- The reader is sophisticated. Do not explain what alternatives are. Do not \
pitch returns. Use "evergreen" at most once, and only if it earns its place.
- Contractions are good. One rhetorical question is allowed if it has teeth.

Return strict JSON only:
{"opener": str, "hook_used": str, "confidence": float 0-1, "risk_flags": [str]}

hook_used: name the particular fact you built the first sentence on.
risk_flags: list any claim you were not fully certain of, or [] if none."""

# The Tier A audio motion is read aloud by a human-sounding voice. A 90-second
# brief that starts with no one addressed sounds like a radio ad; an email that
# opens with a greeting collides with the one the sequencer adds.
REGISTER_SPOKEN = """REGISTER — THIS WILL BE SPOKEN ALOUD, not read.
Open by addressing them by first name, informally, as you would starting a \
voice note to a peer: their first name, then straight into the thought. \
Short clauses. Contractions throughout. It should sound like a person \
thinking out loud, not a paragraph being performed. One aside is welcome."""

REGISTER_WRITTEN = """REGISTER — THIS IS EMAIL BODY TEXT.
No greeting and no name: the sequencer inserts those. Start directly with \
the first sentence."""

VARIANT_INSTRUCTIONS = {
    "filings_anchored": (
        "FRAME A — the edge comes from THEIR OWN SITUATION. Name the tension "
        "between what this specific firm has set up and what the clients it "
        "serves will eventually ask of it. Their particulars are the subject "
        "matter, not decoration."
    ),
    "peer_benchmark": (
        "FRAME B — the edge comes from a CONTRAST between this firm and the "
        "others in its cohort, which means the sentence still has to be about "
        "THIS firm: what it is doing, or not doing, that the comparison makes "
        "visible. A statement about the cohort alone is a billboard and fails."
    ),
}


@dataclass
class Opener:
    opener: str = ""
    hook_used: str = ""
    confidence: float = 0.0
    risk_flags: list[str] | None = None
    variant: str = ""
    variant_name: str = ""
    status: str = "ok"          # ok | needs_review | skipped | error
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


# --- A/B assignment ----------------------------------------------------------

def assign_variant(firm_slug: str, experiment_id: str) -> str:
    """Stable 50/50 split. Same firm, same arm, every run, forever."""
    h = hashlib.sha256(f"{experiment_id}:{firm_slug}".encode()).hexdigest()
    keys = sorted(C.VARIANTS)
    return keys[int(h[:8], 16) % len(keys)]


# --- Context assembly --------------------------------------------------------

def register_for(row: pd.Series) -> str:
    """Tier A principal-led firms get the audio brief; everyone else, email."""
    return REGISTER_SPOKEN if row.get("route") == "A_PRINCIPAL" else REGISTER_WRITTEN


def build_context(row: pd.Series) -> str:
    """Only facts that survived normalization reach the model. The notes field
    is passed verbatim because it carries the warm-context gold, but it is
    labelled as internal so the model does not quote it back at the prospect."""
    def fmt(v, suffix=""):
        if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
            return "unknown"
        return f"{v}{suffix}"

    lines = [
        f"Firm: {row['firm_name']}",
        f"Type: {row['firm_type']}",
        f"AUM: {fmt(round(row['aum_b'], 2) if pd.notna(row.get('aum_b')) else None, 'B USD')}",
        f"Location: {fmt(row.get('city'))}, {fmt(row.get('state'))}",
        f"Contact: {fmt(row.get('primary_contact'))} — {fmt(row.get('title'))} "
        f"(persona: {row.get('persona')})",
        f"Current alternatives sleeves: {fmt(row.get('alts_raw'))} "
        f"(posture: {row.get('alts_posture')})",
        f"Average client size: {fmt(row.get('avg_client_size_m'), 'M USD net worth')}",
        f"Custodian: {fmt(row.get('custodian'))}",
        f"ICP signals detected: {fmt(row.get('signals'))}",
        f"Internal CRM note (context only, never quote): {fmt(row.get('notes'))}",
    ]
    return "\n".join(lines)


def _cache_key(context: str, variant: str, model: str, register: str = "") -> str:
    payload = f"{PROMPT_VERSION}|{model}|{variant}|{register}|{SYSTEM_PROMPT}|{context}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _cost(model: str, pin: int, pout: int) -> float:
    p = C.MODEL_PRICING.get(model, {"in": 0.0, "out": 0.0})
    return round((pin * p["in"] + pout * p["out"]) / 1e6, 6)


# --- Offline renderer --------------------------------------------------------

def _offline_opener(row: pd.Series, variant_name: str) -> Opener:
    """
    Deterministic stand-in so the pipeline is runnable and reviewable without
    credentials. Same input/output contract as the live path. It is a template,
    and it reads like one — that is the point: it demonstrates the plumbing,
    not the copy.
    """
    firm = row["firm_name"]
    sleeves = (row.get("alts_raw") or "").strip()
    aum = row.get("aum_b")
    aum_s = f"${aum:,.1f}B" if pd.notna(aum) else "your current book"
    acs = row.get("avg_client_size_m")
    acs_s = f"${acs:,.0f}M" if pd.notna(acs) else "$5M+"
    first = (row.get("primary_contact") or "").split(" ")[0] or "there"

    if variant_name == "filings_anchored":
        s1 = (f"{firm} is running {sleeves or 'an early alts allocation'} across "
              f"{aum_s} — a sleeve mix that usually starts straining once clients "
              f"at {acs_s} ask for quarterly liquidity.")
        s2 = ("I put together a short read on how that posture maps to evergreen "
              "structures, including the client-facing materials we'd build under "
              "your brand; happy to send it over with nothing attached to it.")
        hook = "own alts sleeve mix vs. liquidity expectations"
    else:
        s1 = (f"Across firms in the {aum_s} band serving {acs_s} clients, most are "
              f"now carrying two or three alts sleeves — {firm} is in that cohort, "
              "and the spread in how they're structured is wide.")
        s2 = ("I have the cohort comparison written up, evergreen versus lock-up "
              "and who's white-labelling it under their own brand; I can send it "
              "across if it's useful.")
        hook = "AUM-band cohort comparison"

    opener = f"{s1} {s2}"
    return Opener(opener=opener, hook_used=hook, confidence=0.55,
                  risk_flags=["offline_template"], status="ok",
                  model="offline-template", cached=False)
    # NOTE: `first` intentionally unused — greetings are the sequencer's job.


# --- Live OpenAI path --------------------------------------------------------

def _avoid_block(used: list[str]) -> str:
    if not used:
        return ""
    recent = used[-12:]
    lines = "\n".join(f"- {u}" for u in recent)
    return ("\n\nThese openings are already taken by other messages in this "
            "same batch. Yours must not start like any of them, and must not "
            "reuse their sentence shape:\n" + lines)


def _call_gemini(client, model: str, context: str, variant_name: str,
                 avoid: str = "", register: str = "") -> tuple[dict, int, int]:
    from google.genai import types
    user_prompt = (
        f"{VARIANT_INSTRUCTIONS[variant_name]}\n\n{register}\n\n"
        f"--- PROSPECT CONTEXT ---\n{context}\n--- END CONTEXT ---"
        f"{avoid}\n\n"
        "Write the opener. JSON only."
    )
    last_err: Exception | None = None
    for attempt in range(C.LLM_MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=C.LLM_TEMPERATURE,
                    response_mime_type="application/json",
                )
            )
            payload = json.loads(resp.text)
            
            # Extraer uso de tokens para el cálculo de Unit Economics
            pin = resp.usage_metadata.prompt_token_count
            pout = resp.usage_metadata.candidates_token_count
            
            return payload, pin, pout
        except Exception as e:                      # noqa: BLE001
            last_err = e
            sleep = (2 ** attempt) + random.random()
            time.sleep(min(sleep, 20))
    raise RuntimeError(f"Gemini call failed after {C.LLM_MAX_RETRIES} attempts: {last_err}")


BANNED = ("hope this finds you well", "reaching out", "circle back",
          "touch base", "quick question", "game-chang", "synerg",
          "i'd love to", "i would love to", "best-in-class")

# Openings that mark the message as machine-written. Caught mechanically so a
# drifting prompt cannot quietly reintroduce the template.
TEMPLATE_OPENINGS = ("managing ", "running ", "given ", "with ", "as a ",
                     "having ", "overseeing ")
# A category claim dressed as an insight. True of a hundred firms, so it is
# not an opener for any of them.
GENERIC_OPENINGS = ("most ", "many ", "firms that", "advisors who", "rias ",
                    "multi-family offices ", "in our experience",
                    "based on my experience", "a lot of ")
TEMPLATE_SECOND = ("i can send", "i can share", "i have a", "i put together",
                   "i mapped", "we recently", "i'd be happy to",
                   # Leading the second sentence with the vendor turns a peer
                   # note into a brochure, whatever the first sentence did.
                   "we built", "we offer", "we help", "we provide", "equi ")


def _split_sentences(text: str) -> list[str]:
    """Split on real sentence boundaries, not on every period.

    "$3.05B" is one token, not two sentences. Splitting naively on "." made
    the validator flag perfectly good two-sentence openers as three, which is
    how a decimal point ends up quarantining a lead.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[\"\u201c(]?[A-Z])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def validate(opener: str) -> list[str]:
    """Cheap guardrails. Anything flagged here routes to human review instead
    of to the sequencer."""
    problems = []
    text = opener.strip()
    if not text:
        return ["empty"]
    if len(text.split()) > 70:
        problems.append("too_long")
    sentences = _split_sentences(text)
    if len(sentences) > 2:
        problems.append("sentence_count")
    low = text.lower()
    problems += [f"banned_phrase:{b}" for b in BANNED if b in low]
    if "[" in text or "{" in text or "XX" in text:
        problems.append("unfilled_placeholder")
    if low.startswith(TEMPLATE_OPENINGS):
        problems.append("template_opening")
    if low.startswith(GENERIC_OPENINGS):
        problems.append("generic_category_opening")
    # Word boundary, or "private equity" flags every alts firm on the list.
    if sentences and re.search(r"\bequi\b", sentences[0], re.I):
        problems.append("product_in_first_sentence")
    # The second sentence starting with an offer is the other half of the tell.
    if len(sentences) > 1 and sentences[1].lower().startswith(TEMPLATE_SECOND):
        problems.append("template_offer")
    return problems


class Personalizer:
    def __init__(self, cfg: C.RunConfig):
        self.cfg = cfg
        # Each call is independent, so nothing stops nineteen of them from
        # reaching for the same opening. We carry the openings already used in
        # this run into the next prompt and forbid them. Variety is a property
        # of the batch, not of any single message.
        self._used_openings: list[str] = []
        self.cache = self._load_cache()
        self.client = None
        if not cfg.offline:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. Run with --offline to use the "
                    "deterministic template renderer instead."
                )
            from google import genai
            self.client = genai.Client(api_key=api_key)

    def _cache_path(self) -> str:
        return os.path.join(self.cfg.out_dir, ".opener_cache.json")

    def _load_cache(self) -> dict:
        try:
            with open(self._cache_path(), encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_cache(self) -> None:
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        with open(self._cache_path(), "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def generate(self, row: pd.Series) -> Opener:
        variant = assign_variant(row["firm_slug"], self.cfg.experiment_id)
        variant_name = C.VARIANTS[variant]
        context = build_context(row)

        if self.cfg.offline:
            op = _offline_opener(row, variant_name)
            op.variant, op.variant_name = variant, variant_name
            op.risk_flags = (op.risk_flags or []) + validate(op.opener)
            return op

        key = _cache_key(context, variant_name, self.cfg.model,
                         register_for(row))
        if key in self.cache:
            op = Opener(**self.cache[key])
            op.cached = True
            op.cost_usd = 0.0
            # The cache stores the TEXT, not the verdict. Validation rules get
            # tightened as we learn what the model slips into, so re-run them
            # on every hit — otherwise a fixed rule never reaches old rows.
            model_flags = [f for f in (op.risk_flags or [])
                           if not f.startswith(("template_", "generic_",
                                                "product_", "banned_",
                                                "sentence_", "unfilled_"))]
            problems = validate(op.opener)
            op.risk_flags = model_flags + problems
            op.status = "needs_review" if problems else "ok"
            return op

        try:
            payload, pin, pout = _call_gemini(
                self.client, self.cfg.model, context, variant_name,
                _avoid_block(self._used_openings), register_for(row))
        except Exception as e:                       # noqa: BLE001
            return Opener(variant=variant, variant_name=variant_name,
                          status="error", risk_flags=[str(e)[:160]],
                          model=self.cfg.model)

        text = str(payload.get("opener", "")).strip()
        problems = validate(text)
        op = Opener(
            opener=text,
            hook_used=str(payload.get("hook_used", ""))[:200],
            confidence=float(payload.get("confidence", 0) or 0),
            risk_flags=list(payload.get("risk_flags") or []) + problems,
            variant=variant,
            variant_name=variant_name,
            status="needs_review" if problems else "ok",
            model=self.cfg.model,
            prompt_tokens=pin,
            completion_tokens=pout,
            cost_usd=_cost(self.cfg.model, pin, pout),
        )
        if text:
            self._used_openings.append(" ".join(text.split()[:7]))
        self.cache[key] = asdict(op) | {"cached": False}
        return op
