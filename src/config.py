"""
Single source of truth for ICP policy. Everything tunable lives here so the
scoring model can be versioned, diffed and A/B tested without touching logic.

SCORE_MODEL_VERSION is stamped onto every output row. Never mutate weights
in place without bumping it — downstream reply-rate analysis is keyed on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SCORE_MODEL_VERSION = "icp-v1.3.0"

# --- Band policy -------------------------------------------------------------
AUM_FLOOR_B = 1.0          # ICP band lower bound (billions)
AUM_CEILING_B = 30.0       # above this, firms in-source; structurally unsellable
AUM_HARD_FLOOR_B = 0.75    # below this, disqualify outright (not even nurture)
AUM_BULLSEYE_B = 5.0

QP_FLOOR_M = 5.0           # qualified purchaser gate (client net worth, millions)
QP_SWEET_MAX_M = 30.0

# --- Firm type taxonomy ------------------------------------------------------
# Raw strings in the CSV are dirty ("ria", "multi-family office", "MFO").
# normalize.py maps them to these canonical keys.
FIRM_TYPE_SYNONYMS: dict[str, str] = {
    "ria": "RIA",
    "registered investment advisor": "RIA",
    "registered investment adviser": "RIA",
    "independent ria": "RIA",
    "mfo": "MFO",
    "multi-family office": "MFO",
    "multi family office": "MFO",
    "multifamily office": "MFO",
    "family office": "MFO",          # ambiguous; name/notes heuristics refine it
    "single family office": "SFO",
    "sfo": "SFO",
    "wirehouse": "WIREHOUSE",
    "tamp": "TAMP",
    "broker-dealer": "BROKER_DEALER",
    "broker dealer": "BROKER_DEALER",
    "bd": "BROKER_DEALER",
    "hedge fund manager": "ASSET_MANAGER",
    "hedge fund": "ASSET_MANAGER",
    "private equity": "ASSET_MANAGER",
    "pe firm": "ASSET_MANAGER",
    "private credit manager": "ASSET_MANAGER",
    "asset manager": "ASSET_MANAGER",
    "bank": "BANK_TRUST",
    "trust company": "BANK_TRUST",
}

# Structural disqualifiers: wrong side of the trade or wrong economics.
DISQUALIFYING_TYPES = {"WIREHOUSE", "TAMP", "BROKER_DEALER", "ASSET_MANAGER"}

# Jurisdiction: fund structures are US-domiciled; non-US is a different
# distribution/regulatory problem, not a scoring penalty.
ALLOWED_COUNTRIES = {"USA", "US", "UNITED STATES", ""}

# Captive custodial platforms where third-party alts are gated by home office.
CAPTIVE_CUSTODIANS = {"LPL FINANCIAL", "LPL", "IN-HOUSE"}
OPEN_CUSTODIANS = {"SCHWAB", "CHARLES SCHWAB", "FIDELITY", "PERSHING",
                   "BNY PERSHING", "NORTHERN TRUST", "MULTIPLE", "RBC"}


@dataclass(frozen=True)
class Weights:
    """Max points each dimension can contribute. Sums to 100 on the base model;
    signal bonuses push the ceiling to ~120 so genuine outliers separate from
    the pack instead of saturating at 100."""
    aum_fit: int = 30
    firm_type_fit: int = 25
    client_qualification: int = 20
    alts_posture: int = 15
    decision_velocity: int = 10

    # Additive signals (can exceed the 100 base)
    bonus_mfo_transition: int = 12      # the stated strongest signal
    bonus_warm_context: int = 5
    bonus_open_custodian: int = 3

    # Penalties
    penalty_captive_custodian: int = -8
    penalty_competitor_installed: int = -10
    penalty_stale_unknown_aum: int = -5


WEIGHTS = Weights()

# --- Routing thresholds ------------------------------------------------------
# Calibrated against the seed list so the shortlist stays a shortlist: on the
# 50-row sample this yields ~8 Tier A / ~12 Tier B out of 35 eligible firms.
# Re-calibrate whenever weights move — a tier boundary is a capacity decision
# (how many firms one AE can work in a fortnight), not a statistical one.
TIER_A_MIN = 100    # founder-direct, audio brief, manual QC before send
TIER_B_MIN = 80     # LLM opener + automated 4-touch sequence
TIER_C_MIN = 62     # nurture: content + answer-engine retargeting, no 1:1
PERSONALIZE_MIN = TIER_B_MIN  # LLM spend gate

# Hard cap for leads missing AUM: never let a prior-filled row outrank a
# verified one. Forces enrichment before a human touches it.
UNVERIFIED_AUM_TIER_CAP = TIER_A_MIN - 1

# Sub-band firms (< $1B) can look excellent on every other axis. Band
# discipline is a policy, not a weight: cap them below the 1:1 send tiers.
BELOW_BAND_TIER_CAP = TIER_B_MIN - 1

# --- Persona model -----------------------------------------------------------
# Ranked by "can this person say yes, and how fast". Used both for scoring and
# for picking the survivor contact when deduping a firm.
PERSONA_RANK: list[tuple[str, str, int]] = [
    # (regex-ish token, canonical persona, decision velocity points 0-10)
    ("founder", "PRINCIPAL", 10),
    ("principal", "PRINCIPAL", 10),
    ("managing partner", "PRINCIPAL", 10),
    ("owner", "PRINCIPAL", 9),
    ("president", "PRINCIPAL", 9),
    ("managing director", "SENIOR_IC", 7),
    ("chief investment officer", "CIO", 9),
    ("cio", "CIO", 9),
    ("head of investments", "CIO", 8),
    ("head of external managers", "RESEARCH_LEAD", 7),
    ("head of manager selection", "RESEARCH_LEAD", 7),
    ("director of research", "RESEARCH_LEAD", 6),
    ("dir. of research", "RESEARCH_LEAD", 6),
    ("director of manager research", "RESEARCH_LEAD", 6),
    ("managing director of research", "RESEARCH_LEAD", 6),
    ("director of due diligence", "RESEARCH_LEAD", 6),
    ("director of investment solutions", "RESEARCH_LEAD", 5),
    ("head of alternative investments", "RESEARCH_LEAD", 6),
    ("portfolio manager", "SENIOR_IC", 4),
    ("svp", "SENIOR_IC", 3),
    ("ea to", "GATEKEEPER", 1),
    ("executive assistant", "GATEKEEPER", 1),
]

GENERIC_INBOX_PREFIXES = {"info", "contact", "hello", "admin", "office", "team"}

# --- Signal lexicons ---------------------------------------------------------
# RIA -> MFO transition. No single field carries it, so we score the
# conjunction of weak signals (name semantics + client size + service mix).
MFO_NAME_TOKENS = ("family office", "family advisors", "private wealth",
                   "private capital", "family wealth", "trust company",
                   "private office", "collective")

COMPETITOR_TOKENS = ("competitor platform", "competing platform",
                     "adopted a competitor", "uses icapital", "cais")

WARM_TOKENS = ("warm intro", "met at", "inbound", "referral", "webinar",
               "conference", "future proof")

# Alts exposure buckets -> (points, posture label)
ALTS_NONE = "NONE"
ALTS_LIGHT = "LIGHT"
ALTS_MODERATE = "MODERATE"
ALTS_HEAVY = "HEAVY"
ALTS_UNKNOWN = "UNKNOWN"

ALTS_POINTS = {
    ALTS_MODERATE: 15,   # the middle is where Equi wins
    ALTS_LIGHT: 12,
    ALTS_UNKNOWN: 8,
    ALTS_HEAVY: 5,       # has own infrastructure
    ALTS_NONE: 6,        # needs us most, longest education curve -> pipeline
}

HEAVY_ALTS_TOKENS = ("everything", "infra", "venture", "hedge funds")

# --- LLM -------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3.6-flash"
LLM_TEMPERATURE = 0.4
LLM_MAX_TOKENS = 220
LLM_MAX_RETRIES = 4
LLM_CACHE_PATH = "out/.opener_cache.json"

# Unit economics for the cost report: USD per 1M tokens, standard paid tier.
# A model absent from this table reports $0.00 cost, which is a silent lie in
# run_report.json — add the model here the same commit you start using it.
# Note: 3.8/3.6 Flash pricing doubles on 2027-01-01 ($1.50 in / $7.50 out).
MODEL_PRICING = {
    "gemini-3.8-flash": {"in": 0.75, "out": 3.75},
    "gemini-3.6-flash": {"in": 0.75, "out": 3.75},
    "gemini-3.5-flash-lite": {"in": 0.30, "out": 2.50},
    # Legacy 2.5 line: served only for API keys that already used it.
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "gemini-2.5-pro": {"in": 1.25, "out": 10.00},
}

# --- A/B framework -----------------------------------------------------------
# Deterministic assignment: same firm always lands in the same arm across runs,
# so re-runs don't contaminate the experiment.
EXPERIMENT_ID = "opener_frame_v1"
VARIANTS: dict[str, str] = {
    "A": "filings_anchored",   # read of their own public alts positioning
    "B": "peer_benchmark",     # where they sit vs. comparable firms in band
}


@dataclass
class RunConfig:
    input_path: str = "data/raw/sample-leads.csv"
    out_dir: str = "out"
    model: str = DEFAULT_MODEL
    offline: bool = False
    personalize_min: int = PERSONALIZE_MIN
    max_personalize: int | None = None
    experiment_id: str = EXPERIMENT_ID
    seed: int = 1337
    weights: Weights = field(default_factory=lambda: WEIGHTS)
