"""
ICP scoring. Two stages, deliberately separated:

  Stage 1 — Eligibility (boolean). Structural facts that make a firm
  unsellable: wrong side of the trade, outside the band, wrong jurisdiction.
  These are not penalties. A 340B wirehouse should not out-rank a 6B RIA
  because it has a nice title in the contact column.

  Stage 2 — Fit (continuous, 0..~120). Weighted additive model over five
  dimensions plus signal bonuses.

Every component writes a line into `score_breakdown`, so the output answers
"why did this firm land here" without re-running anything.

Why additive and not a classifier: there is no labelled outcome data yet. An
additive model with explicit weights is auditable, arguable in a GTM meeting,
and can be re-weighted the moment reply data exists. The first 200 replies are
the training set; until then, a model would be fitting noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from . import config as C


@dataclass
class ScoreResult:
    score: float = 0.0
    eligible: bool = True
    disqualify_reason: str = ""
    components: dict[str, float] = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)

    def add(self, name: str, points: float, why: str) -> None:
        # Round at the point of addition, not at display time. Printing
        # "+16" while adding 16.5 makes the rationale fail the simplest audit
        # there is: summing the column by hand and comparing to the score.
        points = round(points, 1)
        self.components[name] = points
        self.score += points
        self.rationale.append(f"{name} {points:+g}: {why}")


# --- Stage 1: eligibility ----------------------------------------------------

def check_eligibility(row: pd.Series) -> tuple[bool, str]:
    ft = row["firm_type"]
    if ft in C.DISQUALIFYING_TYPES:
        reason = {
            "WIREHOUSE": "wirehouse — home-office gated, model does not fit",
            "TAMP": "TAMP — competes with our distribution layer",
            "BROKER_DEALER": "broker-dealer — product-shelf economics, not a partner",
            "ASSET_MANAGER": "investment-side manager, not a buyer",
        }[ft]
        return False, reason

    country = row.get("country") or "USA"
    if country.upper() not in C.ALLOWED_COUNTRIES:
        return False, f"non-US domicile ({country}) — out of scope for US fund structures"

    aum = row.get("aum_b")
    if pd.notna(aum) and aum is not None:
        if aum > C.AUM_CEILING_B:
            return False, (f"${aum:,.1f}B > ${C.AUM_CEILING_B:.0f}B ceiling — "
                           "in-sources managers, no room to add value")
        if aum < C.AUM_HARD_FLOOR_B:
            return False, f"${aum:,.2f}B below hard floor — sub-scale for the fund minimums"
    return True, ""


# --- Stage 2: fit components -------------------------------------------------

def _score_aum(row: pd.Series, r: ScoreResult) -> None:
    w = C.WEIGHTS.aum_fit
    aum = row.get("aum_b")
    if aum is None or pd.isna(aum):
        # Prior, not a guess: median of the in-band population. Capped tier
        # downstream so an unverified firm can never reach Tier A.
        r.add("aum_fit", w * 0.40, "AUM undisclosed — prior applied, enrichment required")
        r.signals.append("NEEDS_AUM_ENRICHMENT")
        return

    if C.AUM_BULLSEYE_B <= aum <= 20:
        pts, why = w, f"${aum:,.1f}B — bullseye band"
    elif 20 < aum <= C.AUM_CEILING_B:
        # Linear decay toward the ceiling: still sellable, shrinking headroom.
        frac = 1 - (aum - 20) / (C.AUM_CEILING_B - 20)
        pts, why = w * (0.60 + 0.25 * frac), f"${aum:,.1f}B — upper band, approaching ceiling"
    elif 3.0 <= aum < C.AUM_BULLSEYE_B:
        pts, why = w * 0.80, f"${aum:,.1f}B — in band, just under bullseye"
    elif C.AUM_FLOOR_B <= aum < 3.0:
        pts, why = w * 0.55, f"${aum:,.1f}B — lower band"
    else:
        pts, why = w * 0.15, f"${aum:,.2f}B — below the ${C.AUM_FLOOR_B:.0f}B floor"
    r.add("aum_fit", pts, why)


def _score_firm_type(row: pd.Series, r: ScoreResult) -> None:
    w = C.WEIGHTS.firm_type_fit
    table = {
        "MFO": (1.00, "multi-family office — core buyer profile"),
        "RIA": (0.88, "independent RIA — core buyer profile"),
        "BANK_TRUST": (0.55, "trust company — buyer-side but slower governance"),
        "SFO": (0.32, "single-family office — weaker fit than MFO"),
        "UNKNOWN": (0.40, "firm type unresolved — classify before outreach"),
    }
    mult, why = table.get(row["firm_type"], (0.30, "atypical firm type"))
    r.add("firm_type_fit", w * mult, why)
    if row["firm_type"] == "UNKNOWN":
        r.signals.append("NEEDS_TYPE_CLASSIFICATION")


def _score_client_qualification(row: pd.Series, r: ScoreResult) -> None:
    """Qualified-purchaser density is the eligibility gate for the structures,
    so this is a hard economic constraint wearing a soft-score costume."""
    w = C.WEIGHTS.client_qualification
    acs = row.get("avg_client_size_m")
    if acs is None or pd.isna(acs):
        r.add("client_qualification", w * 0.40,
              "avg client size unknown — infer from ADV Part 1 client mix")
        r.signals.append("NEEDS_CLIENT_SIZE_ENRICHMENT")
        return
    if C.QP_FLOOR_M <= acs <= C.QP_SWEET_MAX_M:
        pts, why = w, f"${acs:,.1f}M avg client — inside the QP sweet spot"
    elif acs > C.QP_SWEET_MAX_M:
        pts, why = w * 0.70, f"${acs:,.1f}M avg client — above sweet spot, institutional expectations"
    elif 3.0 <= acs < C.QP_FLOOR_M:
        pts, why = w * 0.45, f"${acs:,.1f}M avg client — partial QP density"
    else:
        pts, why = w * 0.10, f"${acs:,.1f}M avg client — thin QP density, structures gate out"
    r.add("client_qualification", pts, why)


def _score_alts_posture(row: pd.Series, r: ScoreResult) -> None:
    posture = row["alts_posture"]
    pts = C.ALTS_POINTS.get(posture, 8) * (C.WEIGHTS.alts_posture / 15)
    why = {
        C.ALTS_MODERATE: "allocating across a couple of sleeves — the middle, where we win",
        C.ALTS_LIGHT: "early alts programme — short education curve",
        C.ALTS_HEAVY: "alts-native — likely has its own sourcing infrastructure",
        C.ALTS_NONE: "no alts today — highest need, longest education curve (pipeline, not quick win)",
        C.ALTS_UNKNOWN: "alts posture unknown",
    }[posture]
    r.add("alts_posture", pts, why)


def _score_decision_velocity(row: pd.Series, r: ScoreResult) -> None:
    """Principal-led vs committee-led is the stated best predictor of speed."""
    w = C.WEIGHTS.decision_velocity
    persona = row["persona"]
    pts_raw = float(row.get("persona_points") or 0)
    email = row.get("email") or ""

    if not row.get("primary_contact"):
        r.add("decision_velocity", 0, "no named contact — needs identification")
        r.signals.append("NEEDS_CONTACT")
        return
    if "email_generic_inbox" in str(row.get("_parse_flags", "")):
        r.add("decision_velocity", w * 0.10, "generic inbox only — no routable human")
        r.signals.append("NEEDS_CONTACT")
        return

    if persona == "GATEKEEPER":
        # A valid email is not a routable human. Sending the founder-direct
        # audio to an executive assistant spends the one first impression this
        # firm has on the person least able to act on it.
        r.add("decision_velocity", w * 0.10,
              "gatekeeper is the only contact — identify the principal first")
        r.signals.append("NEEDS_DECISION_MAKER")
        return

    pts = w * (pts_raw / 10)
    why = {
        "PRINCIPAL": "principal-led — can say yes without a committee",
        "CIO": "CIO — de facto decision maker, alts-fluent",
        "RESEARCH_LEAD": "research/DD lead — champion to equip, expect internal defence",
        "SENIOR_IC": "senior IC — indirect path to the mandate",
        "GATEKEEPER": "gatekeeper contact — re-route before sending",
        "UNKNOWN": "title unknown",
    }.get(persona, "atypical persona")
    r.add("decision_velocity", pts, why)
    if not email:
        r.signals.append("NEEDS_EMAIL")


# --- Signal bonuses ----------------------------------------------------------

def detect_mfo_transition(row: pd.Series) -> tuple[bool, str]:
    """
    The strongest stated signal: a successful RIA mid-rebrand to MFO.

    No single column carries it, so we score a conjunction of weak signals.
    This is a proxy built from list data only — the production version reads
    ADV Part 1 deltas, domain/brand changes and site copy (see README).
    """
    blob = f"{row.get('firm_name','')} {row.get('notes','')}".lower()
    acs = row.get("avg_client_size_m")
    aum = row.get("aum_b")

    name_signal = any(tok in blob for tok in C.MFO_NAME_TOKENS)
    wealth_signal = acs is not None and not pd.isna(acs) and acs >= 10.0
    scale_signal = aum is not None and not pd.isna(aum) and aum >= 3.0

    if row["firm_type"] == "RIA":
        if wealth_signal and scale_signal:
            return True, (f"RIA at ${aum:,.1f}B serving ${acs:,.0f}M clients — "
                          "economics of an MFO, still branded an RIA")
        if name_signal and scale_signal:
            return True, "RIA carrying family-office naming — mid-repositioning"
    # Deliberately not fired for firms already typed MFO: the signal Equi
    # cares about is the transition itself, and an established MFO is already
    # priced into firm_type_fit. Double-counting it would flatten the ranking.
    return False, ""


def _score_bonuses(row: pd.Series, r: ScoreResult, today: date) -> None:
    is_transition, why = detect_mfo_transition(row)
    if is_transition:
        r.add("signal_mfo_transition", C.WEIGHTS.bonus_mfo_transition, why)
        r.signals.append("MFO_TRANSITION")

    notes = str(row.get("notes", "")).lower()
    if any(tok in notes for tok in C.WARM_TOKENS):
        r.add("signal_warm_context", C.WEIGHTS.bonus_warm_context,
              "prior warm context in the record — open with it, not cold")
        r.signals.append("WARM_CONTEXT")

    cust = row.get("custodian", "")
    if cust in C.CAPTIVE_CUSTODIANS:
        r.add("penalty_captive_custodian", C.WEIGHTS.penalty_captive_custodian,
              f"{cust} — third-party alts gated by home office")
        r.signals.append("CAPTIVE_PLATFORM")
    elif cust in C.OPEN_CUSTODIANS:
        r.add("bonus_open_custodian", C.WEIGHTS.bonus_open_custodian,
              f"{cust} — open architecture, no platform gate")

    if any(tok in notes for tok in C.COMPETITOR_TOKENS):
        r.add("penalty_competitor", C.WEIGHTS.penalty_competitor_installed,
              "competitor platform recently installed — displacement cycle, not a fast win")
        r.signals.append("COMPETITOR_INSTALLED")

    lt = row.get("last_touch")
    if isinstance(lt, date):
        days = (today - lt).days
        if days > 365:
            r.add("recency", -3, f"last touch {days}d ago — relationship is cold")
    elif lt is None:
        r.signals.append("NEVER_TOUCHED")


# --- Public API --------------------------------------------------------------

def score_row(row: pd.Series, today: date | None = None) -> ScoreResult:
    today = today or date.today()
    r = ScoreResult()

    eligible, reason = check_eligibility(row)
    if not eligible:
        r.eligible = False
        r.disqualify_reason = reason
        r.score = 0.0
        r.rationale.append(f"DISQUALIFIED: {reason}")
        return r

    _score_aum(row, r)
    _score_firm_type(row, r)
    _score_client_qualification(row, r)
    _score_alts_posture(row, r)
    _score_decision_velocity(row, r)
    _score_bonuses(row, r, today)

    r.score = max(0.0, round(r.score, 1))
    return r


def assign_tier(score: float, eligible: bool, signals: list[str],
                aum_b: float | None = None) -> str:
    """Tier is score subject to two policy caps. Caps are applied to the
    *effective* score only — the raw score stays visible so a rep can see that
    a capped firm is otherwise excellent."""
    if not eligible:
        return "DISQUALIFIED"
    effective = score
    if "NEEDS_AUM_ENRICHMENT" in signals:
        effective = min(effective, C.UNVERIFIED_AUM_TIER_CAP)
    if aum_b is not None and not pd.isna(aum_b) and aum_b < C.AUM_FLOOR_B:
        effective = min(effective, C.BELOW_BAND_TIER_CAP)
    if effective >= C.TIER_A_MIN:
        return "A"
    if effective >= C.TIER_B_MIN:
        return "B"
    if effective >= C.TIER_C_MIN:
        return "C"
    return "D"


def score_frame(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    out = df.copy()
    results = [score_row(r, today) for _, r in df.iterrows()]

    out["icp_score"] = [r.score for r in results]
    out["eligible"] = [r.eligible for r in results]
    out["disqualify_reason"] = [r.disqualify_reason for r in results]
    out["signals"] = ["|".join(r.signals) for r in results]
    out["tier"] = [assign_tier(r.score, r.eligible, r.signals, row.get("aum_b"))
                   for r, (_, row) in zip(results, df.iterrows())]
    out["score_breakdown"] = [json.dumps(r.components) for r in results]
    out["score_rationale"] = [" ; ".join(r.rationale) for r in results]
    out["score_model_version"] = C.SCORE_MODEL_VERSION

    return out.sort_values(
        ["eligible", "icp_score"], ascending=[False, False]
    ).reset_index(drop=True)
