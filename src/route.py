"""
Routing and sequencing. Scoring without routing is a spreadsheet; this is the
part that decides what actually happens to each firm tomorrow morning.

Channel assignment is driven by two axes, not by score alone:
  - tier              -> how much human attention the firm earns
  - decision topology -> principal-led firms get closed, committee-led firms
                         get equipped (the champion has to defend the pick)

Data-quality signals pre-empt both: a Tier A firm with no routable email goes
to the enrichment queue, not to a sender. Sending into a gatekeeper inbox
burns the domain and the firm.
"""

from __future__ import annotations

import pandas as pd

# Any of these means we cannot responsibly send yet, whatever the score says.
BLOCKING_SIGNALS = ("NEEDS_CONTACT", "NEEDS_EMAIL", "NEEDS_AUM_ENRICHMENT",
                    "NEEDS_DECISION_MAKER")

SEQUENCES = {
    "A_PRINCIPAL": {
        "name": "founder_direct_audio",
        "owner": "AE",
        "touches": [
            "D0  email — LLM opener + 90-sec AI audio brief on their own alts posture",
            "D2  LinkedIn connect, no note",
            "D5  email — white-label alts infrastructure one-pager under their brand",
            "D9  call to the direct line",
            "D14 breakup: send the materials anyway, no ask",
        ],
        "qc": "human review before send",
    },
    "A_COMMITTEE": {
        "name": "champion_enablement",
        "owner": "AE",
        "touches": [
            "D0  email — LLM opener framed as diligence input, not a pitch",
            "D3  send the IC-ready memo (structure, risk, liquidity terms)",
            "D7  offer a 20-min technical call with the PM, not a sales call",
            "D12 forwardable summary the champion can circulate internally",
            "D18 breakup",
        ],
        "qc": "human review before send",
    },
    "B_STANDARD": {
        "name": "signal_sequence_4touch",
        "owner": "SDR",
        "touches": [
            "D0  email — LLM opener (A/B arm assigned)",
            "D4  email — cohort comparison asset",
            "D8  LinkedIn touch",
            "D13 breakup",
        ],
        "qc": "spot-check 20%",
    },
    "C_NURTURE": {
        "name": "content_nurture",
        "owner": "Marketing",
        "touches": [
            "add to evergreen-alts newsletter cohort",
            "retarget on answer-engine content",
            "re-score quarterly on ADV refresh",
        ],
        "qc": "none — no 1:1 sending",
    },
    "ENRICH": {
        "name": "enrichment_queue",
        "owner": "RevOps",
        "touches": ["resolve missing AUM / contact / email before any send"],
        "qc": "blocked",
    },
    "SUPPRESS": {
        "name": "suppressed",
        "owner": "—",
        "touches": ["do not contact"],
        "qc": "n/a",
    },
}


def _is_committee_led(row: pd.Series) -> bool:
    notes = str(row.get("notes", "")).lower()
    if any(t in notes for t in ("committee", "ic-driven", "ic driven", "slow")):
        return True
    return row.get("persona") in {"RESEARCH_LEAD", "SENIOR_IC"}


def route_row(row: pd.Series) -> dict:
    tier = row["tier"]
    signals = str(row.get("signals", ""))

    if tier == "DISQUALIFIED":
        key = "SUPPRESS"
    elif any(s in signals for s in BLOCKING_SIGNALS) and tier in {"A", "B", "C"}:
        key = "ENRICH"
    elif tier == "A":
        key = "A_COMMITTEE" if _is_committee_led(row) else "A_PRINCIPAL"
    elif tier == "B":
        key = "B_STANDARD"
    elif tier == "C":
        key = "C_NURTURE"
    else:
        key = "SUPPRESS"

    seq = SEQUENCES[key]
    return {
        "route": key,
        "sequence": seq["name"],
        "owner": seq["owner"],
        "qc_policy": seq["qc"],
        "first_touch": seq["touches"][0],
        "decision_topology": "committee-led" if _is_committee_led(row) else "principal-led",
    }


def route_frame(df: pd.DataFrame) -> pd.DataFrame:
    routed = pd.DataFrame([route_row(r) for _, r in df.iterrows()], index=df.index)
    return pd.concat([df, routed], axis=1)
