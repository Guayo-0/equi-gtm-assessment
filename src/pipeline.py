"""
Orchestrator / CLI.

    python -m src.pipeline --input data/raw/sample-leads.csv --offline
    OPENAI_API_KEY=sk-... python -m src.pipeline --model gpt-4o-mini

Stages: load -> normalize -> resolve entities -> score -> route ->
personalize (gated) -> emit.

Emits five files into --out:
    scored_leads.csv        every firm, full breakdown, audit-ready
    shortlist.csv           tier A/B with openers — the thing a rep opens
    merge_audit.csv         which records were collapsed and why
    enrichment_queue.csv    blocked rows with the specific missing field
    run_report.json         counts, cost, A/B split, data-quality stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime

import pandas as pd

from . import config as C
from .dedupe import resolve_entities
from .normalize import normalize_frame
from .personalize import Personalizer
from .route import route_frame
from .score import score_frame

# utf-8-sig writes a BOM. Without it Excel opens these as ANSI and every em
# dash in score_rationale renders as "â€"" — on the one column a reviewer
# actually reads.
CSV_ENCODING = "utf-8-sig"

SHORTLIST_COLUMNS = [
    "rank", "tier", "icp_score", "firm_name", "firm_type", "aum_b",
    "city", "state", "primary_contact", "title", "persona", "email",
    "decision_topology", "signals", "ab_variant", "ab_variant_name",
    "opener", "opener_status", "opener_hook", "opener_risk_flags",
    "sequence", "owner", "first_touch", "score_rationale",
]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def run(cfg: C.RunConfig) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    today = date.today()

    raw = load(cfg.input_path)
    norm = normalize_frame(raw)
    resolved, audit = resolve_entities(norm)
    scored = score_frame(resolved, today=today)
    scored["signals"] = scored["signals"].fillna("")
    routed = route_frame(scored)

    # --- personalization gate ------------------------------------------------
    mask = (routed["icp_score"] >= cfg.personalize_min) & routed["eligible"]
    mask &= ~routed["route"].isin(["ENRICH", "SUPPRESS"])
    targets = routed[mask]
    if cfg.max_personalize:
        targets = targets.head(cfg.max_personalize)

    personalizer = Personalizer(cfg)
    cols = {c: "" for c in ("opener", "opener_hook", "opener_status",
                            "opener_risk_flags", "ab_variant", "ab_variant_name",
                            "opener_model")}
    for c, v in cols.items():
        routed[c] = v
    routed["opener_confidence"] = 0.0
    routed["opener_cost_usd"] = 0.0

    total_cost, calls, cached = 0.0, 0, 0
    for idx, row in targets.iterrows():
        op = personalizer.generate(row)
        routed.at[idx, "opener"] = op.opener
        routed.at[idx, "opener_hook"] = op.hook_used
        routed.at[idx, "opener_status"] = op.status
        routed.at[idx, "opener_risk_flags"] = "|".join(op.risk_flags or [])
        routed.at[idx, "opener_confidence"] = op.confidence
        routed.at[idx, "ab_variant"] = op.variant
        routed.at[idx, "ab_variant_name"] = op.variant_name
        routed.at[idx, "opener_model"] = op.model
        routed.at[idx, "opener_cost_usd"] = op.cost_usd
        total_cost += op.cost_usd
        calls += 0 if op.cached else 1
        cached += 1 if op.cached else 0
    if not cfg.offline:
        personalizer.save_cache()

    routed.loc[~mask, "opener_status"] = routed.loc[~mask, "route"].map(
        lambda r: "suppressed" if r == "SUPPRESS" else
                  ("blocked_on_enrichment" if r == "ENRICH" else "below_threshold"))

    # --- emit ---------------------------------------------------------------
    scored_path = os.path.join(cfg.out_dir, "scored_leads.csv")
    routed.to_csv(scored_path, index=False, encoding=CSV_ENCODING)

    shortlist = routed[routed["tier"].isin(["A", "B"])].copy()
    shortlist.insert(0, "rank", range(1, len(shortlist) + 1))
    shortlist = shortlist.reindex(columns=SHORTLIST_COLUMNS)
    shortlist.to_csv(os.path.join(cfg.out_dir, "shortlist.csv"),
                     index=False, encoding=CSV_ENCODING)

    audit.to_csv(os.path.join(cfg.out_dir, "merge_audit.csv"),
                 index=False, encoding=CSV_ENCODING)

    enrich = routed[routed["route"] == "ENRICH"][
        ["firm_name", "tier", "icp_score", "signals", "aum_raw",
         "primary_contact", "email", "_parse_flags"]]
    enrich.to_csv(os.path.join(cfg.out_dir, "enrichment_queue.csv"),
                  index=False, encoding=CSV_ENCODING)

    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "score_model_version": C.SCORE_MODEL_VERSION,
        "experiment_id": cfg.experiment_id,
        "mode": "offline-template" if cfg.offline else cfg.model,
        "input_rows": int(len(raw)),
        "rows_after_entity_resolution": int(len(resolved)),
        "duplicate_clusters_collapsed": int(len(audit)),
        "eligible_firms": int(routed["eligible"].sum()),
        "disqualified": int((~routed["eligible"]).sum()),
        "disqualify_reasons": routed.loc[~routed["eligible"], "disqualify_reason"]
            .value_counts().to_dict(),
        "tier_counts": routed["tier"].value_counts().to_dict(),
        "route_counts": routed["route"].value_counts().to_dict(),
        "openers_generated": int(
            (routed.loc[mask, "opener"].fillna("").str.len() > 0).sum()),
        "openers_attempted": int(len(targets)),
        # Fail loudly: a run where every call 404s still reports 19 "calls".
        # These two lines are what make that visible without reading the CSV.
        "opener_status_counts": routed.loc[mask, "opener_status"]
            .value_counts().to_dict(),
        "opener_errors": int((routed.loc[mask, "opener_status"] == "error").sum()),
        "llm_calls": calls,
        "cache_hits": cached,
        "llm_cost_usd": round(total_cost, 4),
        "cost_per_shortlisted_lead_usd": round(
            total_cost / max(1, len(targets)), 6),
        "ab_split": routed.loc[mask, "ab_variant"].value_counts().to_dict(),
        "needs_review_openers": int(
            (routed["opener_status"] == "needs_review").sum()),
        "data_quality": {
            "missing_aum": int(routed["aum_b"].isna().sum()),
            "missing_email": int((routed["email"] == "").sum()),
            "missing_client_size": int(routed["avg_client_size_m"].isna().sum()),
            "parse_flags": routed["_parse_flags"].str.split("|").explode()
                .replace("", pd.NA).dropna().value_counts().to_dict(),
        },
        "top_10": routed.head(10)[["firm_name", "icp_score", "tier"]]
            .to_dict(orient="records"),
    }
    with open(os.path.join(cfg.out_dir, "run_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    if report["opener_errors"]:
        sample = routed.loc[routed["opener_status"] == "error",
                            "opener_risk_flags"].iloc[0][:200]
        print(f"\n*** WARNING: {report['opener_errors']} openers failed to "
              f"generate. The shortlist has empty opener cells.\n"
              f"*** First error: {sample}\n", file=sys.stderr)

    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Equi ICP lead pipeline")
    p.add_argument("--input", default="data/raw/sample-leads.csv")
    p.add_argument("--out", default="out")
    p.add_argument("--model", default=C.DEFAULT_MODEL)
    p.add_argument("--offline", action="store_true",
                   help="deterministic templates, no API key required")
    p.add_argument("--threshold", type=float, default=C.PERSONALIZE_MIN)
    p.add_argument("--max-personalize", type=int, default=None)
    p.add_argument("--experiment-id", default=C.EXPERIMENT_ID)
    args = p.parse_args()

    cfg = C.RunConfig(
        input_path=args.input, out_dir=args.out, model=args.model,
        offline=args.offline, personalize_min=args.threshold,
        max_personalize=args.max_personalize, experiment_id=args.experiment_id,
    )
    report = run(cfg)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
