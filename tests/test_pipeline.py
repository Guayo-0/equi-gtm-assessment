"""
Tests cover the two places where a silent bug becomes a mis-targeted email:
AUM scale inference and entity resolution. Run: pytest -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.dedupe import resolve_entities
from src.normalize import (classify_alts, classify_persona, firm_slug,
                           normalize_email, normalize_frame,
                           parse_aum_billions, parse_last_touch)
from src.personalize import assign_variant, validate
from src.score import assign_tier, check_eligibility, detect_mfo_transition


@pytest.mark.parametrize("raw,expected", [
    ("4.2B", 4.2),
    ("$4,200,000,000", 4.2),
    ("11500", 11.5),            # bare number in millions
    ("2.5 billion", 2.5),
    ("1.4b", 1.4),
    ("6800000000", 6.8),        # bare number in raw USD
    ("880M", 0.88),
    ("340B", 340.0),
    ("3.05B", 3.05),
])
def test_aum_parsing(raw, expected):
    assert parse_aum_billions(raw)[0] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("raw", ["", "n/a", None, "unknown", "lots"])
def test_aum_null_paths_never_raise(raw):
    value, flag = parse_aum_billions(raw)
    assert value is None and flag


def test_bare_number_scale_is_flagged():
    assert parse_aum_billions("11500")[1].startswith("aum_scale_inferred")
    assert parse_aum_billions("6800000000")[1] == ""


@pytest.mark.parametrize("a,b", [
    ("Harborstone Wealth Partners", "HARBORSTONE WEALTH PARTNERS LLC"),
    ("Bramblewick Capital Advisers", "Bramblewick Capital Advisors Inc"),
    ("Haverlock Advisory Group", "Haverlock Advisory Grp."),
])
def test_firm_slug_collapses_variants(a, b):
    assert firm_slug(a) == firm_slug(b)


def test_firm_slug_keeps_distinct_firms_apart():
    assert firm_slug("Oakhurst Meridian Advisors") != firm_slug("Ashcombe Meridian Family Office")


def test_email_repairs_whitespace_defect():
    assert normalize_email("twierzbicki@ostrandervoss .com") == "twierzbicki@ostrandervoss.com"


@pytest.mark.parametrize("raw,posture", [
    ("none", "NONE"),
    ("private credit", "LIGHT"),
    ("PE / RE", "MODERATE"),
    ("hedge funds, private credit, venture", "HEAVY"),
    ("everything", "HEAVY"),
    ("", "UNKNOWN"),
])
def test_alts_classification(raw, posture):
    assert classify_alts(raw)[0] == posture


def test_persona_longest_match_wins():
    assert classify_persona("Founder & CIO")[0] == "PRINCIPAL"
    assert classify_persona("EA to the CIO")[0] == "GATEKEEPER"
    assert classify_persona("Director of Research")[0] == "RESEARCH_LEAD"


@pytest.mark.parametrize("raw", ["2026-03-14", "03/14/2026", "4/11/26",
                                 "Jan 2026", "April 30 2026"])
def test_date_formats(raw):
    assert parse_last_touch(raw) is not None


def test_structural_disqualifiers():
    base = {"firm_type": "WIREHOUSE", "country": "USA", "aum_b": 340.0}
    assert check_eligibility(pd.Series(base))[0] is False
    assert check_eligibility(pd.Series(base | {"firm_type": "RIA", "aum_b": 33.0}))[0] is False
    assert check_eligibility(pd.Series(base | {"firm_type": "MFO", "aum_b": 2.2,
                                               "country": "CANADA"}))[0] is False
    assert check_eligibility(pd.Series(base | {"firm_type": "RIA", "aum_b": 6.8}))[0] is True


def test_mfo_transition_requires_ria_plus_wealth_and_scale():
    row = pd.Series({"firm_type": "RIA", "firm_name": "Bramblewick Capital Advisers",
                     "notes": "", "avg_client_size_m": 11.0, "aum_b": 9.6})
    assert detect_mfo_transition(row)[0] is True
    assert detect_mfo_transition(row.copy().replace({11.0: 2.0}))[0] is False


def test_unverified_aum_cannot_reach_tier_a():
    assert assign_tier(120.0, True, ["NEEDS_AUM_ENRICHMENT"]) != "A"


def test_sub_band_firm_cannot_reach_send_tiers():
    assert assign_tier(110.0, True, [], aum_b=0.85) == "C"


def test_variant_assignment_is_stable_and_balanced():
    slugs = [f"firm-{i}" for i in range(400)]
    first = [assign_variant(s, "exp") for s in slugs]
    assert first == [assign_variant(s, "exp") for s in slugs]      # deterministic
    share = first.count("A") / len(first)
    assert 0.40 < share < 0.60                                     # balanced


def test_opener_validator_catches_unsendable_copy():
    assert "banned_phrase:reaching out" in validate("I am reaching out today.")
    assert "unfilled_placeholder" in validate("Hi [FIRM], good to connect.")
    assert validate("") == ["empty"]


def test_end_to_end_on_sample(tmp_path):
    df = pd.read_csv("data/raw/sample-leads.csv", dtype=str, keep_default_na=False)
    norm = normalize_frame(df)
    resolved, audit = resolve_entities(norm)
    assert len(resolved) < len(norm)              # duplicates actually collapsed
    assert len(audit) >= 4
    thornbury = resolved[resolved.firm_name.str.contains("Thornbury")].iloc[0]
    assert thornbury["persona"] == "RESEARCH_LEAD"   # elected over the EA
    assert "Oyelaran" in thornbury["alternate_contacts"]
