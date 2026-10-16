"""
Field-level normalization. Every function is pure and total: dirty input in,
typed value or None out, never an exception. Parsing failures are recorded on
the row (`_parse_flags`) rather than dropped, because a silent coercion in a
GTM pipeline becomes a mis-scored lead nobody audits.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime

import pandas as pd

from . import config as C

# --- AUM ---------------------------------------------------------------------

_MULTIPLIERS = {
    "t": 1e12, "tn": 1e12, "trillion": 1e12,
    "b": 1e9, "bn": 1e9, "bil": 1e9, "billion": 1e9, "billions": 1e9,
    "m": 1e6, "mm": 1e6, "mil": 1e6, "million": 1e6, "millions": 1e6,
    "k": 1e3, "thousand": 1e3,
}

_NUM_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)\s*(?P<suffix>[a-z]*)$")

_NULLISH = {"", "n/a", "na", "none", "null", "unknown", "undisclosed", "-", "tbd"}


def parse_aum_billions(raw: object) -> tuple[float | None, str]:
    """
    Normalize heterogeneous AUM strings to float billions.

    Handles: '4.2B', '$4,200,000,000', '11500', '2.5 billion', '1.4b',
             '6800000000', '880M', '' -> None

    Bare numbers are ambiguous. Resolution is by order of magnitude, which is
    safe inside this ICP (no real wealth manager is worth $11,500):
        >= 1e8        -> raw USD        (6800000000 -> 6.8B)
        1e3 .. 1e8    -> USD millions   (11500      -> 11.5B)
        < 1e3         -> USD billions   (4.2        -> 4.2B)

    Returns (billions, flag). flag is '' on a clean parse, otherwise a short
    machine-readable reason.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, "aum_missing"

    s = str(raw).strip().lower()
    if s in _NULLISH:
        return None, "aum_missing"

    s = s.replace("$", "").replace(",", "").replace("_", "").strip()
    s = re.sub(r"\s*(usd|aum)\s*$", "", s)
    s = re.sub(r"[~+<>≈]", "", s).strip()

    m = _NUM_RE.match(s)
    if not m:
        return None, f"aum_unparsed:{str(raw)[:24]}"

    num = float(m.group("num"))
    suffix = m.group("suffix")

    if suffix:
        mult = _MULTIPLIERS.get(suffix)
        if mult is None:
            return None, f"aum_unknown_suffix:{suffix}"
        usd = num * mult
        return round(usd / 1e9, 4), ""

    # Bare number: infer scale.
    if num >= 1e8:
        usd, flag = num, ""
    elif num >= 1e3:
        usd, flag = num * 1e6, "aum_scale_inferred:millions"
    else:
        usd, flag = num * 1e9, "aum_scale_inferred:billions"
    return round(usd / 1e9, 4), flag


def parse_client_size_millions(raw: object) -> float | None:
    """avg_client_size: '6.5M', '900K', '14M' -> float millions."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip().lower().replace("$", "").replace(",", "")
    if s in _NULLISH:
        return None
    m = _NUM_RE.match(s)
    if not m:
        return None
    num = float(m.group("num"))
    suffix = m.group("suffix") or "m"      # bare number defaults to millions
    mult = _MULTIPLIERS.get(suffix)
    if mult is None:
        return None
    return round(num * mult / 1e6, 4)


# --- Firm type ---------------------------------------------------------------

def normalize_firm_type(raw: object, firm_name: str = "", notes: str = "") -> str:
    """Map dirty type strings to the canonical taxonomy in config."""
    s = _clean_text(raw).lower()
    if s in C.FIRM_TYPE_SYNONYMS:
        canon = C.FIRM_TYPE_SYNONYMS[s]
    else:
        canon = "UNKNOWN"
        for token, mapped in C.FIRM_TYPE_SYNONYMS.items():
            if token and token in s:
                canon = mapped
                break

    # "family office" is ambiguous; disambiguate with name/notes.
    if canon == "MFO":
        blob = f"{firm_name} {notes}".lower()
        if "single family" in blob or re.search(r"\bsfo\b", blob):
            canon = "SFO"
    return canon


# --- Text / identity ---------------------------------------------------------

_LEGAL_SUFFIXES = (
    r"\bllc\b", r"\bl\.l\.c\.?", r"\binc\b", r"\bincorporated\b", r"\bltd\b",
    r"\blp\b", r"\bllp\b", r"\bplc\b", r"\bco\b", r"\bcorp\b", r"\bcorporation\b",
)

_NAME_EQUIVALENTS = {
    "advisers": "advisors", "adviser": "advisor", "advisory": "advisor",
    "grp": "group", "mgmt": "management", "cap": "capital",
    "ptrs": "partners", "svcs": "services", "intl": "international",
    "and": "&",
}


def _clean_text(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    return re.sub(r"\s+", " ", str(raw)).strip()


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def firm_slug(name: object) -> str:
    """
    Canonical identity key for a firm. Must collapse:
        'Harborstone Wealth Partners' / 'HARBORSTONE WEALTH PARTNERS LLC'
        'Bramblewick Capital Advisers' / 'Bramblewick Capital Advisors Inc'
        'Haverlock Advisory Group' / 'Haverlock Advisory Grp.'
    """
    s = strip_accents(_clean_text(name)).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    for pat in _LEGAL_SUFFIXES:
        s = re.sub(pat, " ", s)
    tokens = [_NAME_EQUIVALENTS.get(t, t) for t in s.split()]
    # Drop low-information tail words that vary between records.
    tokens = [t for t in tokens if t not in {"the", "and"}]
    return "-".join(tokens).strip("-")


def normalize_email(raw: object) -> str:
    s = _clean_text(raw).lower().replace(" ", "")   # 'ostrandervoss .com' -> fix
    if "@" not in s:
        return ""
    local, _, domain = s.partition("@")
    if "." not in domain:
        return ""
    return f"{local}@{domain}"


def email_domain(email: str) -> str:
    return email.partition("@")[2] if "@" in email else ""


def is_generic_inbox(email: str) -> bool:
    local = email.partition("@")[0]
    return local in C.GENERIC_INBOX_PREFIXES


def normalize_phone(raw: object) -> str:
    digits = re.sub(r"\D", "", _clean_text(raw))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return ""
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def normalize_country(raw: object) -> str:
    s = _clean_text(raw).upper()
    return {"US": "USA", "UNITED STATES": "USA", "": "USA"}.get(s, s)


_STATE_MAP = {
    "north carolina": "NC", "georgia": "GA", "california": "CA",
    "new york": "NY", "massachusetts": "MA", "texas": "TX", "florida": "FL",
    "colorado": "CO", "illinois": "IL", "washington": "WA", "ontario": "ON",
}


def normalize_state(raw: object) -> str:
    s = _clean_text(raw)
    return _STATE_MAP.get(s.lower(), s.upper())


_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y",
                 "%b %Y", "%B %Y", "%B %d %Y", "%b %d %Y", "%B %d, %Y")


def parse_last_touch(raw: object) -> date | None:
    """'2026-03-14', '03/14/2026', '4/11/26', 'Jan 2026', 'April 30 2026'."""
    s = _clean_text(raw)
    if not s or s.lower() in _NULLISH:
        return None
    s_clean = s.replace(",", "")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s_clean, fmt).date()
        except ValueError:
            continue
    return None


# --- Alts posture ------------------------------------------------------------

def classify_alts(raw: object) -> tuple[str, int]:
    """Return (posture label, distinct sleeve count)."""
    s = _clean_text(raw).lower()
    # Order matters: "none" is an assertion that the firm runs no alts, which
    # is a different fact from a blank cell. Checking _NULLISH first would
    # collapse the two and over-score every zero-alts firm.
    if s in {"none", "no", "0", "zero"}:
        return C.ALTS_NONE, 0
    if not s or s in _NULLISH:
        return C.ALTS_UNKNOWN, 0
    if "everything" in s:
        return C.ALTS_HEAVY, 5

    s = (s.replace(" / ", ",").replace("/", ",")
           .replace(" and ", ",").replace(";", ","))
    expand = {"pe": "private equity", "re": "real estate",
              "vc": "venture", "hf": "hedge funds", "infra": "infrastructure"}
    sleeves = {expand.get(p.strip(), p.strip()) for p in s.split(",") if p.strip()}
    n = len(sleeves)
    if n == 0:
        return C.ALTS_UNKNOWN, 0
    if n == 1:
        return C.ALTS_LIGHT, n
    if n == 2:
        return C.ALTS_MODERATE, n
    return C.ALTS_HEAVY, n


# --- Persona -----------------------------------------------------------------

def classify_persona(title: object) -> tuple[str, int]:
    """Return (persona, decision-velocity points 0-10). Longest match wins."""
    s = _clean_text(title).lower().replace("&", "and").replace(".", "")
    if not s:
        return "UNKNOWN", 0
    best: tuple[str, int] | None = None
    best_len = -1
    for token, persona, pts in C.PERSONA_RANK:
        t = token.replace(".", "")
        if t in s and len(t) > best_len:
            best, best_len = (persona, pts), len(t)
    return best if best else ("OTHER", 2)


def normalize_custodian(raw: object) -> str:
    s = _clean_text(raw).upper()
    return {"CHARLES SCHWAB": "SCHWAB", "BNY PERSHING": "PERSHING"}.get(s, s)


# --- Row-level orchestration -------------------------------------------------

NORMALIZED_COLUMNS = [
    "firm_slug", "firm_name", "firm_type", "firm_type_raw", "aum_b", "aum_raw",
    "city", "state", "country", "primary_contact", "title", "persona",
    "persona_points", "email", "email_domain", "phone", "custodian",
    "alts_posture", "alts_sleeves", "alts_raw", "avg_client_size_m",
    "last_touch", "notes", "_parse_flags",
]


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized-enough row normalization (n=50; clarity beats micro-perf)."""
    rows = []
    for _, r in df.iterrows():
        flags: list[str] = []

        firm_name = _clean_text(r.get("firm_name"))
        notes = _clean_text(r.get("notes"))

        aum_b, aum_flag = parse_aum_billions(r.get("aum"))
        if aum_flag:
            flags.append(aum_flag)

        email = normalize_email(r.get("email"))
        if _clean_text(r.get("email")) and not email:
            flags.append("email_invalid")
        if email and is_generic_inbox(email):
            flags.append("email_generic_inbox")

        posture, sleeves = classify_alts(r.get("alts_exposure"))
        persona, persona_pts = classify_persona(r.get("title"))
        last_touch = parse_last_touch(r.get("last_touch"))
        if _clean_text(r.get("last_touch")) and last_touch is None:
            flags.append("last_touch_unparsed")

        rows.append({
            "firm_slug": firm_slug(firm_name),
            "firm_name": firm_name,
            "firm_type": normalize_firm_type(r.get("firm_type"), firm_name, notes),
            "firm_type_raw": _clean_text(r.get("firm_type")),
            "aum_b": aum_b,
            "aum_raw": _clean_text(r.get("aum")),
            "city": _clean_text(r.get("city")).title(),
            "state": normalize_state(r.get("state")),
            "country": normalize_country(r.get("country")),
            "primary_contact": _clean_text(r.get("primary_contact")),
            "title": _clean_text(r.get("title")),
            "persona": persona,
            "persona_points": persona_pts,
            "email": email,
            "email_domain": email_domain(email),
            "phone": normalize_phone(r.get("phone")),
            "custodian": normalize_custodian(r.get("custodian")),
            "alts_posture": posture,
            "alts_sleeves": sleeves,
            "alts_raw": _clean_text(r.get("alts_exposure")),
            "avg_client_size_m": parse_client_size_millions(r.get("avg_client_size")),
            "last_touch": last_touch,
            "notes": notes,
            "_parse_flags": "|".join(flags),
        })

    out = pd.DataFrame(rows, columns=NORMALIZED_COLUMNS)
    return out[out["firm_name"].str.len() > 0].reset_index(drop=True)
