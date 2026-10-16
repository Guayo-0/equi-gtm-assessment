"""
Entity resolution. The source list has three dedupe patterns:

  1. Same firm, cosmetic name drift   -> 'Haverlock Advisory Group' / '... Grp.'
  2. Same firm, different AUM format  -> '11500' / '11.5B'
  3. Same firm, two humans            -> Thornbury (Director of Research + EA)

(3) is the interesting one: it is not a duplicate record, it is a contact
choice. We collapse to one firm row and elect the contact with the highest
decision velocity, keeping the loser as an alternate rather than discarding it.

Resolution is union-find over two blocking keys (name slug, email domain) so
transitive matches merge in one pass.
"""

from __future__ import annotations

import pandas as pd

from .normalize import firm_slug


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _completeness(row: pd.Series) -> int:
    """How much signal a record carries. Used to pick the survivor."""
    fields = ["aum_b", "email", "phone", "avg_client_size_m", "alts_raw",
              "primary_contact", "custodian", "last_touch", "country"]
    return sum(1 for f in fields if _present(row.get(f)))


def _present(v: object) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return str(v).strip() != ""


def _coalesce(rows: list[pd.Series], field: str):
    """First non-empty value, scanning records in survivor-priority order."""
    for r in rows:
        if _present(r.get(field)):
            return r[field]
    return rows[0].get(field)


def resolve_entities(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (resolved_firms, merge_audit).

    The audit frame is a deliverable, not debug output: a GTM team needs to see
    which records were collapsed before it trusts a shortlist.
    """
    uf = _UnionFind()

    # Blocking key 1: canonical name slug.
    for slug in df["firm_slug"]:
        uf.find(slug)

    # Blocking key 2: shared email domain implies same firm.
    domain_to_slug: dict[str, str] = {}
    for _, r in df.iterrows():
        dom = r["email_domain"]
        if not dom:
            continue
        if dom in domain_to_slug:
            uf.union(domain_to_slug[dom], r["firm_slug"])
        else:
            domain_to_slug[dom] = r["firm_slug"]

    # Blocking key 3: near-identical slug (token-subset), catches
    # 'ridgeline-family-office' vs 'ridgeline-family-offices'.
    slugs = sorted(set(df["firm_slug"]))
    for i, a in enumerate(slugs):
        for b in slugs[i + 1:]:
            if _slug_near_match(a, b):
                uf.union(a, b)

    df = df.copy()
    df["_cluster"] = df["firm_slug"].map(uf.find)
    df["_completeness"] = df.apply(_completeness, axis=1)

    resolved, audit = [], []
    for cluster, group in df.groupby("_cluster", sort=False):
        # Survivor priority: most complete record, tie-broken by contact rank.
        ordered = group.sort_values(
            ["_completeness", "persona_points"], ascending=False
        )
        rows = [r for _, r in ordered.iterrows()]
        primary = rows[0]

        merged = {c: _coalesce(rows, c) for c in group.columns
                  if not c.startswith("_")}

        # Contact election is independent of record completeness: the best
        # person to email is the highest decision velocity, not the row with
        # the most filled cells.
        contacts = [r for r in rows if _present(r.get("primary_contact"))]
        if contacts:
            best = max(contacts, key=lambda r: (r["persona_points"],
                                                _present(r.get("email"))))
            for f in ("primary_contact", "title", "persona", "persona_points",
                      "email", "email_domain"):
                merged[f] = best[f]
            alts = [f"{r['primary_contact']} ({r['title']})" for r in contacts
                    if r["primary_contact"] != best["primary_contact"]]
            merged["alternate_contacts"] = "; ".join(sorted(set(alts)))
        else:
            merged["alternate_contacts"] = ""

        merged["firm_slug"] = firm_slug(merged["firm_name"])
        merged["source_record_count"] = len(rows)
        merged["_parse_flags"] = "|".join(
            sorted({f for r in rows for f in str(r["_parse_flags"]).split("|") if f})
        )
        resolved.append(merged)

        if len(rows) > 1:
            audit.append({
                "cluster_key": cluster,
                "surviving_firm_name": merged["firm_name"],
                "merged_record_count": len(rows),
                "merged_names": " | ".join(r["firm_name"] for r in rows),
                "elected_contact": merged["primary_contact"],
                "suppressed_contacts": merged["alternate_contacts"],
                "aum_values_seen": " | ".join(
                    str(r["aum_raw"]) for r in rows if _present(r["aum_raw"])),
            })

    out = pd.DataFrame(resolved).reset_index(drop=True)
    return out, pd.DataFrame(audit)


def _slug_near_match(a: str, b: str) -> bool:
    """Token-set containment with a single-token tolerance, plus plural drift."""
    ta, tb = set(a.split("-")), set(b.split("-"))
    if not ta or not tb:
        return False
    norm = lambda s: {t.rstrip("s") for t in s}
    na, nb = norm(ta), norm(tb)
    if na == nb:
        return True
    smaller, larger = (na, nb) if len(na) <= len(nb) else (nb, na)
    return len(smaller) >= 2 and smaller.issubset(larger) and len(larger - smaller) <= 1
