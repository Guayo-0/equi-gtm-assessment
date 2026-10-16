"""
Quality check de la corrida. Lee lo que hay en out/ y verifica que sea
internamente coherente. No compara contra números memorizados: comprueba que
el sistema no se contradiga a sí mismo, que es lo que de verdad importa.

    python qc.py
    python qc.py --out out
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.personalize import assign_variant  # noqa: E402

OK, FAIL, WARN = "PASS", "FALLA", "AVISO"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "", warn_only: bool = False):
    status = OK if condition else (WARN if warn_only else FAIL)
    results.append((status, name, detail))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    o = args.out

    # --- 0. ¿Existen los cinco archivos? ------------------------------------
    expected_files = ["run_report.json", "scored_leads.csv", "shortlist.csv",
                      "merge_audit.csv", "enrichment_queue.csv"]
    missing = [f for f in expected_files if not os.path.exists(os.path.join(o, f))]
    check("Los 5 archivos de salida existen", not missing,
          f"faltan: {missing}" if missing else "")
    if missing:
        report()
        return 1

    rep = json.load(open(os.path.join(o, "run_report.json"), encoding="utf-8"))
    scored = pd.read_csv(os.path.join(o, "scored_leads.csv"))
    short = pd.read_csv(os.path.join(o, "shortlist.csv"))
    audit = pd.read_csv(os.path.join(o, "merge_audit.csv"))
    enrich = pd.read_csv(os.path.join(o, "enrichment_queue.csv"))

    live = rep.get("mode") != "offline-template"
    print(f"\nModo de la corrida: {rep.get('mode')}"
          f"  ({'LLM real' if live else 'plantillas offline'})\n")

    # --- 1. Integridad del pipeline -----------------------------------------
    check("Entraron 50 filas", rep.get("input_rows") == 50,
          f"input_rows = {rep.get('input_rows')}")
    check("La deduplicación colapsó filas",
          rep.get("rows_after_entity_resolution", 0) < rep.get("input_rows", 0),
          f"{rep.get('input_rows')} filas -> {rep.get('rows_after_entity_resolution')} firmas, "
          f"{rep.get('duplicate_clusters_collapsed')} grupos fusionados")
    check("scored_leads tiene una fila por firma resuelta",
          len(scored) == rep.get("rows_after_entity_resolution"),
          f"{len(scored)} filas")
    check("No quedaron firmas duplicadas",
          scored["firm_slug"].duplicated().sum() == 0,
          f"duplicados: {scored['firm_slug'].duplicated().sum()}")

    # --- 2. Generación de openers (el fallo silencioso de la vez pasada) -----
    att, gen, err = (rep.get("openers_attempted", 0), rep.get("openers_generated", 0),
                     rep.get("opener_errors", 0))
    check("Todos los openers intentados se generaron", att == gen and err == 0,
          f"intentados={att} generados={gen} errores={err}")
    vacios = short["opener"].isna().sum() + (short["opener"].fillna("") == "").sum()
    check("Ninguna fila del shortlist tiene el opener vacío", vacios == 0,
          f"{vacios} filas sin opener")
    # A fully-cached run makes zero API calls, so zero cost is correct. The
    # check that matters is: if we PAID for calls, the cost must be recorded.
    calls = rep.get("llm_calls", 0)
    hits = rep.get("cache_hits", 0)
    cost = rep.get("llm_cost_usd", 0)
    if not live:
        detalle = f"llm_cost_usd = {cost}  (0 es correcto en modo offline)"
        ok = True
    elif calls == 0:
        detalle = f"llm_cost_usd = {cost}  ({hits} del caché, 0 llamadas — 0 es correcto)"
        ok = True
    else:
        detalle = (f"llm_cost_usd = {cost} para {calls} llamadas"
                   + ("" if cost > 0 else "  <- se pagaron llamadas y no se registró costo"))
        ok = cost > 0
    check("El costo quedó registrado", ok, detalle)

    # --- 3. Aritmética del puntaje ------------------------------------------
    def suma_rationale(txt: str) -> float:
        return sum(float(x) for x in re.findall(r"([+-]\d+(?:\.\d+)?):", str(txt)))

    elig = scored[scored["eligible"] == True]  # noqa: E712
    desc = (elig.assign(suma=elig["score_rationale"].map(suma_rationale))
                .pipe(lambda d: d[(d["suma"] - d["icp_score"]).abs() > 0.05]))
    check("La suma del score_rationale cuadra con icp_score en todas las filas",
          len(desc) == 0,
          "" if len(desc) == 0 else f"no cuadran: {list(desc['firm_name'])}")

    # --- 4. La portería ------------------------------------------------------
    dq = scored[scored["eligible"] == False]  # noqa: E712
    check("Toda descalificada tiene puntaje 0 y motivo escrito",
          (dq["icp_score"] == 0).all() and dq["disqualify_reason"].notna().all(),
          f"{len(dq)} descalificadas")
    check("Ninguna descalificada quedó en el shortlist",
          not short["firm_name"].isin(dq["firm_name"]).any())

    # --- 5. Topes de política ------------------------------------------------
    sub = scored[(scored["aum_b"].notna()) & (scored["aum_b"] < 1.0)
                 & (scored["eligible"] == True)]  # noqa: E712
    check("Ninguna firma bajo $1B llegó a un tier de contacto 1:1",
          not sub["tier"].isin(["A", "B"]).any(),
          f"{len(sub)} firmas sub-banda, tiers: {sorted(sub['tier'].unique())}")
    sin_aum = scored[scored["aum_b"].isna() & (scored["eligible"] == True)]  # noqa: E712
    check("Ninguna firma sin AUM verificado llegó a Tier A",
          not (sin_aum["tier"] == "A").any(),
          f"{len(sin_aum)} firmas sin AUM")

    # --- 6. Ruteo ------------------------------------------------------------
    bloqueadas = scored[scored["route"] == "ENRICH"]
    check("Toda firma en la cola de enriquecimiento tiene una señal que lo explica",
          bloqueadas["signals"].fillna("").str.contains("NEEDS_").all(),
          f"{len(bloqueadas)} en cola: {list(bloqueadas['firm_name'])}")
    check("enrichment_queue.csv coincide con el ruteo",
          len(enrich) == len(bloqueadas))
    check("Nadie con ruta SUPPRESS recibió un opener",
          (scored[scored["route"] == "SUPPRESS"]["opener"].fillna("") == "").all())

    # --- 7. A/B determinista -------------------------------------------------
    exp = rep.get("experiment_id")
    con_variante = scored[scored["ab_variant"].notna() & (scored["ab_variant"] != "")]
    malas = [r["firm_name"] for _, r in con_variante.iterrows()
             if assign_variant(r["firm_slug"], exp) != r["ab_variant"]]
    check("La variante A/B de cada firma es reproducible", not malas,
          f"{len(con_variante)} asignadas, reparto {rep.get('ab_split')}"
          if not malas else f"no reproducibles: {malas}")

    # --- 8. Elección de contacto en la deduplicación -------------------------
    th = audit[audit["surviving_firm_name"].str.contains("Thornbury", na=False)]
    if len(th):
        r = th.iloc[0]
        check("En Thornbury el contacto elegido no es el asistente",
              "EA to" not in str(r["elected_contact"])
              and "Oyelaran" in str(r["suppressed_contacts"]),
              f"elegido: {r['elected_contact']} | guardado: {r['suppressed_contacts']}")
    else:
        check("Thornbury aparece en el merge_audit", False, warn_only=True)

    return report()


def report() -> int:
    ancho = max(len(n) for _, n, _ in results) + 2
    fallas = 0
    for status, name, detail in results:
        marca = {OK: "  OK  ", FAIL: " FALLA", WARN: " AVISO"}[status]
        print(f"[{marca}] {name.ljust(ancho)} {detail}")
        fallas += status == FAIL
    print()
    if fallas:
        print(f"{fallas} verificación(es) fallaron. No grabes el video todavía.\n")
        return 1
    print("Todo en orden. La corrida es coherente consigo misma.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
