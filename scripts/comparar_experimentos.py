"""Compara un experimento contra el modelo completo, con la misma configuracion de post-proceso.

    python scripts/comparar_experimentos.py --nuevo mejorado_384

Validacion (decide): completo se toma de su barrido_tamano en 1 mL, que es el post-proceso
oficial; el experimento nuevo de su diagnostico_val (ya corre con min_volumen_ml: 1).
Test (solo se reporta): evaluacion_cuda.json de cada uno, nivel de degradacion 0.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def val_completo(res: Path) -> dict:
    with open(res / "completo" / "diagnostico_val" / "barrido_tamano.csv", encoding="utf-8") as f:
        fila = next(r for r in csv.DictReader(f) if abs(float(r["min_volumen_ml"]) - 1.0) < 1e-6)
    return {k: float(fila[k]) for k in ("dice_fragmento", "dice_principal", "dice_secundario",
                                        "secundarios_detectados", "precision_fragmentos")}


def val_nuevo(res: Path, nombre: str) -> dict:
    with open(res / nombre / "diagnostico_val" / "resumen.json", encoding="utf-8") as f:
        a = json.load(f)["actual"]
    return {k: float(a[k]) for k in ("dice_fragmento", "dice_principal", "dice_secundario",
                                     "secundarios_detectados", "precision_fragmentos")}


def test(res: Path, nombre: str) -> dict | None:
    p = res / nombre / "evaluacion_cuda.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        n0 = json.load(f)["por_nivel_degradacion"]["0"]
    s, d, c, dist = n0["segmentacion"], n0["deteccion"], n0["clasificacion"], n0["distancia"]
    return {"dice_fragmento": s["dice_fragmento"], "iou_fragmento": s["iou_fragmento"],
            "precision_fragmentos": s["precision_fragmentos"], "recall_fragmentos": s["recall_fragmentos"],
            "hd95_mm": s["hd95_mm"], "map50": d["mAP@0.50"], "f1_cls": c["f1_macro"],
            "mae_dist_geom_mm": dist["mae_geometrica_mm"]}


def tabla(titulo, a, b, nombre):
    print(f"\n{titulo}")
    print(f"{'metrica':26s} {'completo':>10s} {nombre:>14s} {'cambio':>9s}")
    for k in a:
        va, vb = a[k], b.get(k)
        if va is None or vb is None:
            continue
        print(f"{k:26s} {va:10.3f} {vb:14.3f} {vb - va:+9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuevo", default="mejorado_384")
    ap.add_argument("--resultados", default="resultados")
    args = ap.parse_args()
    res = Path(args.resultados)

    va, vb = val_completo(res), val_nuevo(res, args.nuevo)
    tabla("VALIDACION (aqui se decide)", va, vb, args.nuevo)
    ta, tb = test(res, "completo"), test(res, args.nuevo)
    if ta and tb:
        tabla("TEST (solo se reporta)", ta, tb, args.nuevo)

    delta = vb["dice_fragmento"] - va["dice_fragmento"]
    print("\nDecision:", "ADOPTAR el nuevo modelo" if delta > 0.01 else
          "mejora dentro del ruido de una semilla: no se adopta, se reporta" if delta > 0 else
          "NO mejora en validacion: se queda completo y el experimento se reporta",
          f"(Dice val {delta:+.3f})")


if __name__ == "__main__":
    main()
