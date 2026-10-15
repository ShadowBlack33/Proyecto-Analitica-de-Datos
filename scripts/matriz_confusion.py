"""Matriz de confusion de la clasificacion por region (SA / LI / RI) sobre el split de test.

Usa la cabeza de clasificacion: por corte, sigmoid(cls[:3]) >= umbral -> "region presente".
La verdad es batch["cls"][:, :3]. Se calcula una matriz 2x2 por region (filas = verdad,
columnas = prediccion) con las mismas funciones numpy que usa evaluar.py, de modo que el
F1 promedio debe coincidir con el de la evaluacion (~0.987).

    python scripts/matriz_confusion.py --ckpt pesos/completo.pth
    python scripts/matriz_confusion.py --ckpt pesos/completo.pth --device cpu

Tambien dibuja la matriz 4x4 por pixel (fondo/SA/LI/RI) que evaluar.py ya guardo en su JSON;
esa no necesita datos ni modelo:

    python scripts/matriz_confusion.py --solo_pixel
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.eval.metricas_seg_cls import f1_por_clase, matriz_confusion  # noqa: E402

REGIONES = ["SA", "LI", "RI"]
NOMBRES = {"SA": "Sacro", "LI": "Coxal izquierdo", "RI": "Coxal derecho"}
CLASES_PIXEL = ["Fondo", "SA", "LI", "RI"]
EVAL_REFERENCIA = RAIZ / "docs" / "resultados" / "completo" / "evaluacion_cuda.json"


# ---------------------------------------------------------------- calculo (numpy puro)
def matrices_por_region(verdad: np.ndarray, prediccion: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """verdad, prediccion (N, 3) booleanas -> 3 matrices 2x2 y el F1 de cada region."""
    verdad = np.asarray(verdad).astype(bool)
    prediccion = np.asarray(prediccion).astype(bool)
    mats = [matriz_confusion(verdad[:, r], prediccion[:, r], 2) for r in range(3)]
    return mats, f1_por_clase(verdad, prediccion)


# ---------------------------------------------------------------- figuras
def _celda(ax, m: np.ndarray, etiquetas: list[str], titulo: str, mostrar_y: bool = True):
    """Una matriz: color = fraccion de la fila (verdad); texto = conteo y porcentaje."""
    fila = m.sum(1, keepdims=True).astype(float)
    frac = np.divide(m, fila, out=np.zeros_like(m, dtype=float), where=fila > 0)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    n = len(etiquetas)
    for i in range(n):
        for j in range(n):
            color = "white" if frac[i, j] > 0.55 else "#1b1b1b"
            ax.text(j, i, f"{m[i, j]:,}\n{100 * frac[i, j]:.1f}%", ha="center", va="center",
                    color=color, fontsize=9 if n == 2 else 8)
    ax.set_xticks(range(n), etiquetas)
    ax.set_yticks(range(n), etiquetas if mostrar_y else [""] * n)
    ax.set_xlabel("Predicción")
    if mostrar_y:
        ax.set_ylabel("Verdad (ground truth)")
    ax.set_title(titulo, fontsize=11)


def graficar_regiones(mats, f1, salida: Path, n_cortes: int, umbral: float, nombre_modelo: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.4))
    for r, ax in enumerate(axs):
        _celda(ax, mats[r], ["Ausente", "Presente"],
               f"{NOMBRES[REGIONES[r]]} ({REGIONES[r]}) · F1 = {f1[r]:.3f}", mostrar_y=(r == 0))
    fig.suptitle(f"Clasificación por región · {nombre_modelo} · test ({n_cortes:,} cortes) · umbral {umbral}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.text(0.5, 0.012, "Color = fracción de cada fila (verdad); número = cortes.", ha="center",
             fontsize=8, color="#555555")
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=140)
    plt.close(fig)


def graficar_pixel(m4: np.ndarray, salida: Path, nombre_modelo: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    _celda(ax, np.asarray(m4, dtype=np.int64), CLASES_PIXEL, "")
    ax.set_title(f"Segmentación por píxel (4 clases) · {nombre_modelo} · test", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.5, 0.012, "Píxeles submuestreados ×4. Color = fracción de cada fila (verdad).",
             ha="center", fontsize=8, color="#555555")
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- inferencia (necesita torch + datos)
def predecir_split(cfg: dict, ckpt: str, device: str | None, split: str, batch: int):
    """Recorre los cortes del split y devuelve (verdad (N,3) bool, probabilidad (N,3))."""
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import PengwinCortes
    from src.data.splits import cargar_splits
    from src.inferencia import cargar_modelo
    from src.utils import dispositivo

    dev = dispositivo(device)
    ids = cargar_splits(cfg["rutas"]["splits"])[split]
    ds = PengwinCortes(cfg["rutas"]["cache"], ids, cfg, entrenamiento=False)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=cfg["datos"].get("num_workers", 0))
    modelo = cargar_modelo(ckpt, cfg, dev)
    print(f"{len(ids)} casos · {len(ds):,} cortes · dispositivo {dev}")

    verdad, prob = [], []
    with torch.no_grad():
        for i, b in enumerate(dl):
            x = b["img"].to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=cfg["entrenamiento"]["amp"] and dev.type == "cuda"):
                out = modelo(x, cabezas=("cls",))
            prob.append(torch.sigmoid(out["cls"][:, :3].float()).cpu().numpy())
            verdad.append(b["cls"][:, :3].numpy() >= 0.5)
            if (i + 1) % 50 == 0:
                print(f"  lote {i + 1}/{len(dl)}")
    return np.concatenate(verdad), np.concatenate(prob)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="pesos/completo.pth")
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--umbral", type=float, default=0.5)
    ap.add_argument("--nombre", default="completo", help="nombre del modelo en el titulo")
    ap.add_argument("--salida", default="docs/figuras/matriz_confusion.png")
    ap.add_argument("--evaluacion", default=str(EVAL_REFERENCIA),
                    help="JSON de evaluar.py: da la matriz 4x4 por pixel y el F1 de referencia")
    ap.add_argument("--solo_pixel", action="store_true", help="solo la matriz 4x4 (no usa datos ni modelo)")
    args = ap.parse_args()

    salida = Path(args.salida)
    ev = None
    if Path(args.evaluacion).exists():
        ev = json.load(open(args.evaluacion, encoding="utf-8"))["por_nivel_degradacion"]["0"]

    if ev is not None and "matriz_confusion_pixel" in ev:
        ruta = salida.with_name("matriz_confusion_pixel.png")
        graficar_pixel(np.array(ev["matriz_confusion_pixel"]), ruta, args.nombre)
        print("guardado:", ruta)
    else:
        print("[aviso] sin matriz por pixel: no se encontro", args.evaluacion)
    if args.solo_pixel:
        return

    from src.utils import cargar_config
    cfg = cargar_config(args.config)
    verdad, prob = predecir_split(cfg, args.ckpt, args.device, args.split, args.batch)
    mats, f1 = matrices_por_region(verdad, prob >= args.umbral)

    for r, m in enumerate(mats):
        (tn, fp), (fn, tp) = m
        print(f"{REGIONES[r]}: TN {tn:,}  FP {fp:,}  FN {fn:,}  TP {tp:,}  F1 {f1[r]:.4f}")
    f1_prom = float(np.mean(f1))
    print(f"F1 promedio: {f1_prom:.4f}")
    if ev is not None:
        ref = ev["clasificacion"]["f1_macro"]
        estado = "OK" if abs(f1_prom - ref) < 0.005 else "REVISAR: no coincide con la evaluacion"
        print(f"  referencia de evaluar.py: {ref:.4f}  ->  {estado}")

    graficar_regiones(mats, f1, salida, len(verdad), args.umbral, args.nombre)
    with open(salida.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"modelo": args.nombre, "split": args.split, "umbral": args.umbral, "cortes": int(len(verdad)),
                   "f1_promedio": f1_prom,
                   "regiones": {REGIONES[r]: {"matriz_2x2": mats[r].tolist(), "f1": float(f1[r])} for r in range(3)}},
                  f, indent=1)
    print("guardado:", salida)


if __name__ == "__main__":
    main()
