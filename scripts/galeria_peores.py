"""Galeria de los peores casos del test: CT | ground truth | prediccion, en el corte con mas error.

Por cada caso procesa UN volumen a la vez (pesan cientos de MB): lee imagen y etiqueta, infiere,
elige el corte con mas voxeles donde la prediccion difiere del ground truth y guarda
docs/figuras/peor_<caso>.png.

Los casos salen de `peores_5` en el JSON de evaluar.py (los 5 de menor Dice de fragmento), asi que
siempre corresponden a la evaluacion vigente. Para forzar otros: --casos 085 084 002

    python scripts/galeria_peores.py --ckpt pesos/completo.pth
    python scripts/galeria_peores.py --ckpt pesos/completo.pth --casos 039

El "error" del corte NO compara los ids crudos: el modelo puede llamar SA-4 a lo que el GT llama
SA-2. Primero se empareja cada fragmento predicho con el del GT de mayor IoU 3D (asignacion
optima, misma que usa evaluar.py) y se le pone el id del GT. Asi, el mismo fragmento tiene el
mismo color en los dos paneles. Un fragmento predicho sin pareja se dibuja en ROJO.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.data.etiquetas import nombre_fragmento, region_de_etiqueta  # noqa: E402
from src.eval.asignacion import asignacion_optima  # noqa: E402

EVAL_REFERENCIA = RAIZ / "docs" / "resultados" / "completo" / "evaluacion_cuda.json"
ROJO = (230, 40, 40)
K = 31  # etiquetas 0..30


# ---------------------------------------------------------------- calculo (numpy puro)
def alinear_ids(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Empareja los fragmentos predichos con los del GT, region por region (IoU 3D, asignacion optima).

    Devuelve (pred_alineada, espurio, pares):
      pred_alineada  la prediccion con los ids del GT en los fragmentos emparejados; los
                     fragmentos sin pareja quedan en 0
      espurio        mascara bool de los voxeles de fragmentos predichos sin pareja
      pares          {id_predicho: id_gt} (los emparejados)
    """
    pred_al = np.zeros_like(pred)
    espurio = np.zeros(pred.shape, bool)
    pares: dict[int, int] = {}
    reg_gt, reg_pr = region_de_etiqueta(gt), region_de_etiqueta(pred)
    for r in (1, 2, 3):
        g = np.where(reg_gt == r, gt, 0).astype(np.int64)
        p = np.where(reg_pr == r, pred, 0).astype(np.int64)
        union_mask = (g > 0) | (p > 0)
        if not union_mask.any():
            continue
        ids_g = [i for i in np.unique(g) if i > 0]
        ids_p = [i for i in np.unique(p) if i > 0]
        conjunta = np.bincount((g[union_mask] * K + p[union_mask]), minlength=K * K).reshape(K, K)
        area_g, area_p = conjunta.sum(1), conjunta.sum(0)
        iou = np.zeros((len(ids_g), len(ids_p)))
        for a, ig in enumerate(ids_g):
            for b, ip in enumerate(ids_p):
                inter = conjunta[ig, ip]
                iou[a, b] = inter / max(area_g[ig] + area_p[ip] - inter, 1)
        emparejados = set()
        if iou.size:
            filas, cols = asignacion_optima(-iou)
            for a, b in zip(filas, cols):
                if iou[a, b] > 0:
                    pares[int(ids_p[b])] = int(ids_g[a])
                    emparejados.add(int(ids_p[b]))
        for ip in ids_p:
            m = p == ip
            if int(ip) in emparejados:
                pred_al[m] = pares[int(ip)]
            else:
                espurio |= m
    return pred_al, espurio, pares


def error_por_corte(gt: np.ndarray, pred_al: np.ndarray, espurio: np.ndarray) -> np.ndarray:
    """Voxeles en error de cada corte axial: hueso donde GT y prediccion (alineada) difieren."""
    err = ((gt != pred_al) & ((gt > 0) | (pred_al > 0))) | espurio
    return err.reshape(err.shape[0], -1).sum(1)


def elegir_corte(gt, pred_al, espurio) -> tuple[int, int]:
    """(indice de corte, voxeles en error) del corte con mas error."""
    por_corte = error_por_corte(gt, pred_al, espurio)
    z = int(np.argmax(por_corte))
    return z, int(por_corte[z])


# ---------------------------------------------------------------- figura
def _rgb(lbl: int) -> tuple[int, int, int]:
    from src.dashboard.visualizadores import color_fragmento  # colores del dashboard
    return tuple(int(v) for v in color_fragmento(int(lbl))[4:-1].split(","))


def _overlay(base01: np.ndarray, etq: np.ndarray, espurio: np.ndarray | None, alpha=0.55) -> np.ndarray:
    img = np.repeat(base01[..., None], 3, axis=2).astype(float)
    for lbl in np.unique(etq[etq > 0]):
        m = etq == lbl
        img[m] = (1 - alpha) * img[m] + alpha * np.array(_rgb(lbl)) / 255.0
    if espurio is not None and espurio.any():
        img[espurio] = (1 - alpha) * img[espurio] + alpha * np.array(ROJO) / 255.0
    # contorno blanco de cada fragmento: sin el, un secundario pegado al principal (mismo color,
    # tono mas claro) casi no se distingue, y esa es justo la separacion que importa ver
    from skimage.segmentation import find_boundaries
    particion = etq.astype(np.int32)
    if espurio is not None:
        particion[espurio] = 999
    img[find_boundaries(particion, mode="inner") & (particion > 0)] = 1.0
    return img


def _recorte(gt2d, pr2d, esp2d, margen=24):
    m = (gt2d > 0) | (pr2d > 0) | esp2d
    if not m.any():
        return slice(None), slice(None)
    ys, xs = np.where(m)
    H, W = m.shape
    return (slice(max(0, ys.min() - margen), min(H, ys.max() + margen)),
            slice(max(0, xs.min() - margen), min(W, xs.max() + margen)))


def figura_caso(cid: str, hu_z: np.ndarray, gt_z: np.ndarray, pred_z: np.ndarray, esp_z: np.ndarray,
                z: int, ventana_hu, titulo_extra: str, salida: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lo, hi = ventana_hu
    base = np.clip((hu_z.astype(float) - lo) / (hi - lo), 0, 1)
    sl = _recorte(gt_z, pred_z, esp_z)
    base, gt_z, pred_z, esp_z = base[sl], gt_z[sl], pred_z[sl], esp_z[sl]

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.9))
    axs[0].imshow(base, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title(f"CT (ventana de hueso {lo} a {hi} HU)", fontsize=10)
    axs[1].imshow(_overlay(base, gt_z, None))
    axs[1].set_title("Ground truth", fontsize=10)
    axs[2].imshow(_overlay(base, pred_z, esp_z))
    axs[2].set_title("Predicción (ids alineados con el GT)", fontsize=10)
    for a in axs:
        a.axis("off")

    handles = [Patch(color=np.array(_rgb(l)) / 255.0, label=nombre_fragmento(int(l)))
               for l in np.unique(np.concatenate([gt_z[gt_z > 0], pred_z[pred_z > 0]]))]
    if esp_z.any():
        handles.append(Patch(color=np.array(ROJO) / 255.0, label="predicho sin pareja en el GT"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8), fontsize=8, frameon=False)
    fig.suptitle(f"Caso {cid} · corte z = {z} · {titulo_extra}", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- ejecucion (necesita torch + datos)
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="pesos/completo.pth")
    ap.add_argument("--device", default=None)
    ap.add_argument("--casos", nargs="*", default=None, help="ids a graficar (por defecto, peores_5 de la evaluacion)")
    ap.add_argument("--evaluacion", default=str(EVAL_REFERENCIA))
    ap.add_argument("--salida", default="docs/figuras")
    args = ap.parse_args()

    ev = json.load(open(args.evaluacion, encoding="utf-8"))["por_nivel_degradacion"]["0"]
    metricas = {x["caso"]: x for x in ev.get("peores_5", [])}
    casos_ids = args.casos or list(metricas)
    if not casos_ids:
        sys.exit("No hay casos: el JSON no trae peores_5 y no se paso --casos")
    print("casos:", casos_ids)

    from src.data.io import leer_volumen, listar_casos
    from src.inferencia import cargar_modelo, inferir_volumen
    from src.utils import cargar_config, dispositivo

    cfg = cargar_config(args.config)
    dev = dispositivo(args.device)
    modelo = cargar_modelo(args.ckpt, cfg, dev)
    disponibles = {c["id"]: c for c in listar_casos(cfg["rutas"]["imagenes"], cfg["rutas"]["etiquetas"])}

    resumen = {}
    for cid in casos_ids:
        if cid not in disponibles or disponibles[cid]["etiqueta"] is None:
            print(f"[aviso] caso {cid}: no esta en {cfg['rutas']['imagenes']} o no tiene etiqueta; se omite")
            continue
        print(f"\n== caso {cid}")
        hu, spacing, _ = leer_volumen(disponibles[cid]["imagen"])
        gt = leer_volumen(disponibles[cid]["etiqueta"])[0].astype(np.uint8)
        res = inferir_volumen(modelo, hu, spacing, cfg, dev, amp=cfg["entrenamiento"]["amp"])
        pred = res["etiquetas"].astype(np.uint8)
        assert pred.shape == gt.shape == hu.shape, (pred.shape, gt.shape, hu.shape)

        pred_al, esp, pares = alinear_ids(gt, pred)
        z, n_err = elegir_corte(gt, pred_al, esp)
        m = metricas.get(cid, {})
        extra = (f"Dice {m['dice']:.2f} · HD95 {m['hd95']:.0f} mm · " if m else "") + f"{n_err:,} vóxeles en error"
        figura_caso(cid, hu[z], gt[z], pred_al[z], esp[z], z, cfg["datos"]["ventana_hu"], extra,
                    Path(args.salida) / f"peor_{cid}.png")
        resumen[cid] = {"corte": z, "voxeles_error": n_err, "fragmentos_gt": int(len(np.unique(gt[gt > 0]))),
                        "fragmentos_pred": int(len(np.unique(pred[pred > 0]))),
                        "fragmentos_sin_pareja": int(len(np.unique(pred[esp]))), **{k: v for k, v in m.items() if k != "caso"}}
        print(f"   corte {z} · {n_err:,} vóxeles en error · guardado peor_{cid}.png")
        del hu, gt, pred, pred_al, esp, res

    with open(Path(args.salida) / "peores_casos.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=1)
    print("\nlisto. Revisa cada figura y completa docs/peores_casos.md")


if __name__ == "__main__":
    main()
