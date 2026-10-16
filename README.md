# 🦴 PENGWIN · Pelvic Fracture Detection → Fragment Segmentation → Displacement in mm

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)](https://pytorch.org/)
[![SimpleITK](https://img.shields.io/badge/SimpleITK-CT_I%2FO-2E7D32)](https://simpleitk.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-watershed_%7C_marching_cubes-F7931E)](https://scikit-image.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-3D_viewer-3F4F75)](https://plotly.com/)

**PENGWIN** is an end-to-end computer-vision system for pelvic fractures on CT: it finds each pelvic bone on every slice, segments every bone fragment and assigns it to the bone it broke from, measures how far each loose fragment ended up from its parent bone **in millimetres**, and shows it all in an interactive 3D dashboard — built on a custom multi-task network, with no off-the-shelf detector or segmenter behind it.

> Data: **PENGWIN Task 1 (CT)**, MICCAI 2024 challenge — 100 annotated CT scans, fragment masks validated by orthopaedic surgeons ([Zenodo](https://doi.org/10.5281/zenodo.10927452), CC BY 4.0).
> **Team**: Carlos Andrés Orozco Caicedo · José David Mesa Ramírez · Sara Lucía Rojas Mejía · Esteban Cobo Gómez 🇨🇴

> ⚠️ Research / educational project. **Not a medical device**, not clinically validated, not for surgical decisions.

---

## 🎯 Project direction

Pelvic fractures are hard to plan because a surgeon needs three answers per fragment: *which bone is it from, where is it, and how far did it move?* Answering that by hand takes up to an hour per complex case.

This project deliberately **does not** wrap YOLO, Detectron2 or a pretrained Mask R-CNN. Every piece is built and tested in-house, so each design decision can be measured:

- ✅ **One backbone, four heads** — a 5-block CNN (CBAM attention was tested and removed after an ablation) feeding anatomical classification, an anchor-free detector (own grid + own NMS), instance segmentation and a dense distance-regression head.
- ✅ **3D where it matters** — the network reads 2.5D slices (each slice plus its neighbours), but fragment separation and every distance are computed on the stacked volume at native resolution, using the real voxel spacing.
- ✅ **Diagnostics before tuning** — a dedicated tool explains *why* a fragment is wrong (absorbed, split, merged, lost) instead of blindly sweeping hyper-parameters.
- ✅ **Robustness and interpretability as first-class outputs** — four synthetic degradation levels, BatchNorm γ analysis, Smooth Grad-CAM vs. objectness, and box-level IoU across the whole test set.
- ✅ **Honest evaluation** — patient-level splits (never slice-level), parameters chosen on validation and applied once to test, and metrics that look at both sides (Dice *and* fragment precision).

## 📊 Results

Held-out test set: **15 patients** never seen during training or tuning. Final model: **`sin_cbam_384`** — 384 px, without CBAM attention, selected on validation (see *How the final model was chosen*).

| Task | Metric | Base (256 px) | 384 px with CBAM | **Final (384 px, no CBAM)** |
|---|---|---|---|---|
| Anatomical classification | F1 / AUC | 0.987 / 0.999 | 0.985 / 0.999 | 0.981 / 0.999 |
| Bone detection | Box IoU | 0.885 | 0.898 | 0.893 |
| Bone detection | mAP@0.50 / mAP@[0.50:0.95] | 0.877 / 0.750 | 0.907 / 0.776 | 0.906 / 0.767 |
| Bone (semantic) segmentation | Dice / IoU | 0.947 / 0.900 | 0.972 / 0.946 | 0.972 / 0.946 |
| Fragment segmentation | Dice / IoU | 0.676 / 0.597 | 0.719 / 0.645 | 0.700 / 0.628 |
| Fragment segmentation | HD95 / ASSD | 21.5 mm / 4.6 mm | 23.3 mm / 4.9 mm | 23.2 mm / 5.3 mm |
| Fragment instances | Precision / recall | 0.602 / 0.879 | 0.600 / 0.923 | 0.600 / 0.923 |
| Displacement | Mean abs. error (geometric) | 5.1 mm | 3.5 mm | 4.0 mm |
| **Validation** | **Fragment Dice (decides)** | 0.695 | 0.719 | **0.793** |
| Validation | Secondary-fragment Dice / absorbed | 0.479 / — | 0.509 / 19 of 45 | **0.652 / 7 of 45** |
| Model | Parameters | 6.79 M | 6.79 M | 6.75 M |
| Inference | Latency (RTX 3060 Laptop) | ~8 ms per slice | ~16 ms per slice | ~13.5 ms per slice |

**How the final model was chosen.** Every decision is made on **validation**; test is only reported. Two steps:

1. **256 px → 384 px.** Pre-registered rule: adopt the new recipe (384 px, ×3 weight on the secondary class, 35° rotation on 80 % of samples, thicker contact border) only if validation fragment Dice improves by more than 0.01. It went from 0.695 to 0.719, and test also improved (0.676 → 0.719).
2. **Removing CBAM.** The ablation (below) showed CBAM does not help: without it, validation Dice rose to **0.793** (+0.074) and absorbed secondary fragments fell from 19 to 7 of 45, while test stayed within noise (0.719 → 0.700, same precision and recall). Following the same rule, the model without CBAM is the final one — simpler and faster.

Single seed per model and 15 patients per split: differences of ~0.02 between models are within split-to-split variation, which is why the rule uses validation and both splits are reported.

**Ablation: does CBAM help?** Each pair differs only in CBAM (same recipe, seed, post-processing and data).

| Recipe | With CBAM → without CBAM, validation Dice | Test Dice |
|---|---|---|
| Base, 256 px | 0.695 → **0.736** (+0.041) | 0.676 → 0.667 (−0.009) |
| Final, 384 px | 0.719 → **0.793** (+0.074) | 0.719 → 0.700 (−0.019) |

Same pattern at both resolutions: CBAM does not contribute measurably, so it was removed. Transfer learning from the Taller 3 backbone (base recipe) also did not improve Dice (0.661) but reduced spurious fragments (precision 0.678).

**Robustness of the final model** — same test set with increasing noise, blur, resolution loss and contrast shift:

| Degradation | none | light | medium | heavy |
|---|---|---|---|---|
| F1 classification | 0.981 | 0.979 | 0.983 | 0.978 |
| mAP@0.50 | 0.906 | 0.906 | 0.878 | 0.876 |
| Fragment Dice | 0.700 | 0.694 | 0.669 | 0.614 |
| Fragment precision | 0.600 | 0.606 | 0.565 | 0.441 |
| Displacement MAE (mm) | 4.0 | 3.8 | 4.0 | 5.8 |

Under the heaviest degradation, detection (mAP@0.50 0.876) is still at the level of the base model on clean images (0.877); fragment separation degrades most and produces more spurious fragments.

**What the analysis showed**

- **Finding and locating bones is solved and robust** — detection and classification stay well above target even under heavy degradation (final model: mAP@0.50 0.876 and F1 0.978 at the heaviest level).
- **Orientation is handled from the metadata.** 34 of the 100 CTs are stored rotated 180° in-plane (direction matrix `-1,-1,1`). Every volume is reoriented losslessly to RAI from its direction matrix (no interpolation); we checked that the slice normal equals the cross product of the in-plane axes and that no volume is mirrored. Residual patient tilt, measured from the centroids of the two hip bones, has a median of 3°, so it is covered by rotation augmentation instead of resampling the volume (which would interpolate labels and distort millimetre distances, since z spacing differs from x/y).
- **The hard part is touching fragments.** Main fragments reach Dice 0.93; the gap is in secondary fragments. With CBAM, diagnostics showed about 40 % of them *absorbed* into the main fragment — mostly large ones, where a slice-by-slice network can't tell which piece is "the main one" (a 3D property). Thirteen post-processing variants didn't fix it. The 384-px recipe improved the *quality* of the fragments it separates (Dice of correctly separated secondaries 0.76 → 0.82) but not the absorption rate; **removing CBAM** is what reduced it on validation (19 → 7 of 45 secondaries absorbed). Absorption remains the main limitation of deciding a 3D property slice by slice.
- **Dice hides false fragments.** On validation the model predicted 216 fragments for 90 real ones. A fragment-precision metric exposed it, and a 1 mL minimum fragment size chosen on validation keeps test precision at 0.60, at a measured Dice cost: a documented precision/Dice trade-off.
- **Transfer learning didn't help.** Initialising the backbone from a detector trained on natural photos gave the same results as training from scratch, and both models end with near-identical BatchNorm γ — training washes the initialisation out.
- **The network uses all its capacity** — no BatchNorm channel is switched off in any layer. Grad-CAM shows the classifier uses anatomical context (the opposite side and the spine) to tell left from right, while the detection head's objectness stays tightly on the bone.

---

## 🏗️ Architecture

```mermaid
flowchart LR
  CT[CT volume .mha] --> P[HU window + 2.5D slices 3x256x256]
  P --> B[Backbone: 5 conv blocks 32→512]
  B --> C[Classification head: which bones are in the slice]
  B --> D[Detection head: anchor-free 16x16 grid + own NMS]
  B --> U[U-Net decoder]
  U --> S[Segmentation head: bone, fragment role, contact border]
  U --> R[Regression head: dense distance map in mm]
  C & D & S & R --> V[3D post-processing: watershed instances + EDT distances]
  V --> O[Labelled fragments + displacement table + 3D mesh]
```

6.75 M parameters (final model). Trained in four curriculum stages (detection → segmentation → distance → joint fine-tuning) at 256 px, mixed precision, ~3 h on a 6 GB laptop GPU; then fine-tuned from those weights at 384 px (segmentation + joint stages, 20 epochs, ~3 h). Every layer is convolutional, so the same weights work at the higher resolution and the detection grid simply grows from 16×16 to 24×24.

```
src/
├── data/          .mha I/O with physical spacing, HU windowing, mmap slice cache, patient-level splits, augmentation
├── models/        extended backbone, CBAM, the four heads
├── losses/        7-term multi-task loss with gradient-calibrated weights
├── postprocess/   own NMS, 3D instance separation (watershed), edge-to-edge distance in mm
├── eval/          F1/AUC, IoU/mAP, Dice/IoU/HD95/ASSD, fragment precision, own Hungarian matching
├── analisis.py    BatchNorm γ, Smooth Grad-CAM, box analysis, fracture-case selection
├── inferencia.py  full-volume inference, GPU or CPU
└── dashboard/     Streamlit app + Plotly viewers

scripts/           one entry point per step: data, EDA, sanity checks, training, evaluation, diagnostics, SAM baseline
notebooks/         control notebook that runs the whole pipeline and renders every result
configs/           single source of truth for all parameters (+ ablation configs that change one factor)
pesos/             trained weights (main model and ablations)
docs/              results (JSON/CSV), installation guide, code guide
tests/             19 automated checks
```

---

## ⚙️ Install & Run

### 1. Environment

```bash
git clone https://github.com/ShadowBlack33/Proyecto-Analitica-de-Datos.git
cd Proyecto-Analitica-de-Datos

python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python scripts/verificar_entorno.py    # checks every library and project module, plus the GPU
```

Install PyTorch **before** `requirements.txt`, so you get the CUDA build. Only an up-to-date NVIDIA driver is needed — not the CUDA Toolkit.

### 2. Data

Download the three Task 1 zips from [Zenodo](https://doi.org/10.5281/zenodo.10927452) into the folder above the repo, then:

```bash
python scripts/organizar_datos.py --zips .. --md5   # verifies checksums, unpacks, pairs images with labels
python scripts/preparar_datos.py                    # slice cache + patient-level split (sha256 99267ddb…)
python scripts/eda.py
```

### 3. Train, evaluate, infer

```bash
python scripts/train.py --nombre completo                       # base model, full curriculum (~5 h on an RTX 3060)
powershell -ExecutionPolicy Bypass -File scripts/correr_noche.ps1      # 384-px recipe with CBAM (mejorado_384)
powershell -ExecutionPolicy Bypass -File scripts/correr_sin_cbam.ps1   # CBAM ablation at 256 and 384 px -> final model sin_cbam_384 (~6 h)

python scripts/evaluar.py --config configs/ablacion_sin_cbam_384.yaml --ckpt pesos/sin_cbam_384.pth --nombre sin_cbam_384 --degradacion 0 1 2 3
python scripts/inferir.py --config configs/ablacion_sin_cbam_384.yaml --ckpt pesos/sin_cbam_384.pth --ct data/raw/images/039.mha   # exports a labelled .mha
```

Pretrained weights are in `pesos/`, so evaluation and inference work without training. Interrupted runs resume with `--reanudar checkpoints/<name>/ultimo.pth`.

### 4. Dashboard

```bash
streamlit run src/dashboard/app.py
cloudflared tunnel --url http://localhost:8501      # optional: public link, works on a phone
```

### 5. Everything in one place

`notebooks/PENGWIN_pipeline.ipynb` walks through the full pipeline and renders every table and figure: EDA, sanity checks, training curves, test metrics, 2D/3D predictions, BatchNorm γ, Grad-CAM, box analysis and a gallery of fractured cases. It reads saved results and never retrains by accident.

---

## 🖥️ Dashboard

Three linked viewers over any CT:

- **Raw volume** — maximum-intensity projection of bone by HU threshold, before any model runs.
- **Slice-by-slice inference** — boxes, colour-coded fragment masks and the displacement in mm next to each fragment, with a slice slider.
- **3D reconstruction** — one marching-cubes mesh per fragment, coloured by parent bone (main fragment dark, loose fragments lighter), labelled with its displacement.

A fragment table and the measured latency sit alongside the viewers.

---

## 🔬 Analysis tooling

| Tool | What it answers |
|---|---|
| `scripts/diagnostico_fragmentos.py` | What happened to each real fragment (correct / absorbed / split / merged / lost), and whether the failure is the network or the post-processing |
| `… --barrido` / `--barrido_tamano` | Post-processing and minimum-fragment-size sweeps on **validation**, with a stated selection rule |
| `scripts/evaluar.py --degradacion 0 1 2 3` | Robustness under four fixed, reproducible degradation levels |
| `scripts/comparar_experimentos.py` | New recipe vs. base model on validation (decides) and test (reported), with the pre-registered adoption rule |
| `scripts/calibrar_lambdas.py` | Loss weights from gradient-norm balance on the shared backbone, instead of copied values |
| `scripts/overfit_test.py` · `perfil_memoria.py` | Correctness (can it memorise a batch?) and peak VRAM per resolution/batch |
| `scripts/baseline_sam.py` | Zero-shot SAM prompted with our boxes, as an out-of-domain baseline |
| `src/analisis.py` | BatchNorm γ per block, Smooth Grad-CAM vs. objectness, NMS before/after, box IoU on test |

---

## 🧾 Configuration

Everything lives in `configs/default.yaml`; ablations inherit from it and change exactly one factor (`configs/ablacion_sin_cbam.yaml`, `configs/ablacion_con_tl.yaml`). `configs/mejorado_384.yaml` inherits from it and overrides resolution (384), secondary-class weight (×3), rotation (35° on 80 % of samples), contact-border thickness and the starting weights. The **final model**, `configs/ablacion_sin_cbam_384.yaml`, inherits from `mejorado_384.yaml` and only removes CBAM (`cbam_en_bloques: []`) and starts from the base model without CBAM.

```yaml
datos:
  resolucion: 256
  contexto: 1                 # 2.5D input: slice ± 1 neighbour = 3 channels
  ventana_hu: [-200, 1200]
  split: [0.70, 0.15, 0.15]   # by patient
modelo:
  canales: [32, 64, 128, 256, 512]
  cbam_en_bloques: [4, 5]     # base recipe; the final model uses []
perdida:                      # gradient-calibrated weights
  lambda_cls: 0.5
  lambda_det_obj: 3.0
  lambda_det_box: 5.0
  lambda_sem: 1.0
  lambda_rol: 1.0
  lambda_borde: 2.0
  lambda_dist: 0.5
entrenamiento:
  batch_size: 16
  amp: true
postproceso:
  min_volumen_ml: 1           # chosen on validation with diagnostico_fragmentos.py --barrido_tamano
```

---

## 🔁 Reproducibility & tests

- Fixed seed across Python, NumPy and PyTorch; the patient split is versioned with a SHA-256 hash and refused if altered; every checkpoint stores a hash of its predictions on a fixed batch.
- `python -m pytest -q tests` runs 19 checks, including: own NMS vs. `torchvision.ops.nms`, own metrics and Hungarian matching vs. scikit-learn/SciPy, transfer-learning weight loading with numerical equivalence, exact ground-truth reconstruction by the post-processing, and fragment-precision bookkeeping.
- The pipeline runs on GPU or CPU; a synthetic-CT generator (`scripts/generar_datos_sinteticos.py`) lets the whole flow run without the real dataset.

---

## 🛠️ Troubleshooting

| Issue | Suggestion |
|---|---|
| `DLL load failed … An Application Control policy has blocked this file` (Windows) | Windows 11 **Smart App Control** blocks unsigned compiled files installed by pip. Turn it off in *Windows Security → App & browser control* (irreversible) or work inside WSL2 |
| `verificar_entorno.py` reports no GPU | PyTorch is the CPU build: reinstall with the `--index-url …/cu124` command and `--force-reinstall` |
| `CUDA out of memory` | Lower `batch_size` to 8 and set `acumulacion: 2` in the config |
| `OSError 22` from DataLoader workers on Windows | Pull the latest `main` (fixed), or set `num_workers: 0` |
| `splits.json` hash differs from `99267ddb…` | Your data differs from the team's: re-run `organizar_datos.py --md5` |
| Plotly figures don't render in the notebook | `pip install nbformat` and restart the kernel |

---

## 📜 Data, license & disclaimer

- **Data**: PENGWIN Task 1, MICCAI 2024 — [Zenodo](https://doi.org/10.5281/zenodo.10927452), CC BY 4.0. Cite the challenge if you use it.
- **Code**: © 2026 the authors.
- **Use**: research and educational purposes only. Not a medical device, not clinically validated, not for diagnosis or surgical planning.

## 👥 Team

**Carlos Andrés Orozco Caicedo** · **José David Mesa Ramírez** · **Sara Lucía Rojas Mejía** · **Esteban Cobo Gómez**
Data & AI Engineering · Universidad Autónoma de Occidente · Cali, Colombia 🇨🇴
