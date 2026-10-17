# Model Card — PENGWIN `completo` (CBAM, sin transfer learning)

> **Advertencia:** este sistema tiene fines exclusivamente académicos. No es un dispositivo
> médico, no ha sido validado clínicamente y no debe usarse para apoyar decisiones
> quirúrgicas reales.

Pesos: `pesos/completo.pth`. Configuración: `configs/default.yaml`. Todas las cifras de esta tarjeta salen de
`docs/resultados/completo/evaluacion_cuda.json` (evaluación en test, nivel de degradación 0, salvo que se indique otra cosa).

## Arquitectura
- Backbone: FundidoraPC extendida de 5 bloques (32 → 64 → 128 → 256 → 512 canales) + CBAM en los bloques 4 y 5 (reducción 16), antes de bifurcar hacia las cabezas.
- Cabezas, todas entrenadas desde cero:
  - Clasificación anatómica por corte: presencia de SA/LI/RI y si tienen fragmentos secundarios.
  - Detección anchor-free sobre un grid de 16×16 con NMS propio, una caja por región y corte.
  - Segmentación: decoder tipo U-Net con tres salidas (región, rol principal/secundario y borde de contacto). Los fragmentos se separan después en 3D (componente conexa más grande como principal y watershed sembrado con núcleos sin borde).
  - Regresión de distancia: mapa denso de log(1 + mm) al fragmento principal.
- Parámetros entrenables: 6,791,354 (del perfil de memoria del notebook).
- Transfer learning: **no**. El backbone se entrena desde cero. La variante con transfer learning es la ablación `con_tl`.
- Entrenamiento: currículum de 4 etapas (detección 15 épocas, + segmentación 20, + distancia con backbone congelado 10, ajuste conjunto 10), 55 épocas en total. Batch 16, lr 1e-3, precisión mixta, semilla 42. Según el log de entrenamiento del notebook, 5.2 h en una RTX 3060 Laptop (6 GB). Se conserva el mejor checkpoint por validación de cada etapa.
- Pérdida: siete términos con pesos calibrados midiendo la norma del gradiente de cada uno sobre el backbone (`scripts/calibrar_lambdas.py`).

## Datos de entrenamiento
- Dataset: PENGWIN Task 1 (CT), Zenodo 10.5281/zenodo.10927452. Se usan los 100 casos con etiquetas.
- Casos entrenamiento / validación / test: 70 / 15 / 15, repartidos **por paciente** (nunca por corte) y estratificados por número de fragmentos secundarios. Semilla 42.
- Splits: `data/splits.json`, fijado con SHA-256 que termina en `99267ddb2310271e`.
- Casos de test evaluados: 15 (002, 004, 005, 015, 017, 018, 023, 028, 038, 039, 043, 048, 076, 084, 085).
- Preprocesamiento: orientación LPS, ventana de hueso −200 a 1200 HU, corte a 256×256 con 2 cortes vecinos como canales (2.5D). El espaciado se lee del header, de modo que toda distancia sale en mm.

## Métricas obtenidas
Test, 15 pacientes, nivel de degradación 0.

| Tarea | Métrica | Objetivo | Obtenido | ¿Cumple? |
|---|---|---|---|---|
| Clasificación | F1-score | >= 0.85 | 0.987 | Sí |
| Clasificación | AUC | >= 0.85 | 0.999 | Sí |
| Detección | IoU promedio | >= 0.65 | 0.885 | Sí |
| Detección | mAP@0.50 | >= 0.65 | 0.877 | Sí |
| Detección | mAP@[0.50:0.95] | >= 0.40 | 0.750 | Sí |
| Segmentación de fragmentos | Dice | >= 0.85 | 0.676 | **No** |
| Segmentación de fragmentos | IoU | >= 0.70 | 0.597 | **No** |
| Segmentación (extra) | HD95 | — | 21.5 mm | — |
| Segmentación (extra) | ASSD | — | 4.57 mm | — |
| Medición de distancia | error medio vs. GT, método geométrico | — | 5.06 mm | — |
| Medición de distancia | error medio vs. GT, cabeza de regresión | — | 6.05 mm | — |

Detalle de fragmentos:
- Fragmentos: 91 en el ground truth y 133 predichos (precisión 0.60, recall 0.88).
- Fragmentos espurios (predichos sin correspondencia en el GT): 53, con volumen mediano de 2.3 mL.
- Fragmentos secundarios del GT con pareja: 22 de 46.

Clasificación y detección se cumplen con margen; la segmentación de fragmentos no alcanza sus objetivos.

### Robustez a imágenes de baja calidad
| Nivel de degradación | F1 | mAP@0.50 | Dice de fragmento | Error de distancia (geométrico, mm) |
|---|---|---|---|---|
| 0 | 0.987 | 0.877 | 0.676 | 5.06 |
| 1 | 0.987 | 0.877 | 0.660 | 5.56 |
| 2 | 0.987 | 0.876 | 0.627 | 5.19 |
| 3 | 0.971 | 0.845 | 0.574 | 6.24 |

El nivel 0 es la imagen original. El nivel 3 es la peor degradación: ruido 0.10, desenfoque 1.8, submuestreo ×3 y gamma 1.4 (`NIVELES_DEGRADACION` en `src/data/aumentos.py`). La clasificación y la detección casi no cambian; la segmentación de fragmentos cae de 0.676 a 0.574.

Dice de fragmento según la calidad real de la imagen (terciles de nitidez, 5 casos por tercil): alta 0.782, media 0.636, baja 0.664. Con tan pocos casos no se puede afirmar una tendencia.

## Latencia
- GPU (RTX 3060 Laptop): unos **8 ms por corte**. Los niveles 1 a 3 midieron 8.2, 8.0 y 7.9 ms. El nivel 0 reportó 22.3 ms, probablemente por el calentamiento de la GPU en la primera pasada (hipótesis; conviene repetir la medición en caliente).
- CPU: 103.0, 102.8 y 102.6 ms por corte en los casos 005, 023 y 048 (promedio ≈ 102.8 ms), medidos con `python scripts/inferir.py --ckpt pesos/completo.pth --ct data/raw/images/<caso>.mha --device cpu`. Equipo: AMD Ryzen 7 5800H (8 núcleos), 31.3 GB de RAM (32 GB instalados).

## Alcances
- Detectar y segmentar el sacro y los dos coxales en tomografías de pelvis, separar sus fragmentos y estimar en milímetros la distancia de cada fragmento al principal.
- Referencia de la receta base a 256 px. No es el modelo final: el dashboard usa `sin_cbam_384` (ver `docs/model_card_sin_cbam_384.md`).
- Sirve como punto de partida para comparar variantes (con y sin CBAM, con y sin transfer learning).

## Limitaciones conocidas
- **Fragmentos en contacto:** los que se tocan en la superficie de fractura pueden fusionarse por errores de segmentación (ver watershed en `src/postprocess/`). En validación, el 38 % de los fragmentos secundarios (17 de 45) quedó absorbido por el fragmento principal (`docs/resultados/diagnostico_val/resumen.json`).
- **Dice bajo en fragmentos secundarios:** en validación, 0.91 para los principales y 0.48 para los secundarios. Los principales superan el objetivo de 0.85; el déficit de Dice de fragmento viene de los secundarios.
- **Fragmentos espurios:** 53 fragmentos predichos sin correspondencia en el GT (precisión de fragmentos 0.60). Casi todos pequeños, pero algunos casos tienen espurios de 18 a 23 mL (ver `docs/peores_casos.md`).
- **Error de la cabeza de distancia:** 6.1 mm de error medio, peor que el 5.1 mm del método geométrico sobre la máscara. Casi todas las distancias reales rondan 1 mm, así que probablemente la cabeza vio pocos desplazamientos grandes.
- **Datos escasos:** solo 100 casos con etiquetas, 70 para entrenar. El test tiene 15 pacientes, por lo que las cifras tienen una incertidumbre grande y un caso malo mueve el promedio.
- **Sin validación clínica** ni evaluación en otros centros o escáneres.

## Estudio de ablación
Cada par difiere en un solo factor (misma semilla 42, mismo post-proceso con fragmento mínimo de 1 mL y mismos 15 pacientes de test). Cifras de `docs/resultados/<modelo>/evaluacion_cuda.json` (test, nivel 0) y `docs/resultados/<modelo>/diagnostico_val/resumen.json` o `docs/resultados_ablacion_sam.md` (validación).

| Configuración | Dice de fragmento (test) | mAP@0.50 (test) | Dice de fragmento (validación) | Secundarios absorbidos (validación) |
|---|---|---|---|---|
| `completo`: 256 px, con CBAM | 0.676 | 0.877 | 0.695 | 17 de 45 |
| `sin_cbam`: 256 px, sin CBAM | 0.667 | 0.876 | 0.736 | 11 de 45 |
| `con_tl`: 256 px, con CBAM y backbone del Taller 3 | 0.661 | 0.875 | 0.658 | 22 de 45 |
| `mejorado_384`: 384 px, con CBAM | 0.719 | 0.907 | 0.719 | 19 de 45 |
| **`sin_cbam_384`: 384 px, sin CBAM (modelo final)** | **0.700** | **0.906** | **0.793** | **7 de 45** |

- **CBAM:** quitarlo mejora la validación en las dos resoluciones (+0.041 a 256 px y +0.074 a 384 px) y casi no cambia el test (−0.009 y −0.019, dentro de lo que varía un conjunto de 15 pacientes). No aporta de forma medible, así que se retiró.
- **Transfer learning:** no mejora el Dice (0.661 frente a 0.676), pero reduce los fragmentos espurios (precisión 0.678 frente a 0.602).
- **384 px:** mejora el Dice frente a la receta base en test y en validación.
- El modelo final se eligió por **validación**, con la regla fijada antes de ver resultados; el test solo se reporta.
