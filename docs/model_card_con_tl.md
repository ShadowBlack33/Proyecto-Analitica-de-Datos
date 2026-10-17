# Model Card — PENGWIN `con_tl` (CBAM, con transfer learning)

> **Advertencia:** este sistema tiene fines exclusivamente académicos. No es un dispositivo
> médico, no ha sido validado clínicamente y no debe usarse para apoyar decisiones
> quirúrgicas reales.

Pesos: `pesos/con_tl.pth`. Configuración: `configs/ablacion_con_tl.yaml` (hereda `default.yaml` y solo cambia `modelo.pesos_backbone`).
Todas las cifras salen de `docs/resultados/con_tl/evaluacion_cuda.json` (test, nivel de degradación 0). Esta variante solo se evaluó en el nivel 0.

## Arquitectura
- Idéntica al modelo `completo`: FundidoraPC extendida de 5 bloques (32 → 512 canales) + CBAM en los bloques 4 y 5, y las mismas cuatro cabezas (clasificación, detección anchor-free con NMS propio, segmentación con watershed 3D y regresión de distancia).
- Transfer learning: **sí, solo en el backbone.** Se inicializa desde la FundidoraPC entrenada en el Taller 3 (Face Mask Detection), el único lugar donde el enunciado permite transfer learning. Las cabezas parten desde cero.
- El resto del entrenamiento es el del modelo principal: currículum de 4 etapas (15 + 20 + 10 + 10 épocas), batch 16, lr 1e-3, precisión mixta, semilla 42.

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
| Detección | IoU promedio | >= 0.65 | 0.882 | Sí |
| Detección | mAP@0.50 | >= 0.65 | 0.875 | Sí |
| Detección | mAP@[0.50:0.95] | >= 0.40 | 0.741 | Sí |
| Segmentación de fragmentos | Dice | >= 0.85 | 0.661 | **No** |
| Segmentación de fragmentos | IoU | >= 0.70 | 0.583 | **No** |
| Segmentación (extra) | HD95 | — | 20.3 mm | — |
| Segmentación (extra) | ASSD | — | 5.14 mm | — |
| Medición de distancia | error medio vs. GT, método geométrico | — | 5.40 mm | — |
| Medición de distancia | error medio vs. GT, cabeza de regresión | — | 6.53 mm | — |

Detalle de fragmentos:
- Fragmentos: 91 en el ground truth y 121 predichos (precisión 0.68, recall 0.90).
- Fragmentos espurios (predichos sin correspondencia en el GT): 39, con volumen mediano de 2.5 mL.
- Fragmentos secundarios del GT con pareja: 20 de 46.

### Frente al modelo `completo` (sin transfer learning)
| Métrica | `completo` | `con_tl` | Diferencia |
|---|---|---|---|
| F1 de clasificación | 0.987 | 0.987 | +0.000 |
| mAP@0.50 | 0.877 | 0.875 | -0.002 |
| mAP@[0.50:0.95] | 0.750 | 0.741 | -0.009 |
| Dice de fragmento | 0.676 | 0.661 | -0.016 |
| Precisión de fragmentos | 0.602 | 0.678 | +0.076 |
| Fragmentos espurios | 53 | 39 | -14 |

El transfer learning no mejora de forma apreciable las métricas principales: clasificación y detección son prácticamente iguales y el Dice de fragmento es ligeramente menor. Lo único que cambia algo es el balance de fragmentos: 39 espurios frente a 53 (precisión 0.68 frente a 0.60), con menos secundarios emparejados (20 frente a 22). Con 15 pacientes de test, diferencias de este tamaño no son concluyentes. Una explicación posible, que no se probó, es que el Taller 3 era detección de mascarillas en imágenes RGB corrientes y sus filtros aportan poco a tomografías.

## Latencia
- GPU (RTX 3060 Laptop): 7.8 ms por corte (nivel 0).
- CPU: 103.4 ms por corte en el caso 005 (una sola medición), con `python scripts/inferir.py --ckpt pesos/con_tl.pth --ct data/raw/images/005.mha --device cpu`. Equipo: AMD Ryzen 7 5800H (8 núcleos), 31.3 GB de RAM. Para comparar, `completo` dio 103.0 ms en el mismo caso.

## Alcances
- Comparar el efecto del transfer learning en el backbone frente al entrenamiento desde cero (ablación obligatoria del enunciado).
- No es el modelo final del proyecto ni el que usa el dashboard (ver `docs/model_card_sin_cbam_384.md`).

## Limitaciones conocidas
- Las mismas del modelo `completo`: fragmentos en contacto que se fusionan por errores de segmentación, Dice bajo en secundarios, espurios, error de la cabeza de distancia (6.5 mm frente a 5.4 mm del método geométrico) y solo 100 casos (70 de entrenamiento). Ver `docs/model_card_completo.md`.
- Una sola corrida por configuración y un test de 15 pacientes: no hay forma de separar el efecto del transfer learning del azar del entrenamiento.
- Sin validación clínica.

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
