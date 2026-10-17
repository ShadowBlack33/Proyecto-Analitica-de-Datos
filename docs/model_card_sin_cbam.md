# Model Card — PENGWIN `sin_cbam` (256 px, sin CBAM; ablación)

> **Advertencia:** este sistema tiene fines exclusivamente académicos. No es un dispositivo
> médico, no ha sido validado clínicamente y no debe usarse para apoyar decisiones
> quirúrgicas reales.

Pesos: `pesos/sin_cbam.pth`. Configuración: `configs/ablacion_sin_cbam.yaml` (hereda `default.yaml` y solo quita CBAM). Las cifras salen de
`docs/resultados/sin_cbam/evaluacion_cuda.json` (test, nivel de degradación 0; esta variante solo se evaluó en ese nivel) y, las de validación, de `docs/resultados/sin_cbam/diagnostico_val/resumen.json`.

## Arquitectura
- Igual que `completo`, pero **sin CBAM** (`cbam_en_bloques: []`): FundidoraPC extendida de 5 bloques (32 → 512 canales) y las mismas cuatro cabezas (clasificación, detección anchor-free con NMS propio, segmentación con watershed 3D y regresión de distancia).
- Parámetros: 6 750 198 (41 156 menos que `completo`).
- Transfer learning: no.
- Entrenamiento: el mismo currículum de 4 etapas que `completo` (55 épocas, batch 16, precisión mixta, semilla 42), a 256 px. Unas 3.2 h en una RTX 3060 Laptop (frente a 5.2 h de `completo`).
- Es la ablación de CBAM de la receta base y el punto de partida del modelo final `sin_cbam_384`.

## Datos de entrenamiento
- Dataset: PENGWIN Task 1 (CT), Zenodo 10.5281/zenodo.10927452. Se usan los 100 casos con etiquetas.
- Casos entrenamiento / validación / test: 70 / 15 / 15, repartidos **por paciente** y estratificados por número de fragmentos secundarios. Semilla 42.
- Splits: `data/splits.json`, fijado con SHA-256 que termina en `99267ddb2310271e`.
- Casos de test evaluados: 15 (002, 004, 005, 015, 017, 018, 023, 028, 038, 039, 043, 048, 076, 084, 085).
- Preprocesamiento: ventana de hueso −200 a 1200 HU, corte a 256×256 con los 2 cortes vecinos como canales (2.5D).

## Métricas obtenidas
Test, 15 pacientes, nivel de degradación 0.

| Tarea | Métrica | Objetivo | Obtenido | ¿Cumple? |
|---|---|---|---|---|
| Clasificación | F1-score | >= 0.85 | 0.980 | Sí |
| Clasificación | AUC | >= 0.85 | 0.998 | Sí |
| Detección | IoU promedio | >= 0.65 | 0.882 | Sí |
| Detección | mAP@0.50 | >= 0.65 | 0.876 | Sí |
| Detección | mAP@[0.50:0.95] | >= 0.40 | 0.751 | Sí |
| Segmentación de fragmentos | Dice | >= 0.85 | 0.667 | **No** |
| Segmentación de fragmentos | IoU | >= 0.70 | 0.590 | **No** |
| Segmentación (extra) | HD95 | — | 23.2 mm | — |
| Segmentación (extra) | ASSD | — | 5.52 mm | — |
| Medición de distancia | error medio vs. GT, método geométrico | — | 4.84 mm | — |
| Medición de distancia | error medio vs. GT, cabeza de regresión | — | 6.49 mm | — |

Detalle de fragmentos:
- 91 fragmentos en el ground truth y 125 predichos (precisión 0.65, recall 0.89).
- Fragmentos espurios: 44, con volumen mediano de 3.4 mL.
- Fragmentos secundarios del GT con pareja: 20 de 46.
- F1 por región: SA 0.971, LI 0.979, RI 0.992.

### Comparación con `completo` (misma receta, con CBAM)
| Métrica | `completo` | `sin_cbam` |
|---|---|---|
| Dice de fragmento, test | 0.676 | 0.667 |
| Precisión / recall de fragmentos, test | 0.60 / 0.88 | 0.65 / 0.89 |
| Dice de fragmento, validación | 0.695 | 0.736 |
| Dice de secundarios, validación | 0.479 | 0.557 |
| Secundarios absorbidos, validación | 17 de 45 | 11 de 45 |

Quitar CBAM no empeora el test de forma medible y mejora la validación, sobre todo en los secundarios.

## Latencia
- GPU (RTX 3060 Laptop): 8.1 ms por corte (nivel 0, según el JSON de evaluación).
- CPU: no se midió para esta variante. Por resolución y arquitectura debería parecerse a `completo` (≈ 103 ms por corte), pero es una suposición.

## Alcances
- Medir el efecto de CBAM en la receta base (ablación obligatoria del enunciado).
- Punto de partida del modelo final `sin_cbam_384`. No es el modelo que usa el dashboard.

## Limitaciones conocidas
- **Fragmentos en contacto:** los que se tocan en la superficie de fractura pueden fusionarse por errores de segmentación. En validación, 11 de 45 secundarios quedan absorbidos por el principal.
- **Dice bajo en secundarios:** en validación, 0.92 para los principales y 0.56 para los secundarios.
- **Fragmentos espurios:** 44 en test (precisión 0.65).
- **Error de la cabeza de distancia:** 6.5 mm, peor que los 4.8 mm del método geométrico.
- **Evaluación limitada:** solo nivel de degradación 0; una sola semilla y 15 pacientes de test, así que no se puede separar el efecto de CBAM del azar del entrenamiento con certeza.
- **Sin validación clínica.** No usar para decisiones médicas.

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
