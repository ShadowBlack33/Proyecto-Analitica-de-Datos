# Model Card — PENGWIN `mejorado_384` (384 px, con CBAM; referencia de la ablación)

> **Advertencia:** este sistema tiene fines exclusivamente académicos. No es un dispositivo
> médico, no ha sido validado clínicamente y no debe usarse para apoyar decisiones
> quirúrgicas reales.

Pesos: `pesos/mejorado_384.pth`. Configuración: `configs/mejorado_384.yaml` (hereda de `configs/default.yaml`). Las cifras de esta tarjeta salen de
`docs/resultados/mejorado_384/evaluacion_cuda.json` (evaluación en test, nivel de degradación 0, salvo que se indique otra cosa) y, las de validación, de `docs/resultados/mejorado_384/diagnostico_val/resumen.json`.

## Arquitectura
- Misma arquitectura que `completo` (ver `docs/model_card_completo.md`): FundidoraPC de 5 bloques con CBAM en los bloques 4 y 5, y las cuatro cabezas (clasificación, detección anchor-free con NMS propio, segmentación con decoder tipo U-Net y regresión de distancia). Las capas son convolucionales, así que los mismos pesos funcionan a mayor resolución; el grid de detección pasa de 16×16 a 24×24.
- Transfer learning: **no** desde otro dataset. El modelo continúa desde los pesos de `completo` (`checkpoints/completo/mejor_conjunto.pth`).
- Qué cambia respecto a `completo` (según `configs/mejorado_384.yaml`):
  - resolución de entrada de 256 a **384 px**;
  - peso **×3** a la clase "secundario" en la pérdida de rol;
  - rotación de hasta **35°** en el 80 % de las muestras de entrenamiento;
  - borde de contacto más grueso (grosor 2).
- Entrenamiento (ajuste fino): etapa de segmentación 12 épocas y etapa conjunta 8 épocas, 20 en total; batch 12. Según el README, unas 3 h en una GPU laptop de 6 GB.
- Regla de decisión fijada antes de ver resultados: se adopta solo si mejora el Dice de fragmento en **validación** frente a `completo`. Pasó de 0.695 a 0.719. El test se reporta, no se usa para decidir. Es una sola semilla: la mejora no se midió con varias semillas.

## Datos de entrenamiento
- Dataset: PENGWIN Task 1 (CT), Zenodo 10.5281/zenodo.10927452. Se usan los 100 casos con etiquetas.
- Casos entrenamiento / validación / test: 70 / 15 / 15, repartidos **por paciente** y estratificados por número de fragmentos secundarios. Semilla 42.
- Splits: `data/splits.json`, fijado con SHA-256 que termina en `99267ddb2310271e`.
- Casos de test evaluados: 15 (002, 004, 005, 015, 017, 018, 023, 028, 038, 039, 043, 048, 076, 084, 085).
- Preprocesamiento: ventana de hueso −200 a 1200 HU, corte a 384×384 con 2 cortes vecinos como canales (2.5D). La orientación se corrige desde la matriz de direcciones del header (ver README) y el espaciado se lee del header, de modo que toda distancia sale en mm.

## Métricas obtenidas
Test, 15 pacientes, nivel de degradación 0.

| Tarea | Métrica | Objetivo | Obtenido | ¿Cumple? |
|---|---|---|---|---|
| Clasificación | F1-score | >= 0.85 | 0.985 | Sí |
| Clasificación | AUC | >= 0.85 | 0.999 | Sí |
| Detección | IoU promedio | >= 0.65 | 0.898 | Sí |
| Detección | mAP@0.50 | >= 0.65 | 0.907 | Sí |
| Detección | mAP@[0.50:0.95] | >= 0.40 | 0.776 | Sí |
| Segmentación de fragmentos | Dice | >= 0.85 | 0.719 | **No** |
| Segmentación de fragmentos | IoU | >= 0.70 | 0.645 | **No** |
| Segmentación (extra) | HD95 | — | 23.3 mm | — |
| Segmentación (extra) | ASSD | — | 4.92 mm | — |
| Medición de distancia | error medio vs. GT, método geométrico | — | 3.54 mm | — |
| Medición de distancia | error medio vs. GT, cabeza de regresión | — | 5.06 mm | — |

Detalle de fragmentos:
- Fragmentos: 91 en el ground truth y 140 predichos (precisión 0.60, recall 0.92).
- Fragmentos espurios (predichos sin correspondencia en el GT): 56, con volumen mediano de 2.6 mL.
- Fragmentos secundarios del GT con pareja: 26 de 46.

F1 por región según el JSON de evaluación: SA 0.969, LI 0.992, RI 0.994. La matriz de confusión de `scripts/matriz_confusion.py` (`docs/figuras/matriz_confusion.png`) da SA 0.969 (igual), LI 0.996 y RI 0.996, con un F1 promedio de 0.987 frente a 0.985 del JSON. La diferencia se explica por los conjuntos de cortes: la matriz usa los 4005 cortes de la caché (rango con hueso ± 15 cortes) y `evaluar.py` los 4821 cortes de los volúmenes completos. Los 816 cortes adicionales están fuera de la pelvis y solo pueden aportar falsos positivos, por lo que la evaluación completa da un F1 algo menor. El sacro coincide exacto; LI y RI cambian, coherente con cortes inferiores que incluyen los fémures.

Clasificación y detección se cumplen con margen; la segmentación de fragmentos mejora frente a `completo` (0.676) pero sigue sin alcanzar sus objetivos.

### Comparación con los otros modelos
| Métrica (test, nivel 0) | completo | con_tl | mejorado_384 |
|---|---|---|---|
| F1 de clasificación | 0.987 | — | 0.985 |
| mAP@0.50 | 0.877 | 0.875 | 0.907 |
| Dice de fragmento | 0.676 | 0.661 | 0.719 |
| Fragmentos espurios | 53 | 39 | 56 |
| Secundarios emparejados (de 46) | 22 | 20 | 26 |
| Error de distancia, geométrico (mm) | 5.06 | — | 3.54 |

### Robustez a imágenes de baja calidad
| Nivel de degradación | F1 | mAP@0.50 | Dice de fragmento | Error de distancia (geométrico, mm) |
|---|---|---|---|---|
| 0 | 0.985 | 0.907 | 0.719 | 3.54 |
| 1 | 0.984 | 0.907 | 0.701 | 3.94 |
| 2 | 0.985 | 0.907 | 0.672 | 4.70 |
| 3 | 0.979 | 0.877 | 0.617 | 3.47 |

El nivel 0 es la imagen original y el nivel 3 la peor degradación (`NIVELES_DEGRADACION` en `src/data/aumentos.py`). La clasificación y la detección casi no cambian hasta el nivel 2; la segmentación de fragmentos cae de 0.719 a 0.617.

Dice de fragmento según la calidad real de la imagen (terciles de nitidez, 5 casos por tercil): alta 0.802, media 0.714, baja 0.695. Con tan pocos casos no se puede afirmar una tendencia.

## Latencia
- GPU: el README reporta unos 16 ms por corte típicos y 36 ms de media en la corrida de test, en una GPU laptop de 6 GB. En el JSON de evaluación, la media de la corrida con nivel 0 es 38.8 ms por corte; la medición varía con el estado de la GPU y conviene repetirla en caliente.
- CPU: 219.7, 249.2 y 250.9 ms por corte en los casos 005, 023 y 048 (promedio ≈ 240 ms), medidos con `python scripts/inferir.py --config configs/mejorado_384.yaml --ckpt pesos/mejorado_384.pth --ct data/raw/images/<caso>.mha --device cpu`. Varía entre casos (el 005 salió más rápido) y no se investigó la causa. Equipo: AMD Ryzen 7 5800H (8 núcleos), 31.3 GB de RAM. Con `completo` el promedio en CPU fue ≈ 103 ms.

## Alcances
- Detectar y segmentar el sacro y los dos coxales en tomografías de pelvis, separar sus fragmentos y estimar en milímetros la distancia de cada fragmento al principal.
- Demostración académica en el dashboard (MIP del volumen, inferencia corte a corte y reconstrucción 3D).
- Fue el modelo final hasta la ablación de CBAM. Ahora es la referencia con CBAM de la receta a 384 px; el modelo final es `sin_cbam_384`, que solo le quita CBAM (ver `docs/model_card_sin_cbam_384.md`).

## Limitaciones conocidas
- **Fragmentos en contacto:** los que se tocan en la superficie de fractura pueden fusionarse. En validación, 19 de 45 fragmentos secundarios (42 %) quedaron absorbidos por el principal; con `completo` eran 17 de 45 (38 %). La receta de 384 px mejoró la calidad de los secundarios que sí separa (Dice de los correctos 0.76 → 0.82), pero no la tasa de absorción. Quitar CBAM (`sin_cbam_384`) la bajó a 7 de 45.
- **Dice bajo en secundarios:** en validación, 0.93 para los principales y 0.51 para los secundarios.
- **Fragmentos espurios:** 56 predichos sin correspondencia en el GT (precisión de fragmentos 0.60); un poco más que con `completo` (53).
- **Sacro:** es la región con menor F1 de clasificación por corte (0.969, frente a 0.992 y 0.994 de los coxales).
- **Error de la cabeza de distancia:** 5.1 mm de error medio, peor que los 3.5 mm del método geométrico sobre la máscara.
- **Datos escasos:** 100 casos con etiquetas, 70 para entrenar, y 15 de test: las cifras tienen mucha incertidumbre y un caso malo mueve el promedio. Una sola semilla.
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
