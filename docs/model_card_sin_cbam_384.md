# Model Card — PENGWIN `sin_cbam_384` (modelo final, 384 px, sin CBAM)

> **Advertencia:** este sistema tiene fines exclusivamente académicos. No es un dispositivo
> médico, no ha sido validado clínicamente y no debe usarse para apoyar decisiones
> quirúrgicas reales.

Pesos: `pesos/sin_cbam_384.pth`. Configuración: `configs/ablacion_sin_cbam_384.yaml` (hereda de `configs/mejorado_384.yaml` y solo quita CBAM). Las cifras salen de
`docs/resultados/sin_cbam_384/evaluacion_cuda.json` (test, nivel de degradación 0, salvo que se indique otra cosa) y, las de validación, de `docs/resultados/sin_cbam_384/diagnostico_val/resumen.json`.

## Arquitectura
- FundidoraPC extendida de 5 bloques (32 → 512 canales) **sin CBAM** (`cbam_en_bloques: []`), con las cuatro cabezas: clasificación anatómica, detección anchor-free con NMS propio (grid 24×24 a 384 px), segmentación con decoder tipo U-Net (región, rol principal/secundario y borde de contacto) y regresión densa de distancia.
- Parámetros: 6 750 198 (41 156 menos que las variantes con CBAM).
- Transfer learning: **no** desde otro dataset. Continúa desde los pesos de `sin_cbam` (receta base a 256 px sin CBAM, `pesos/sin_cbam.pth`).
- Receta (heredada de `mejorado_384`): entrada de **384 px**, peso **×3** a la clase "secundario", rotación de hasta **35°** en el 80 % de las muestras y borde de contacto de grosor 2. Ajuste fino de 20 épocas (segmentación 12 + conjunta 8), batch 12.
- Única diferencia con `mejorado_384`: no tiene CBAM. Por eso es la comparación directa del efecto de CBAM a 384 px.

## Datos de entrenamiento
- Dataset: PENGWIN Task 1 (CT), Zenodo 10.5281/zenodo.10927452. Se usan los 100 casos con etiquetas.
- Casos entrenamiento / validación / test: 70 / 15 / 15, repartidos **por paciente** y estratificados por número de fragmentos secundarios. Semilla 42.
- Splits: `data/splits.json`, fijado con SHA-256 que termina en `99267ddb2310271e`.
- Casos de test evaluados: 15 (002, 004, 005, 015, 017, 018, 023, 028, 038, 039, 043, 048, 076, 084, 085).
- Preprocesamiento: ventana de hueso −200 a 1200 HU, corte a 384×384 con los 2 cortes vecinos como canales (2.5D). La orientación se corrige desde la matriz de direcciones del header y el espaciado se lee del header, así que toda distancia sale en mm.

## Por qué es el modelo final
Se eligió por **validación**: Dice de fragmento 0.793 frente a 0.719 de `mejorado_384`, y los secundarios absorbidos por el principal bajan de 19 a 7 de 45. En test queda dentro del ruido (0.700 frente a 0.719, misma precisión y recall). Es además más simple y más rápido. Una sola semilla por modelo.

## Métricas obtenidas
Test, 15 pacientes, nivel de degradación 0.

| Tarea | Métrica | Objetivo | Obtenido | ¿Cumple? |
|---|---|---|---|---|
| Clasificación | F1-score | >= 0.85 | 0.981 | Sí |
| Clasificación | AUC | >= 0.85 | 0.999 | Sí |
| Detección | IoU promedio | >= 0.65 | 0.893 | Sí |
| Detección | mAP@0.50 | >= 0.65 | 0.906 | Sí |
| Detección | mAP@[0.50:0.95] | >= 0.40 | 0.767 | Sí |
| Segmentación de fragmentos | Dice | >= 0.85 | 0.700 | **No** |
| Segmentación de fragmentos | IoU | >= 0.70 | 0.628 | **No** |
| Segmentación (extra) | HD95 | — | 23.2 mm | — |
| Segmentación (extra) | ASSD | — | 5.31 mm | — |
| Medición de distancia | error medio vs. GT, método geométrico | — | 4.00 mm | — |
| Medición de distancia | error medio vs. GT, cabeza de regresión | — | 5.36 mm | — |

Detalle de fragmentos:
- 91 fragmentos en el ground truth y 140 predichos (precisión 0.60, recall 0.92).
- Fragmentos espurios (predichos sin correspondencia en el GT): 56, con volumen mediano de 2.3 mL.
- Fragmentos secundarios del GT con pareja: 25 de 46.

F1 por región según el JSON de evaluación: SA 0.974, LI 0.981, RI 0.990. La matriz de confusión de `scripts/matriz_confusion.py` (`docs/figuras/matriz_confusion.png`) da SA 0.974, LI 0.994 y RI 0.996, con un F1 promedio de 0.988 frente a 0.981 del JSON. La diferencia viene de los conjuntos de cortes: la matriz usa los 4005 cortes de la caché (rango con hueso ± 15 cortes) y `evaluar.py` los 4821 cortes de los volúmenes completos. Los 816 cortes extra están fuera de la pelvis y solo pueden aportar falsos positivos. El sacro casi coincide y la diferencia está en los coxales, coherente con cortes inferiores que incluyen los fémures. La diferencia (0.0066) supera la tolerancia de 0.005 del script, por eso imprime "REVISAR".

Por píxel (matriz 4×4, submuestreada ×4): el hueso bien clasificado va de 96.5 % (SA) a 97.7 % (RI). El error es casi todo hueso predicho como fondo (2.3 a 3.4 %); la confusión entre regiones óseas no pasa de 0.1 %.

Clasificación y detección se cumplen con margen; la segmentación de fragmentos no alcanza sus objetivos.

### Robustez a imágenes de baja calidad
| Nivel de degradación | F1 | mAP@0.50 | Dice de fragmento | Precisión de fragmentos | Error de distancia (geométrico, mm) |
|---|---|---|---|---|---|
| 0 | 0.981 | 0.906 | 0.700 | 0.60 | 4.00 |
| 1 | 0.979 | 0.906 | 0.694 | 0.61 | 3.81 |
| 2 | 0.983 | 0.878 | 0.669 | 0.56 | 3.99 |
| 3 | 0.978 | 0.876 | 0.614 | 0.44 | 5.80 |

El nivel 0 es la imagen original y el 3 la peor degradación (`NIVELES_DEGRADACION` en `src/data/aumentos.py`). La clasificación casi no cambia; la detección baja poco, y lo que más se degrada es la separación de fragmentos: en el nivel 3 los espurios suben de 56 a 105.

Dice de fragmento según la calidad real de la imagen (terciles de nitidez, 5 casos por tercil): alta 0.758, media 0.689, baja 0.702. Con tan pocos casos no se puede afirmar una tendencia.

## Latencia
- GPU: el README reporta unos 13.5 ms por corte en una RTX 3060 Laptop. En el JSON de evaluación, la media de la corrida de nivel 0 es 20.5 ms por corte; la medición depende del estado de la GPU.
- CPU: 226.0, 222.8 y 226.6 ms por corte en los casos 005, 023 y 048 (promedio ≈ 225 ms), medidos con `python scripts/inferir.py --config configs/ablacion_sin_cbam_384.yaml --ckpt pesos/sin_cbam_384.pth --ct data/raw/images/<caso>.mha --device cpu`. Equipo: AMD Ryzen 7 5800H (8 núcleos), 31.3 GB de RAM. Para comparar, `completo` (256 px) da ≈ 103 ms y `mejorado_384` ≈ 240 ms en el mismo equipo.

## Alcances
- Detectar y segmentar el sacro y los dos coxales en tomografías de pelvis, separar sus fragmentos y estimar en milímetros la distancia de cada fragmento a su principal.
- Es el modelo que usa el dashboard (MIP del volumen, inferencia corte a corte y reconstrucción 3D), como demostración académica.

## Limitaciones conocidas
- **Fragmentos en contacto:** los que se tocan en la superficie de fractura pueden fusionarse o partirse por errores de segmentación (separación con watershed en `src/postprocess/`). En validación, 7 de 45 secundarios (16 %) quedan absorbidos por el principal, y 2 más se fusionan entre sí. Es la limitación principal: decidir en 3D cuál es el fragmento principal a partir de una red que mira corte a corte. En los 5 peores casos de test se ven secundarios absorbidos y fragmentos partidos (ver `docs/peores_casos.md`).
- **Dice bajo en secundarios:** en validación, 0.93 para los principales y 0.65 para los secundarios.
- **Fragmentos espurios:** 56 predichos sin correspondencia en el GT (precisión de fragmentos 0.60). La mediana es 2.3 mL, pero hay algunos grandes: los mayores, de 20 a 25 mL, están en los casos 018, 038 y 002.
- **Sacro:** es la región con menor F1 de clasificación por corte (0.974) y con menor acierto por píxel (96.5 %).
- **Error de la cabeza de distancia:** 5.4 mm de error medio, peor que los 4.0 mm del método geométrico sobre la máscara.
- **Más lento en CPU:** unas 2.2 veces el modelo base de 256 px.
- **Datos escasos:** 100 casos con etiquetas, 70 para entrenar, y 15 de test. Las cifras tienen mucha incertidumbre y un caso malo mueve el promedio. Una sola semilla.
- **Sin validación clínica** ni evaluación en otros centros o escáneres. No usar para decisiones médicas.

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
