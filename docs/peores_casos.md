# Los 5 peores casos del test

Modelo final `sin_cbam_384` (`pesos/sin_cbam_384.pth`, 384 px, sin CBAM), evaluación en test (15 pacientes) a nivel de degradación 0.
Los casos son los 5 de menor Dice de fragmento según `peores_5` en `docs/resultados/sin_cbam_384/evaluacion_cuda.json`.
Las cifras por fragmento salen de `docs/resultados/sin_cbam_384/fragmentos_nivel0.json`.

> **Cambio respecto al reparto de tareas.** El reparto listaba 002, 085, 039, 038 y 018, de una evaluación anterior.
> Con el modelo final los cinco peores son **039, 085, 002, 084 y 015**. Con `mejorado_384` el quinto era el 018;
> ahora el 018 queda sexto (0.667) y entra el 015.

## Resumen

| Caso | Dice | IoU | HD95 (mm) | Fragmentos GT | GT sin pareja | Predichos sin pareja | Dice con `mejorado_384` | Dice con `completo` |
|---|---|---|---|---|---|---|---|---|
| 039 | 0.497 | 0.408 | 57.1 | 9 | 0 | 5 | 0.545 | 0.593 |
| 085 | 0.546 | 0.477 | 50.3 | 8 | 1 | 5 | 0.568 | 0.480 |
| 002 | 0.612 | 0.547 | 11.9 | 6 | 1 | 2 | 0.617 | 0.578 |
| 084 | 0.636 | 0.573 | 17.3 | 8 | 2 | 5 | 0.626 | 0.517 |
| 015 | 0.659 | 0.576 | 16.7 | 6 | 1 | 7 | 0.707 | 0.658 |

"GT sin pareja" cuenta los fragmentos del ground truth sin fragmento predicho o con Dice 0.
"Predichos sin pareja" son los fragmentos espurios (predichos sin correspondencia en el GT).
Entre los 5 casos suman 24 de los 56 espurios del test; el 015 es el caso con más espurios de todo el test (7).
Frente a `mejorado_384`, solo el 084 mejora (0.626 → 0.636); el 002 queda igual y 039, 085 y 015 empeoran.
Esto no contradice la elección del modelo: se eligió por validación, y en test la diferencia global es pequeña (0.719 → 0.700).

## Patrón común

- **Los fragmentos principales suelen salir bien** (Dice 0.87–0.98 en la mayoría) y el problema está sobre todo en los
  secundarios. Las excepciones son el SA-1 y el LI-1 de 039 (0.70 y 0.79) y el LI-1 de 085 (0.23), donde el modelo
  confunde cuál de los fragmentos grandes es el principal.
- **Secundarios absorbidos o fundidos:** en 039 los secundarios del sacro (SA-2 y SA-3) quedan casi sin recuperar
  (0.05 y 0.08), y en 002 dos fragmentos reales del coxal derecho salen como uno solo. En validación, el modelo final
  absorbe 7 de 45 secundarios (16 %), frente a 19 de 45 con `mejorado_384` (`diagnostico_val/resumen.json`): el problema
  bajó, pero sigue apareciendo en los peores casos.
- **Fragmentos partidos:** en 002 una parte del coxal izquierdo sale como un fragmento separado de 20 mL, y en 015 el sacro
  se divide en varios fragmentos (un espurio de 16 mL y cinco pequeños).
- **Fragmentos perdidos:** 5 fragmentos del GT sin pareja en total (SA-2 de 085, RI-2 de 002, LI-3 y LI-4 de 084, LI-3 de 015).

## Caso por caso

### 039 · Dice 0.497 · el peor

- **Números:** los 9 fragmentos del GT tienen pareja, pero varias son malas: SA-2 → Dice 0.05, SA-3 → 0.08, LI-2 → 0.10 y
  RI-2 → 0.20. Los principales SA-1 y LI-1 llegan a 0.70 y 0.79 (HD95 de 72 mm en LI-1); RI-1 sale bien (0.95). Hay 5
  espurios pequeños (1.0 a 3.8 mL). Es el único caso con tamaño de píxel de 1.22 mm, el máximo del dataset.
- **Lo que se ve en la figura (corte 241):** el GT divide el sacro en tres piezas (SA-1 al centro, SA-2 y SA-3 a los
  lados, en tonos más claros). En la predicción todo el sacro sale con un solo color, el del principal: los dos
  secundarios quedan absorbidos. Aparece además un fragmento rojo pequeño, sin pareja, en el borde inferior del sacro.
  Los dos coxales se ven bien.
- **Hipótesis (por confirmar):** los secundarios del sacro están pegados al principal y la red no ve la línea de fractura;
  el tamaño de píxel grande deja menos vóxeles para separarlos. Empeora frente a los dos modelos anteriores.

### 085 · Dice 0.546

- **Números:** de 8 fragmentos del GT, el SA-2 no tiene pareja. En el coxal izquierdo las parejas salen cruzadas: el
  principal del GT (LI-1) se empareja con el fragmento que el modelo llama LI-3 (Dice 0.23) y el LI-3 del GT con el LI-1
  predicho (0.45); el LI-4 queda en 0.04. LI-2, RI-1, RI-2 y SA-1 salen bien (0.84 a 0.96). Hay 5 espurios, uno grande en
  el coxal derecho (14.2 mL) y cuatro de 1.2 a 2.6 mL.
- **Lo que se ve en la figura (corte 250):** en el GT, el coxal izquierdo tiene dos piezas: el principal LI-1 (la parte
  baja, azul oscuro) y el secundario LI-3 (la punta superior del ala, azul claro), separados por la línea de fractura.
  En la predicción las dos piezas salen con el color de LI-3: el fragmento que el modelo empareja con el secundario cubre
  también la zona del principal. El sacro y el coxal derecho se ven bien, y en este corte no hay fragmentos rojos.
- **Hipótesis (por confirmar):** el modelo no distingue cuál de los fragmentos grandes del coxal izquierdo es el
  principal, una decisión que depende de la forma en 3D.

### 002 · Dice 0.612 · el de menor HD95 (11.9 mm)

- **Números:** de 6 fragmentos del GT, el RI-2 no tiene pareja y el RI-3 llega a 0.66; el SA-2 queda en 0.17. Los
  principales salen bien (0.92 a 0.98). Hay 2 espurios: LI-2 (20.1 mL) y RI-3 (1.5 mL). El HD95 bajó mucho frente a
  `mejorado_384` (35.8 → 11.9 mm).
- **Lo que se ve en la figura (corte 78, a la altura del pubis):** en el coxal derecho, el GT separa dos fragmentos (RI-2
  y RI-3) con una línea de fractura; en la predicción salen como uno solo. En el coxal izquierdo, la rama superior, que en
  el GT es parte del principal LI-1, aparece como un fragmento rojo sin pareja: es el espurio de 20 mL. Las partes
  inferiores de los dos coxales salen bien.
- **Hipótesis (por confirmar):** dos errores opuestos en el mismo corte: se *funden* dos fragmentos reales y se *parte*
  un principal.

### 084 · Dice 0.636

- **Números:** sin pareja LI-3 y LI-4. El SA-2 se detecta con Dice 0.57 pero HD95 de 73 mm; el LI-2 llega a 0.72. Los
  principales salen bien (0.91 a 0.97). Hay 5 espurios pequeños: tres en el sacro (1.6 a 3.8 mL) y dos en LI (1.4 y 3.0 mL).
- **Lo que se ve en la figura (corte 169):** el SA-2 del GT es el bloque más claro del sacro junto a la articulación con
  el coxal izquierdo. En la predicción esa zona aparece en parte con el color de SA-2 y con un fragmento rojo, sin pareja,
  entre el sacro y el coxal izquierdo. El resto del sacro y los dos coxales se ven bien.
- **Hipótesis (por confirmar):** en la zona de la articulación sacroilíaca el modelo genera piezas que no existen en el GT,
  y en el coxal izquierdo se pierden dos fragmentos reales pequeños.

### 015 · Dice 0.659 · nuevo en la lista

- **Números:** de 6 fragmentos del GT, el LI-3 no tiene pareja. El SA-2 del GT se empareja con el SA-3 predicho (0.45) y
  el LI-2 llega a 0.74. Los principales salen bien (0.87 a 0.97). Es el caso con más espurios del test: 7, seis de ellos
  en el sacro (uno de 16.2 mL y cinco de 1.1 a 2.4 mL) y uno en RI (3.5 mL).
- **Lo que se ve en la figura (corte 157):** el GT divide el sacro en dos piezas: el principal SA-1 y el SA-2, la pieza más
  clara junto al coxal izquierdo. En la predicción, casi todo el SA-2 sale en rojo, como un fragmento sin pareja, y solo
  una tira pequeña arriba conserva el color de SA-2. El principal y los dos coxales se ven bien.
- **Hipótesis (por confirmar):** el modelo detecta el secundario del sacro pero lo parte: la pieza grande queda como
  espurio (probablemente el de 16.2 mL) y la pequeña es la que se empareja con el SA-2 del GT, por eso su Dice es solo 0.45.

## Causas probables (hipótesis)

1. **Fragmento absorbido o fundido** (039, 002): la cabeza `rol` no separa el secundario del principal, o dos secundarios
   entre sí, cuando están pegados. Quitar CBAM redujo la absorción en validación, pero no la eliminó.
2. **Fragmento partido** (002, 015 y, en la articulación, 084): un hueso se divide en varios fragmentos predichos. El
   watershed sembrado con núcleos sin borde puede generar semillas de más cuando la cabeza `borde` marca contactos falsos.
3. **Principal mal elegido** (085, 039): cuando hay varios fragmentos grandes, decidir cuál es el principal depende de la
   forma en 3D, que una red corte a corte ve solo en parte.

Cada figura muestra un solo corte, el de más error, así que un fallo que ocurre en otros cortes no aparece. Con 15
pacientes de test, 5 casos son un tercio de la muestra: conviene presentarlos como ejemplos de los modos de fallo, no como
una estadística de su frecuencia.

## Cómo regenerar las figuras

```bash
python scripts/galeria_peores.py --config configs/ablacion_sin_cbam_384.yaml --ckpt pesos/sin_cbam_384.pth --evaluacion docs/resultados/sin_cbam_384/evaluacion_cuda.json
python scripts/galeria_peores.py --config configs/ablacion_sin_cbam_384.yaml --ckpt pesos/sin_cbam_384.pth --evaluacion docs/resultados/sin_cbam_384/evaluacion_cuda.json --casos 039   # uno solo
```

Escribe `docs/figuras/peor_<caso>.png` y `docs/figuras/peores_casos.json` (corte elegido y vóxeles en error).
En las figuras, el contorno blanco marca cada fragmento y el **rojo** un fragmento predicho sin pareja en el GT.
