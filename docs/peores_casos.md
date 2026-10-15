# Los 5 peores casos del test

Modelo `completo` (`pesos/completo.pth`), evaluación en test (15 pacientes) a nivel de degradación 0.
Los casos son los 5 de menor Dice de fragmento según `peores_5` en `docs/resultados/completo/evaluacion_cuda.json`.
Las cifras por fragmento salen de `docs/resultados/completo/fragmentos_nivel0.json`.

> **Cambio respecto al reparto de tareas.** El reparto listaba 002, 085, 039, 038 y 018, pero esa lista
> era de la evaluación anterior. Con la reevaluación vigente entra el **084** y sale el **038**.
> Orden de peor a mejor: 085, 084, 002, 039, 018.

> **Nota.** Cada figura muestra un solo corte, el de más error, así que un fallo que ocurre en otros cortes
> no aparece (es el caso de 002 y 084). Lo escrito como *hipótesis* sale de los números; la línea
> "Lo que se ve en la figura" dice qué confirma o no la imagen.

## Resumen

| Caso | Dice | IoU | HD95 (mm) | Fragmentos GT | GT sin pareja | Predichos sin pareja | Píxel (mm) |
|---|---|---|---|---|---|---|---|
| 085 | 0.480 | 0.408 | 23.4 | 8 | 3 | 2 | 0.91 |
| 084 | 0.517 | 0.453 | 21.2 | 8 | 3 | 4 | 0.85 |
| 002 | 0.578 | 0.521 | 55.5 | 6 | 2 | 4 | 0.82 |
| 039 | 0.593 | 0.481 | 35.2 | 9 | 0 | 2 | 1.22 |
| 018 | 0.656 | 0.597 | 46.0 | 4 | 1 | 3 | 0.78 |

"GT sin pareja" cuenta los fragmentos del ground truth sin fragmento predicho o con Dice 0.
"Predichos sin pareja" son los fragmentos espurios (predichos sin correspondencia en el GT).
Entre los 5 casos suman 15 de los 53 espurios del test.

## Patrón común

- **Los fragmentos principales suelen salir bien** (Dice 0.91–0.96 en los tres principales de 002 y 084, y en SA-1 y RI-1
  de 085), y el problema está sobre todo en los secundarios. Las excepciones son LI-1 de 085 (0.58), RI-1 de 018 (0.70)
  y SA-1 y LI-1 de 039 (0.71 y 0.79). Que el error se concentre en los secundarios coincide con el diagnóstico en validación
  (`docs/resultados/diagnostico_val/resumen.json`): el 38 % de los secundarios queda absorbido por el principal.
- **Hay 9 fragmentos del GT sin pareja en 4 de los 5 casos** (085, 084, 002 y 018). Con los números no se distingue
  si un fragmento está *perdido* o *absorbido*; la figura sí lo muestra (se ve el contorno del GT sin contorno
  correspondiente en la predicción).
- **Aparecen fragmentos espurios** en los 5 casos, casi todos pequeños (1–6 mL, apenas sobre el umbral de 1 mL),
  pero en 002 y 018 hay espurios grandes de 18–23 mL.

## Caso por caso

### 085 · Dice 0.480 · el peor

- **Números:** de 8 fragmentos del GT, no tienen pareja SA-2, LI-3 y LI-4. El principal LI-1 llega solo a Dice 0.58
  (HD95 56 mm), y LI-2 y RI-2 a 0.75 y 0.61. Hay 2 espurios de 1.2 y 1.5 mL.
- **Hipótesis (por confirmar):** secundarios pequeños de LI absorbidos por el principal o perdidos. El LI-1 con HD95
  de 56 mm sugiere que además hay vóxeles de LI lejos del hueso principal.
- **Lo que se ve en la figura:** En el corte 95 (el de más error) los fragmentos pequeños y claros de la parte superior derecha del GT, entre ellos LI-4, parecen fundidos con el LI-1 en la predicción, que queda con el contorno irregular. Es coherente con un secundario absorbido por el principal. No hay fragmentos rojos en este corte.

### 084 · Dice 0.517

- **Números:** sin pareja LI-3, LI-4 y RI-2. Hay 4 espurios: tres en el sacro (1.1–1.6 mL) y uno en LI de 6.2 mL.
  El SA-2 se detecta pero con Dice 0.64 y HD95 de 47 mm.
- **Hipótesis (por confirmar):** el sacro se fragmenta en trozos pequeños que no existen en el GT, mientras que en LI
  y RI se pierden fragmentos reales. Junto con 002, es el caso con más espurios (4).
- **Lo que se ve en la figura:** En el corte 65 no hay rojo. La diferencia está en el fragmento delgado y curvo de la parte superior derecha (LI-2 del GT), que la predicción dibuja más grueso y con otra forma. Los espurios del sacro no se pueden ver aquí, porque este corte no pasa por el sacro.

### 002 · Dice 0.578 · HD95 55 mm

- **Números:** el GT tiene 6 fragmentos y el modelo predice LI-1…LI-4, aunque el GT solo tiene LI-1 en esa región.
  Los espurios son LI-2 (20 mL), LI-3 (2.8 mL), LI-4 (2.6 mL) y RI-3 (1.4 mL). SA-2 y RI-2 del GT no tienen pareja,
  y RI-3 del GT se empareja con un fragmento con Dice 0.67 y HD95 de 195 mm.
- **Hipótesis (por confirmar):** el coxal izquierdo se *parte* en varios fragmentos (uno de 20 mL) donde el GT tiene
  un solo hueso; el HD95 de 195 mm sugiere un fragmento lejano mal asignado.
- **Lo que se ve en la figura:** En el corte 221 el GT y la predicción se ven casi iguales (SA-1, SA-2, LI-1 y RI-1) y no hay rojo. La hipótesis del coxal izquierdo partido en varios fragmentos (uno de 20 mL) no se ve en este corte; queda sin verificar en la figura.

### 039 · Dice 0.593 · el atípico

- **Números:** es el único donde los 9 fragmentos del GT tienen pareja, pero varias son malas: SA-2 → Dice 0.15,
  SA-3 → 0.31 y LI-2 → 0.13. Es también el que tiene el mayor tamaño de píxel (1.22 mm, el máximo del dataset).
  Hay 2 espurios de 5.1 y 3.2 mL.
- **Hipótesis (por confirmar):** los fragmentos se detectan, pero sus bordes quedan mal delineados. El tamaño de
  píxel grande deja menos vóxeles por fragmento pequeño. El reparto menciona que el EDA lo marcó por tener el
  campo de visión recortado; eso no se puede verificar con estos JSON, solo mirando el volumen.
- **Lo que se ve en la figura:** En el corte 253 hay un fragmento rojo, sin pareja en el GT, junto a la unión entre el sacro y el coxal derecho. Los fragmentos del sacro (SA-1, SA-2, SA-3) están en la zona correcta pero con bordes distintos a los del GT, lo que coincide con una delineación pobre. Sobre el campo de visión recortado no se puede afirmar nada con esta imagen; hay que mirarlo en el volumen.

### 018 · Dice 0.656

- **Números:** el GT tiene 4 fragmentos (uno por región y un secundario en RI). El principal RI-1 llega a Dice 0.70, y el
  secundario RI-2 queda con Dice 0.00 y HD95 de 145 mm. Hay 3 espurios en RI: 23.2, 17.9 y 3.5 mL.
- **Hipótesis (por confirmar):** el coxal derecho se *parte* en trozos grandes (23 y 18 mL) donde el GT solo tiene
  dos fragmentos; el principal pierde volumen y el secundario real no se recupera.
- **Lo que se ve en la figura:** En el corte 279 el RI-2 del GT (un fragmento pequeño del coxal derecho) está cubierto en la predicción por un fragmento rojo más ancho, sin pareja. El modelo encontró algo en esa zona, pero no coincide lo bastante en 3D con RI-2, lo que concuerda con su Dice de 0.00 y con los espurios grandes de la tabla.

## Causas probables (hipótesis)

1. **Fragmento absorbido o perdido** (085, 084, 002, 018): es la causa dominante, coherente con el 38 % de
   absorbidos del diagnóstico de validación. La cabeza `rol` no separa el secundario del principal cuando están pegados.
2. **Fragmento partido** (002, 018): un hueso grande se divide en varios fragmentos predichos. El watershed sembrado
   con núcleos sin borde puede generar semillas de más cuando la cabeza `borde` marca contactos falsos.
3. **Delineación pobre** (039): la detección acierta pero los bordes no.

Con 15 pacientes de test, 5 casos son un tercio de la muestra: conviene presentarlos como ejemplos de los modos de
fallo, no como una estadística de su frecuencia.

## Cómo regenerar las figuras

```bash
python scripts/galeria_peores.py --ckpt pesos/completo.pth                 # los 5 de peores_5
python scripts/galeria_peores.py --ckpt pesos/completo.pth --casos 039     # uno solo
```

Escribe `docs/figuras/peor_<caso>.png` y `docs/figuras/peores_casos.json` (corte elegido y vóxeles en error).
En las figuras, el contorno blanco marca cada fragmento y el **rojo** un fragmento predicho sin pareja en el GT.
