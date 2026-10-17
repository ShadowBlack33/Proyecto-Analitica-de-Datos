# Ablaciones y línea base SAM

## 1. Qué es una ablación y cómo se hizo

Una ablación quita o cambia **un solo componente** del modelo y lo reentrena con todo lo demás igual, para medir cuánto aporta ese componente.

Cada configuración hereda de otra y cambia una sola línea:

| Configuración | Hereda de | Qué cambia |
| --- | --- | --- |
| `ablacion_sin_cbam.yaml` | `default.yaml` | Quita los bloques de atención CBAM |
| `ablacion_con_tl.yaml` | `default.yaml` | Inicializa el backbone con los pesos del Taller 3 (transfer learning) |
| `ablacion_sin_cbam_384.yaml` | `mejorado_384.yaml` | Quita los bloques de atención CBAM |

Todos los modelos usan la misma semilla (42), los mismos 15 pacientes de test, el mismo post-proceso (fragmento mínimo de 1 mL) y las mismas métricas.

**Regla de comparación:** cada modelo se compara contra el modelo con su misma receta de entrenamiento. Por eso hay dos bloques: la receta base a 256 px y la receta final a 384 px.

**Regla de selección:** el modelo final se elige por validación. El test se usa solo para reportar; si se usara para decidir, dejaría de ser una medición independiente.

**Tiempos de entrenamiento:** todos los modelos se entrenaron en una RTX 3060 Laptop de 6 GB. Los modelos base (`completo`, `sin_cbam` y `con_tl`) siguen el currículo completo de 55 épocas; los de 384 px son un ajuste fino de 20 épocas a partir de su modelo base (`mejorado_384` desde `completo` y `sin_cbam_384` desde `sin_cbam`). Los tiempos de los modelos base no se comparan con precisión, porque se entrenaron en momentos distintos con otras cargas en el equipo; sirven como orden de magnitud: entre 3 y 5 horas por modelo base y unas 3 horas por ajuste fino a 384 px.

## 2. Bloque A: receta base a 256 px (55 épocas)

Métricas de test sobre 15 pacientes, salvo las filas marcadas como validación.

| Métrica | `completo` (referencia, con CBAM) | `sin_cbam` | `con_tl` (backbone del Taller 3) |
| --- | --- | --- | --- |
| Parámetros | 6 791 354 | 6 750 198 | 6 791 354 |
| Dice por fragmento | **0.676** | 0.667 | 0.661 |
| IoU por fragmento | **0.597** | 0.590 | 0.583 |
| Precisión / recall de fragmentos | 0.602 / 0.879 | 0.648 / 0.890 | **0.678 / 0.901** |
| HD95 / ASSD (mm) | 21.5 / **4.57** | 23.2 / 5.52 | **20.3** / 5.14 |
| mAP@0.50 / mAP@[0.50:0.95] | **0.877** / 0.750 | 0.876 / **0.751** | 0.875 / 0.741 |
| IoU de caja | **0.885** | 0.882 | 0.882 |
| F1 de clasificación | **0.987** | 0.980 | **0.987** |
| Error de distancia, geométrico / cabeza (mm) | 5.06 / **6.05** | **4.84** / 6.49 | 5.40 / 6.53 |
| Dice por fragmento, validación | 0.695 | **0.736** | 0.658 |
| Dice de secundarios, validación | 0.479 | **0.557** | 0.408 |
| Tiempo de entrenamiento | 5.2 h | 3.2 h | 4.0 h |

## 3. Bloque B: receta final a 384 px (ajuste fino de 20 épocas)

Métricas de test sobre 15 pacientes, salvo las filas marcadas como validación.

| Métrica | `mejorado_384` (referencia, con CBAM) | `sin_cbam_384` (modelo final) |
| --- | --- | --- |
| Dice por fragmento | **0.719** | 0.700 |
| IoU por fragmento | **0.645** | 0.628 |
| Precisión / recall de fragmentos | 0.600 / 0.923 | 0.600 / 0.923 |
| HD95 / ASSD (mm) | 23.3 / **4.92** | **23.2** / 5.31 |
| mAP@0.50 / mAP@[0.50:0.95] | **0.907 / 0.776** | 0.906 / 0.767 |
| IoU de caja | **0.898** | 0.893 |
| F1 de clasificación | **0.985** | 0.981 |
| Error de distancia, geométrico / cabeza (mm) | **3.54 / 5.06** | 4.00 / 5.36 |
| Dice por fragmento, validación | 0.719 | **0.793** |
| Dice de secundarios, validación | 0.509 | **0.652** |
| Secundarios absorbidos, validación | 19 de 45 (42 %) | **7 de 45 (16 %)** |
| Latencia en GPU | 16–36 ms por corte | **13.5 ms por corte** |
| Tiempo de entrenamiento | 3.0 h | ~2.7 h (4.4 h medidas: 4 épocas fueron lentas por falta de RAM) |

## 4. Conclusiones

- **CBAM no aporta.** En las dos escalas, quitarlo mejora la validación (+0.041 a 256 px y +0.074 a 384 px) y casi no cambia el test (−0.009 y −0.019), una diferencia dentro de lo que varía el resultado entre conjuntos de 15 pacientes.
- **La mejora en validación está en el problema central del proyecto:** los fragmentos secundarios absorbidos por el principal bajan de 19 a 7 de 45, y el Dice de secundarios sube de 0.509 a 0.652.
- **Decisión:** el modelo final es `sin_cbam_384`. Se elige por validación, la misma regla del protocolo, y además es más simple (41 156 parámetros menos) y más rápido (13.5 ms por corte).
- **Transfer learning:** inicializar el backbone con los pesos del Taller 3 no mejora el Dice en test (0.661 contra 0.676), pero reduce los fragmentos espurios (precisión de 0.602 a 0.678). En validación queda por debajo de la referencia (0.658 contra 0.695), con 22 de 45 secundarios absorbidos por el principal. Es un cambio de perfil, no una mejora general, y por eso no se adoptó.
- **Limitación:** una sola semilla por modelo y 15 pacientes por conjunto. Por eso la decisión se basa en validación y se reportan ambos conjuntos.

## 5. Línea base SAM

### Protocolo

- **SAM:** ViT-B (`sam_vit_b_01ec64.pth`), sin ajuste fino (zero-shot). Recibe como prompt la caja que predice el modelo propio para cada región.
- **Modelo propio:** el modelo final `sin_cbam_384` (`configs/ablacion_sin_cbam_384.yaml`), que aporta las cajas y la segmentación de referencia.
- **Datos:** los 15 pacientes de test, evaluando 1 de cada 2 cortes.
- **Métrica:** Dice e IoU **por región ósea completa** (sacro, coxal izquierdo y coxal derecho), no por fragmento. Por eso estas cifras no se comparan con el Dice de fragmento de las tablas anteriores.
- **Hardware:** RTX 3050 Laptop (4 GB). Cada corrida tarda cerca de una hora.

Comando:

```
python -u scripts/baseline_sam.py --config configs/ablacion_sin_cbam_384.yaml --ckpt pesos/sin_cbam_384.pth --sam checkpoints/sam_vit_b_01ec64.pth
```

### Resultados por región

| Región | SAM Dice | SAM IoU | Propio Dice | Propio IoU | Diferencia Dice |
| --- | --- | --- | --- | --- | --- |
| Sacro (SA) | 0.879 | 0.786 | 0.963 | 0.929 | +0.084 |
| Coxal izquierdo (LI) | 0.936 | 0.881 | 0.971 | 0.945 | +0.036 |
| Coxal derecho (RI) | 0.938 | 0.884 | 0.974 | 0.949 | +0.036 |
| **Promedio** | **0.918** | **0.851** | **0.969** | **0.941** | **+0.051** |

### Resultados por paciente (Dice)

| Caso | SA SAM | SA propio | LI SAM | LI propio | RI SAM | RI propio |
| --- | --- | --- | --- | --- | --- | --- |
| 002 | 0.889 | 0.974 | 0.920 | 0.978 | 0.943 | 0.976 |
| 004 | 0.877 | 0.956 | 0.918 | 0.974 | 0.959 | 0.975 |
| 005 | 0.928 | 0.968 | 0.952 | 0.971 | 0.969 | 0.971 |
| 015 | 0.822 | 0.949 | 0.922 | 0.969 | 0.904 | 0.975 |
| 017 | 0.860 | 0.969 | 0.936 | 0.970 | 0.936 | 0.976 |
| 018 | 0.867 | 0.968 | 0.963 | 0.979 | 0.886 | 0.971 |
| 023 | 0.898 | 0.975 | 0.950 | 0.982 | 0.952 | 0.980 |
| 028 | 0.881 | 0.962 | 0.944 | 0.973 | 0.922 | 0.970 |
| 038 | 0.907 | 0.960 | 0.974 | 0.976 | 0.964 | 0.974 |
| 039 | 0.852 | 0.956 | 0.929 | 0.964 | 0.913 | 0.972 |
| 043 | 0.898 | 0.974 | 0.952 | 0.974 | 0.956 | 0.976 |
| 048 | 0.864 | 0.960 | 0.830 | 0.960 | 0.863 | 0.966 |
| 076 | 0.913 | 0.968 | 0.938 | 0.971 | 0.972 | 0.977 |
| 084 | 0.800 | 0.936 | 0.953 | 0.964 | 0.959 | 0.975 |
| 085 | 0.930 | 0.971 | 0.955 | 0.966 | 0.967 | 0.976 |

Los promedios por región son la media de los 15 pacientes; el promedio general coincide con el resumen que imprime el script.

La comparación se corrió también con las cajas de `mejorado_384` (el modelo con CBAM) y dio prácticamente lo mismo: SAM 0.919 contra 0.969 del modelo propio. Era lo esperable, porque los dos modelos tienen una detección equivalente (mAP@0.50 de 0.907 y 0.906).

### Lectura

- **Qué hace SAM:** segmenta el contorno de una región a partir de una caja.
- **Qué no puede hacer:** separar fragmentos, decir cuál es el principal ni medir distancias, que son el objetivo del proyecto.
- **Resultado:** el modelo propio supera a SAM en las 45 combinaciones de paciente y región, con Dice promedio de 0.969 contra 0.918. La brecha más grande está en el sacro (0.879 contra 0.963), el hueso de forma más irregular; el peor caso es el 084 (0.800 contra 0.936). En los coxales SAM queda más cerca (alrededor de 0.94 contra 0.97), porque son huesos grandes y bien delimitados, y la caja del modelo propio ya resuelve buena parte del problema.
- **Conclusión:** un modelo generalista sin entrenamiento en CT rinde razonablemente cuando recibe una buena caja, pero no alcanza al modelo especializado y no resuelve la tarea completa. Eso justifica el modelo propio.
