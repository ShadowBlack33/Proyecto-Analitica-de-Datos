# Ablación y baseline SAM

## 1. Baseline SAM contra el modelo propio

### Protocolo

- **Modelo propio:** `pesos/mejorado_384.pth` con `configs/mejorado_384.yaml`.
- **SAM:** ViT-B (`sam_vit_b_01ec64.pth`), sin ajuste fino (zero-shot). Recibe como prompt la caja que predice el modelo propio para cada región.
- **Datos:** los 15 pacientes de test, evaluando 1 de cada 2 cortes (`--cada 2`, valor por defecto).
- **Métrica:** Dice e IoU **por región ósea completa** (sacro, coxal izquierdo y coxal derecho), no por fragmento. Por eso estas cifras no se comparan con el Dice de fragmento del modelo.
- **Hardware:** RTX 3050 Laptop (4 GB), PyTorch 2.11 con CUDA 12.8. Duración aproximada: 1 hora.

Comando:

```
python -u scripts/baseline_sam.py --config configs/mejorado_384.yaml --ckpt pesos/mejorado_384.pth --sam checkpoints/sam_vit_b_01ec64.pth
```

### Resultados por región

| Región | SAM Dice | SAM IoU | Propio Dice | Propio IoU | Diferencia Dice |
| --- | --- | --- | --- | --- | --- |
| Sacro (SA) | 0.877 | 0.783 | 0.963 | 0.928 | +0.086 |
| Coxal izquierdo (LI) | 0.940 | 0.888 | 0.972 | 0.945 | +0.032 |
| Coxal derecho (RI) | 0.940 | 0.888 | 0.973 | 0.948 | +0.033 |
| **Promedio** | **0.919** | **0.853** | **0.969** | **0.940** | **+0.050** |

Los promedios por región son la media de los 15 pacientes. El promedio general coincide con el resumen que imprime el script.

### Resultados por paciente (Dice)

| Caso | SA SAM | SA propio | LI SAM | LI propio | RI SAM | RI propio |
| --- | --- | --- | --- | --- | --- | --- |
| 002 | 0.894 | 0.973 | 0.920 | 0.977 | 0.945 | 0.975 |
| 004 | 0.877 | 0.956 | 0.912 | 0.973 | 0.960 | 0.975 |
| 005 | 0.928 | 0.966 | 0.956 | 0.971 | 0.970 | 0.971 |
| 015 | 0.811 | 0.949 | 0.921 | 0.971 | 0.910 | 0.974 |
| 017 | 0.866 | 0.973 | 0.937 | 0.970 | 0.938 | 0.976 |
| 018 | 0.870 | 0.966 | 0.965 | 0.979 | 0.895 | 0.973 |
| 023 | 0.898 | 0.977 | 0.952 | 0.983 | 0.955 | 0.980 |
| 028 | 0.884 | 0.958 | 0.926 | 0.974 | 0.928 | 0.970 |
| 038 | 0.911 | 0.959 | 0.974 | 0.976 | 0.964 | 0.975 |
| 039 | 0.857 | 0.958 | 0.929 | 0.960 | 0.913 | 0.973 |
| 043 | 0.904 | 0.973 | 0.953 | 0.974 | 0.958 | 0.976 |
| 048 | 0.861 | 0.958 | 0.901 | 0.966 | 0.865 | 0.955 |
| 076 | 0.910 | 0.968 | 0.949 | 0.971 | 0.972 | 0.977 |
| 084 | 0.738 | 0.936 | 0.954 | 0.965 | 0.960 | 0.975 |
| 085 | 0.939 | 0.969 | 0.955 | 0.969 | 0.967 | 0.976 |

### Lectura

- **El modelo propio supera a SAM en las 45 combinaciones de paciente y región**, con una diferencia media de 0.05 en Dice.
- **La brecha está en el sacro** (0.877 contra 0.963). Es el hueso de forma más irregular y el que más se superpone con estructuras vecinas dentro de su caja, así que un modelo generalista no sabe qué parte del contenido de la caja es hueso. El peor caso es el 084 (0.738 contra 0.936).
- **En los coxales SAM queda cerca** (0.940 contra 0.97) y en 12 de las 30 combinaciones la diferencia es menor a 0.02. Son huesos grandes y bien delimitados, y la caja del modelo propio ya resuelve casi todo el problema.
- **SAM depende de nuestras cajas.** Su desempeño no es independiente: hereda la calidad de la detección propia (IoU de cajas 0.885 en test). Con cajas del ground truth (`--caja_gt`) se mediría su techo.
- **Conclusión:** un modelo generalista sin entrenamiento en CT rinde razonablemente en regiones grandes cuando se le da una buena caja, pero no alcanza al modelo especializado, sobre todo en la región anatómicamente más difícil. Además, SAM solo segmenta la región: no separa fragmentos ni mide distancias, que son el objetivo del proyecto.

## 2. Ablación sin CBAM

Pendiente: se completará con los resultados de la corrida sin CBAM.

| Modelo | F1 | AUC | IoU cajas | mAP@0.50 | mAP@[0.50:0.95] | Dice fragmento | IoU fragmento | HD95 | Tiempo de entrenamiento |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Completo | | | | | | | | | |
| Sin CBAM | | | | | | | | | |
| Con transfer learning | | | | | | | | | |
