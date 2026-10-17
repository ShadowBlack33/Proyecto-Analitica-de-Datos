# Registro de uso de IA generativa

Este archivo documenta cada uso de IA generativa durante el proyecto, como lo exige la
sección 7 del enunciado. Una entrada por sesion/decision relevante — no hace falta registrar
cada mensaje suelto, pero si toda decision de arquitectura, codigo generado, o correccion
importante que haya surgido con apoyo de IA.

Formato de cada entrada:

## [Fecha] — [Tema corto]

**Modelo usado:** (ej. Claude Sonnet 4.6, ChatGPT, etc.)

**Prompt exacto (o resumen fiel si fue muy largo):**
> ...

**Que se obtuvo:**
- Codigo / decision / explicacion que vino de la IA.

**Que se modifico o verifico manualmente:**
- Que cambiaron ustedes despues, y por que.

**Analisis critico:**
- Que acerto la IA, que se equivoco o no aplicaba directamente, y como lo detectaron.

---

<!-- Agregar entradas nuevas abajo de esta linea -->

## 2026-09-23 — Primera version del codigo base (sin datos reales)

**Modelo usado:** Claude (Anthropic), Opus 5.5, en claude.ai.

**Prompt (resumen fiel):**
> "No te puedo pasar los datos porque son muy pesados, ya actualizamos la bitacora con cosas
> de este proyecto, puedes adelantar lo que mas se pueda del codigo por favor"
> (con el contexto previo: enunciado completo, bitacora de decisiones, hardware del equipo).

**Que se obtuvo:**
- Estructura completa de `src/` (datos, modelo, perdidas, post-proceso, metricas, inferencia,
  dashboard) y `scripts/` (preparar datos, EDA, overfit test, calibrar lambdas, entrenar,
  evaluar, baseline SAM, perfil de memoria, inferir, generador de CT sinteticos).
- Pruebas de control en `tests/` y guia del codigo en `docs/guia_codigo.md`.
- Correccion de un bug en `.gitignore` (`data/` tambien ignoraba `src/data/`).

**Que se verifico (por la IA, sobre CT sinteticos, en CPU):**
- `tests/test_basicos.py` pasa completo, incluido NMS propio == `torchvision.ops.nms`.
- Post-proceso alimentado con el GT reconstruye exactamente los fragmentos (Dice 1.0) y
  las distancias en mm.
- Overfit test con las 4 cabezas: la perdida cae y el IoU de cajas sobre el lote llega a ~0.87.
- `train.py` completo (4 etapas) y `evaluar.py` corren de punta a punta.
- Dashboard (`app.py`) ejecutado en modo headless con `streamlit.testing.AppTest`: las 4
  pestanas, los 3 visualizadores, el slider de cortes y la latencia funcionan sin excepciones.
- Evaluacion de un checkpoint entrenado solo hasta la etapa de segmentacion, sobre CT
  sinteticos (faciles, NO comparables con PENGWIN real): nivel 0 -> F1 1.00, mAP@0.50 0.95,
  Dice fragmento 0.75; nivel 3 de degradacion -> mAP@0.50 0.91, Dice fragmento 0.46.
  Sirve solo para confirmar que las metricas y el analisis de robustez reaccionan bien.

**Que NO se ha verificado (pendiente del equipo):**
- Nada se ha corrido con los datos reales de PENGWIN ni en GPU.
- La convencion de etiquetas (1/11/21 = principal) se tomo de la literatura del challenge;
  confirmarla con `scripts/eda.py`.
- La interpretacion de la "cabeza de regresion" como distancia densa esta pendiente de
  confirmar con el profesor.
- Los lambdas del config son provisionales hasta correr `calibrar_lambdas.py` con datos reales.
- `baseline_sam.py` no se pudo ejecutar en el entorno de la IA (requiere descargar los pesos de SAM).
- El tunel de Cloudflare no se probo; el dashboard si.

**Que se modifico manualmente:** corri todo con los datos reales en mi GPU y reemplace
los valores provisionales: ventana de HU segun el EDA y lambdas calibrados, redondeados por mi.

**Analisis critico:** la IA dio la estructura, pero solo estaba probada con datos
sinteticos. Decidi no aceptar ningun resultado que no saliera de una corrida con datos reales.

## Sesion — eliminacion de scikit-learn (bloqueo de Windows)

**Contexto:** en el PC de entrenamiento, Windows (Smart App Control / Application Control)
bloqueo el DLL `scipy.stats._rcont`. scikit-learn importa `scipy.stats` al cargar, asi que
`train.py` fallaba en la primera validacion.

**Que se obtuvo de la IA:** F1, AUC-ROC (formula de Mann-Whitney con empates promediados)
y matriz de confusion implementadas con NumPy en `src/eval/metricas_seg_cls.py`; se quito
scikit-learn de `requirements.txt`; script `scripts/verificar_entorno.py` para diagnosticar.

**Verificacion:** prueba `test_metricas_clasificacion_igual_a_sklearn` (resultados identicos
a sklearn, incluidos empates y clases vacias). Se simulo el bloqueo (import de `scipy.stats`
y `sklearn` fallando) y se corrieron pruebas, entrenamiento, evaluacion, inferencia y
dashboard: todo funciona y las metricas coinciden con la version anterior.

**Analisis critico:** el error lo encontre al entrenar en mi PC y le pase el traceback a la
IA. Le pedi que la metrica propia se comparara contra sklearn antes de aceptarla.

## Sesion — configuraciones de ablacion y transfer learning desde el Taller 3

**Que se obtuvo de la IA:** herencia de configuracion (`hereda:` en el YAML) para que cada
ablacion difiera de `default.yaml` en un solo factor; `configs/ablacion_sin_cbam.yaml` y
`configs/ablacion_con_tl.yaml`; carga de pesos del backbone reescrita para emparejar por capa
(conv con conv, BatchNorm con BatchNorm) y absorber el bias de la conv en la media del
BatchNorm; al reanudar a mitad de etapa se conserva el mejor puntaje ya alcanzado.

**Error detectado por la propia prueba:** la version anterior de la carga emparejaba tensores
en orden; si las convoluciones del Taller 3 tienen bias, el emparejamiento se desalineaba y
solo copiaba 5 tensores. Se corrigio y se agrego una prueba de equivalencia numerica.

**Verificacion:** 14 pruebas pasan; ambas ablaciones entrenan en datos sinteticos; la salida
del bloque copiado coincide con la original (diferencia maxima 5e-7).

**Analisis critico:** pedi que cada ablacion cambiara un solo factor para poder compararlas.
El transfer learning no mejoro el Dice y decidimos reportarlo igual.

## Sesion — cargador con varios procesos y segundo bloqueo de Windows

**Contexto:** (1) con `num_workers: 4` la ablacion con transfer learning fallo en la epoca 11
(`OSError 22`): en Windows cada epoca recreaba los procesos y les enviaba una copia del dataset con
los volumenes abiertos. (2) Windows bloqueo `scipy.optimize` (DLL `_arpacklib`), que se usaba solo para
emparejar fragmentos con el algoritmo hungaro.

**Que se obtuvo de la IA:** el dataset ya no envia los volumenes abiertos y los procesos de carga se crean
una sola vez (`persistent_workers`); algoritmo hungaro propio en `src/eval/asignacion.py`; el verificador de
entorno ahora tambien importa los modulos del proyecto, para detectar bloqueos indirectos; el cargador del
checkpoint del Taller 3 busca los pesos dentro de la clave `backbone`.

**Errores de la IA detectados:** la primera version del cargador de pesos asumia un solo diccionario y fallo
con el formato real del Taller 3 (`backbone`, `rpn`, `det`); `verificar_entorno.py` reporto `scipy.optimize`
como OK porque en ese momento aun no estaba bloqueado: la comprobacion por librerias sueltas no basta.

**Verificacion:** 16 pruebas pasan, entre ellas el hungaro propio contra scipy en 200 matrices aleatorias con
empates; con el bloqueo simulado corren entrenamiento, evaluacion y diagnostico; y las metricas por fragmento
son identicas a las de la version con scipy en 144 fragmentos (mismas parejas, mismo Dice).

**Analisis critico:** los dos errores salieron en mis corridas en Windows y yo se los
reporte a la IA; ella propuso las correcciones y las pruebas contra scipy.

## Sesion — interpretabilidad en el notebook

**Contexto:** el notebook no mostraba el gamma de BatchNorm de los bloques, Grad-CAM, como se ajustan las
cajas ni pacientes con fractura (solo se veian caderas sin fragmentos).

**Que se obtuvo de la IA:** `src/analisis.py` con gamma de BatchNorm por capa (y % de canales apagados),
Grad-CAM y Smooth Grad-CAM de la cabeza de clasificacion comparados con el mapa de objectness (correlacion y
masa dentro de la caja, como en el Taller 3), candidatos antes y despues del NMS, IoU de cajas en test y
seleccion de pacientes y cortes con fractura; nueva seccion 7 del notebook y seccion 6 sobre el paciente de
test con mas fragmentos.

**Verificacion:** el notebook se ejecuto completo en modo prueba sin errores; prueba automatica de que el
calculo del gamma detecta canales apagados. Con el modelo real, ningun canal del backbone esta apagado
(gamma medio entre 0.91 y 0.99 por bloque).

**Analisis critico:** le pedi a la IA que el notebook mostrara pacientes con fractura,
Grad-CAM y el gamma de BatchNorm, porque sin eso no se podia explicar como decide la red.

## Sesion — precision de fragmentos y tamano minimo en mL

**Contexto:** en el caso 039 el modelo predijo 19 fragmentos para 9 reales; los sobrantes miden 0.27-0.68 mL.
El Dice de fragmento se calcula desde el ground truth y no penaliza fragmentos predichos de mas.

**Que se obtuvo de la IA:** `metricas_fragmentos` reporta fragmentos predichos y reales, emparejados, precision
y la lista de fragmentos espurios con su volumen; `evaluar.py` los agrega por nivel; parametro
`postproceso.min_volumen_ml` (tamano minimo en unidades fisicas, convertido a voxeles con el spacing de cada CT);
`diagnostico_fragmentos.py --barrido_tamano` lo ajusta sobre validacion con el criterio "la mayor precision sin
perder mas de 0.005 de Dice de fragmento".

**Verificacion:** 19 pruebas pasan (precision con un fragmento espurio, conversion mL a voxeles); evaluar y el
barrido corren de punta a punta con un modelo de prueba, y el barrido rechaza un minimo que elimina fragmentos
reales pequenos.

**Analisis critico:** vi que el caso 039 tenia 19 fragmentos predichos para 9 reales aunque
el Dice salia bien. La IA ayudo a medir la precision y a elegir el minimo de 1 mL en validacion.


## Octubre — Orientacion de los CT

**Modelo usado:** Claude (Anthropic), Opus 5.5, en claude.ai.

**Que pedi:** revisar si nuestra rotacion estaba bien, con las notas del profesor y su articulo.

**En que ayudo la IA:** revisar las matrices de direccion de los 100 CT (34 rotados 180 grados) y
medir la inclinacion residual (mediana 3 grados).

**Lo que decidi:** no remuestrear el volumen y cubrir la inclinacion con la rotacion de 35 grados
que sugirio el profesor.

## Octubre — Receta a 384 px

**Modelo usado:** Claude (Anthropic), Opus 5.5, en claude.ai.

**Que pedi:** una forma de subir el Dice de fragmentos, y dejar el PC entrenando toda la noche.

**En que ayudo la IA:** la receta (384 px, peso x3 a secundarios, rotacion de 35 grados) y el script
para correrla de noche.

**Lo que decidi:** correr todo en mi PC, sin Colab ni Drive, y adoptar la receta solo si mejoraba
en validacion. Paso de 0.695 a 0.719.

## Octubre — Ablacion de CBAM y modelo final

**Modelo usado:** Claude (Anthropic), Opus 5.5, en claude.ai.

**Que pedi:** ajustar a 384 px el modelo sin CBAM para compararlo contra el mejorado. La idea fue mia;
la IA habia dejado como final el modelo con CBAM.

**En que ayudo la IA:** la configuracion y el script de la ablacion.

**Lo que decidi:** cuando las epocas tardaban 1800 s vi que el kernel del notebook ocupaba la RAM y
lo cerre (bajo a 473 s). En equipo elegimos el modelo sin CBAM (validacion 0.793) como final.

## Octubre — Dashboard con el modelo final

**Modelo usado:** Claude (Anthropic), Opus 5.5, en claude.ai.

**Que pedi:** que el dashboard cargara el modelo final; en el despliegue se vio que solo cargaba el modelo base.

**En que ayudo la IA:** que el dashboard use la configuracion guardada en cada checkpoint.

## 2026-10-04 — Despliegue del dashboard con tunel de Cloudflare

**Modelo usado:** Claude (Anthropic), Opus 5.5, en claude.ai.

**Prompts (resumen fiel; fue una sesion larga de ida y vuelta):**
> "segun el informe que me toca esperar para poder hacer x cosa y que puedo ir haciendo"
> (adjuntando el PDF del reparto de tareas y el enlace del repositorio)
>
> "dame el codigo de la terminal para activar el entorno y eso, con los pasos a seguir"
>
> "aclaremos que mi pc no tiene gpu nvidia"
>
> Despues, en cada paso, se pegaba la salida de la terminal o una captura del error
> (`Move-Item` sin permisos, `no such host` de cloudflared, `502 Bad Gateway`) y se pedia
> el siguiente paso.
>
> "todo eso para hacer el paso mio?"
>
> "dame mi IA_USAGE de mi parte ... me ayudas a preparar el despliegue.md"

**Que se obtuvo:**
- Que tareas de la seccion 6 se podian empezar ya y cuales dependian de otros integrantes.
- La secuencia de comandos de PowerShell para clonar, crear el entorno, instalar PyTorch en
  version CPU, organizar y preparar los datos, copiar los pesos a `checkpoints/`, levantar
  Streamlit y abrir el tunel.
- La confirmacion, leyendo el codigo, de que el pipeline corre en CPU sin cambios:
  `src/inferencia.py` carga con `map_location="cpu"`, el autocast solo se activa en CUDA y el
  dashboard solo ofrece `cpu` cuando no hay GPU.
- El diagnostico de dos fallas de red: el DNS del PC no respondia (comprobado con
  `nslookup` contra `1.1.1.1`) y el error 502, que no era del tunel sino de Streamlit apagado.
- La revision del video de respaldo y el borrador de `docs/despliegue.md` y de esta entrada.

**Que se modifico o verifico manualmente:**
- Todos los comandos se ejecutaron en el PC propio y se verificaron sus salidas: md5 de los
  3 zips, 100 imagenes y 100 etiquetas emparejadas, split con sha256 `99267ddb2310271e`
  (igual al del equipo), PyTorch `2.14.1+cpu`, tunel con todos los chequeos en `PASS`.
- La latencia (170.4 ms por corte, 74.2 s para el caso 002) se tomo del dashboard, no de la IA.
- El video se volvio a grabar porque la primera version duraba 58 s y no mostraba la
  advertencia de uso no clinico.

**Analisis critico:**
- *Aciertos:* leer el codigo antes de responder evito instalar cosas innecesarias (por
  ejemplo, confirmo que no hacia falta modificar nada para correr en CPU). El diagnostico del
  DNS y del 502 fue correcto y rapido porque se basaba en la salida real (`nslookup`, la
  pagina de error de Cloudflare). Tambien advirtio que el caso `001` es de entrenamiento y que
  la demo debia hacerse con un caso de test (`002`), lo que evita mostrar un desempeno inflado.
- *Errores de la IA:*
  1. Dio el comando de PyTorch con CUDA (2.5 GB) antes de preguntar si el PC tenia GPU. Hubo
     que reinstalar la version CPU.
  2. Pidio clonar el repositorio sin indicar que primero habia que salir de la carpeta del
     zip, asi que el clon quedo anidado y el entorno virtual se creo en la carpeta equivocada.
     La solucion que propuso despues (mover la carpeta) fallo porque estaba abierta en VS Code;
     lo que funciono fue clonar de nuevo.
  3. Se salio del alcance de la tarea: propuso medir la latencia en 3 casos con `inferir.py`,
     correr el EDA y analizar el MAE, cuando el reparto pide un solo caso y la latencia formal
     en CPU es tarea de otra integrante. Al preguntarle "todo eso para hacer el paso mio?", releyo el
     reparto y corrigio la lista.
  4. Supuso que habia que instalar cloudflared con `winget`, pero ya estaba instalado.
  5. No advirtio desde el principio que la terminal de Streamlit no debia reutilizarse, lo que
     causo el 502.
- *Limitaciones:* la IA no tiene acceso al PC; todo lo que sabe sale de lo que se le pega, asi
  que un error que no se copia completo no lo puede diagnosticar. Tampoco puede comprobar el
  enlace desde un celular: esa verificacion es manual. Los borradores de documentacion hay que
  revisarlos, porque los datos del equipo (procesador, RAM) y las capturas no los puede saber.

## 2026-10-16 — Ablaciones y línea base SAM

Herramienta: Claude (Anthropic), en chat, octubre de 2026. La IA se usó como apoyo de consulta y para armar comandos; las decisiones sobre qué correr, con qué modelo y qué subir fueron propias.

### 1. Encontrar y probar SAM

- **Qué se le pidió:** qué es SAM, de dónde se instala, qué pesos usar en una GPU de 4 GB y qué opciones tiene `scripts/baseline_sam.py`.
- **Qué aportó:** el repositorio oficial de SAM (Meta AI) y la URL de los pesos `sam_vit_b_01ec64.pth`, la versión más liviana.
- **Decisiones propias:** instalar SAM sin tocar las dependencias del proyecto (`--no-deps`); revisar las opciones reales del script con `--help` antes de usarlo; hacer una prueba rápida con pocos cortes antes de la corrida completa; repetir la comparación con el modelo final (`sin_cbam_384`) en lugar de quedarse con la del modelo anterior.
- **Análisis crítico:** la IA dio un comando sin `python -u`, y con la salida redirigida no se veía el avance; se detectó y se reinició la corrida. Las capturas de instrucciones del equipo llegaban recortadas, así que los comandos se confirmaron contra el `--help` del script.

### 2. Actualizar Git y bajar los cambios del equipo

- **Qué se le pidió:** cómo apuntar al repositorio renombrado, bajar `main`, trabajar en una rama propia y resolver un error 403 al subir.
- **Qué aportó:** los comandos de `git remote set-url`, `git pull --rebase origin main` y el diagnóstico del 403 (Windows usaba la credencial guardada de otra cuenta).
- **Decisiones propias:** mover fuera del repositorio los archivos de otra integrante que se habían generado en este equipo, para que ella los suba desde su cuenta; agregar siempre los archivos por nombre y no con `git add -A`; subir el trabajo en la rama `feat/baseline-sam` con Pull Request y revisión.
- **Análisis crítico:** en el 403 la IA supuso primero que faltaba la invitación de colaborador; la causa real se encontró al indicarle la cuenta correcta.

### 3. Volver a correr con los modelos nuevos

- **Qué se le pidió:** qué cambió en `main` después de la ablación de CBAM y qué había que repetir.
- **Qué aportó:** confirmó que tras el `pull` estaban los pesos y configuraciones nuevos (`mejorado_384`, `sin_cbam`, `sin_cbam_384`) y qué medición faltaba (validación de `con_tl`).
- **Decisiones propias:** medir la validación de `con_tl` para completar la tabla; correr SAM sobre el modelo final elegido; reportar los resultados de CBAM como "no aporta" y no como "mejora mucho", porque mejora en validación pero no en test.

### 4. Comandos usados en las pruebas

```
# Entorno y datos
python scripts/verificar_entorno.py
python scripts/preparar_datos.py

# SAM: instalación, prueba rápida y corridas completas
python -m pip install --no-deps git+https://github.com/facebookresearch/segment-anything.git
Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile checkpoints\sam_vit_b_01ec64.pth
python scripts/baseline_sam.py --help
python scripts/baseline_sam.py --config configs/mejorado_384.yaml --ckpt pesos/mejorado_384.pth --sam checkpoints/sam_vit_b_01ec64.pth --cada 50
python -u scripts/baseline_sam.py --config configs/mejorado_384.yaml --ckpt pesos/mejorado_384.pth --sam checkpoints/sam_vit_b_01ec64.pth
python -u scripts/baseline_sam.py --config configs/ablacion_sin_cbam_384.yaml --ckpt pesos/sin_cbam_384.pth --sam checkpoints/sam_vit_b_01ec64.pth

# Validación de con_tl
python scripts/diagnostico_fragmentos.py --config configs/ablacion_con_tl.yaml --ckpt pesos/con_tl.pth

# Git
git remote set-url origin https://github.com/ShadowBlack33/Proyecto-Analitica-de-Datos.git
git checkout -b feat/baseline-sam
git pull --rebase origin main
git add docs/resultados_ablacion_sam.md
git push -u origin feat/baseline-sam
```

## 2026-10-16 — Informe final, registro de uso de IA y diseno del dashboard

**Modelo usado:** Claude (Anthropic), Opus 5.5, en claude.ai.

**Prompts (resumen fiel):**
> "necesito el informe final en doc, Arial 12, color negro, sin marca de agua, maximo 4 paginas
> incluyendo bibliografia"
>
> "el IA_USAGE formalo en un solo archivo" (con las entradas de cada integrante)
>
> "quiero que el front del proyecto de analitica se vea parecido al de esta carpeta, con colores
> hospitalarios blanco y verde azulado; no agregues cosas ni cambies el funcionamiento, solo diseno"
>
> Ademas se le paso el enunciado del proyecto y las paginas del reto PENGWIN para las referencias.

**Que se obtuvo:**
- Un borrador del informe en Word, con las cifras de `docs/resultados/` y las tablas pedidas por el
  enunciado: metricas contra objetivos, ablacion con tiempos de entrenamiento, matriz de confusion,
  cinco peores casos con hipotesis y comparacion de la distancia predicha contra el ground truth.
- La union de las entradas de uso de IA del equipo en este archivo.
- El diseno del dashboard (`src/dashboard/estilo.css`, `src/dashboard/marca.py`, `.streamlit/config.toml`
  y el logo del reto), conectado a `app.py` con cuatro lineas que no tocan ningun control.

**Que se modifico o verifico manualmente:**
- El borrador se reviso contra el enunciado: faltaban los objetivos reales de la tabla de metricas,
  la matriz de confusion, la galeria de peores casos y la comparacion de distancias, y se agregaron.
- Las referencias del reto se tomaron de pengwin.grand-challenge.org.
- El diseno se pidio mas sobrio, con barra lateral verde y el logo de PENGWIN, hasta la version final.

**Analisis critico:**
- *Aciertos:* calculo la comparacion de distancias desde los JSON de evaluacion y encontro el hallazgo
  principal: 20 de 25 fragmentos emparejados tienen error de 1 mm o menos, y el error medio lo causan
  tres fragmentos realmente desplazados que el modelo midio como pegados. Tambien detecto que el
  enunciado exige CBAM en el backbone, algo que el modelo final ya no tiene, y que el dashboard solo
  cargaba la configuracion del modelo base; lo verifico cargando los pesos.
- *Errores de la IA:* la primera version del informe no cubria todos los requisitos del enunciado
  (faltaban la matriz de confusion, la galeria y la comparacion de distancias); una actualizacion del
  diseno no llego al PC aunque la herramienta la reporto como escrita, y solo se noto cuando el
  dashboard seguia viendose igual (desde entonces se verifica con checksum); una version del informe
  quedo en 5 paginas y hubo que recortarla.
- *Limitaciones:* las hipotesis de los peores casos salen de los numeros y no de mirar las imagenes, asi
  que hay que confirmarlas con la galeria; el informe quedo justo en 4 paginas y Word puede acomodar las
  tablas distinto al convertidor usado para verificarlo.

## 2026-10-17 — Evaluación: matriz de confusión, peores casos, latencia en CPU y model cards

**Modelo usado:** Claude (Anthropic), Sonnet 5.5 y Opus 5.5, en claude.ai.

**Prompt (resumen fiel):**
> Guíame en mi parte (matriz de confusión, 5 peores casos, latencia en CPU y model cards) dándome los comandos
> para correrla en el PC de otro integrante del equipo y para subirla desde mi git, y hazme un documento de avance
> tipo checklist para el docente para exponerlo.

**Mis decisiones:**
- Correr todo en el PC de otro integrante del equipo (RTX 3050, Ryzen 7 5800H, 31.3 GB de RAM), porque mi PC no
  tiene GPU ni la capacidad para trabajar con los datos.
- No dar por buenos los resultados del modelo `completo` ni de `mejorado_384`: cuando el equipo cambió el modelo final,
  volví a correr toda mi parte con `sin_cbam_384`.
- Revisar las cifras: noté que la matriz usaba 4005 cortes y no los 4821 de los volúmenes completos, y eso explicó
  la diferencia de F1 con la evaluación (0.988 contra 0.981).
- Revisar cada figura de los peores casos antes de aceptar la descripción.
- Pedir un documento de avance ("PENGWIN · Avance semana 10") para la exposición.

**Comandos que pedí y usé:**
```
git fetch origin
git merge origin/main
python scripts/matriz_confusion.py --config configs/ablacion_sin_cbam_384.yaml --ckpt pesos/sin_cbam_384.pth --nombre sin_cbam_384 --evaluacion docs/resultados/sin_cbam_384/evaluacion_cuda.json
python scripts/galeria_peores.py --config configs/ablacion_sin_cbam_384.yaml --ckpt pesos/sin_cbam_384.pth --evaluacion docs/resultados/sin_cbam_384/evaluacion_cuda.json
python scripts/inferir.py --config configs/ablacion_sin_cbam_384.yaml --ckpt pesos/sin_cbam_384.pth --ct data/raw/images/<caso>.mha --device cpu   # casos 005, 023 y 048
git add <mis archivos>
git commit -m "..."
git push -u origin feat/matriz-galeria
```

**Qué se obtuvo de la IA:** los scripts `matriz_confusion.py` y `galeria_peores.py`, los borradores de
`docs/peores_casos.md` y de las model cards, y los textos del documento de avance.

**Análisis crítico:**
- Las primeras versiones de `peores_casos.md` tenían afirmaciones que no se sostenían al compararlas con los datos
  (por ejemplo, qué caso tenía más espurios); se corrigieron.
- La IA no pudo explicar la diferencia de F1 entre la matriz y la evaluación; la causa salió de revisar el número de cortes.
- Supuso cosas del equipo que no sabía (qué PC estaba entrenando, el orden de los pasos de git); hubo que corregirla.
- Las causas de los peores casos son hipótesis a partir de un solo corte por caso.
