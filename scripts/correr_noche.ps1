# Corrida nocturna del experimento mejorado_384. Desde la raiz del repo, con el venv activo:
#   powershell -ExecutionPolicy Bypass -File scripts/correr_noche.ps1
# Todo queda tambien en resultados/mejorado_384/noche.log

$cfg = "configs/mejorado_384.yaml"
$ck = "checkpoints/mejorado_384/mejor_conjunto.pth"
New-Item -ItemType Directory -Force -Path "resultados/mejorado_384" | Out-Null
$log = "resultados/mejorado_384/noche.log"
$env:PYTHONUNBUFFERED = "1"

function Paso($titulo, $comando) {
    $t0 = Get-Date
    "`n===== $titulo  ($(Get-Date -Format 'HH:mm'))" | Tee-Object -FilePath $log -Append
    Invoke-Expression "$comando 2>&1" | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        "`n[FALLO] '$titulo' termino con codigo $LASTEXITCODE. Revisa $log" | Tee-Object -FilePath $log -Append
        exit 1
    }
    "[ok] $titulo en $([int]((Get-Date) - $t0).TotalMinutes) min" | Tee-Object -FilePath $log -Append
}

# que Windows no suspenda el PC mientras corre (enchufado)
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

Paso "1/6 Memoria de GPU a 384 px" "python scripts/perfil_memoria.py --config $cfg --resoluciones 384 --batches 8 12 16"
Paso "2/6 Cache a 384 px" "python scripts/preparar_datos.py --config $cfg"
Paso "3/6 Entrenamiento (segmentacion 12 + conjunto 8 epocas)" "python scripts/train.py --config $cfg --nombre mejorado_384"
Paso "4/6 Diagnostico en validacion" "python scripts/diagnostico_fragmentos.py --config $cfg --ckpt $ck"
Paso "5/6 Evaluacion en test" "python scripts/evaluar.py --config $cfg --ckpt $ck --degradacion 0"
Paso "6/6 Comparacion contra completo" "python scripts/comparar_experimentos.py --nuevo mejorado_384"
"`nTodo listo $(Get-Date -Format 'HH:mm')." | Tee-Object -FilePath $log -Append
