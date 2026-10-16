# Ablacion completa de CBAM en una sola corrida (~8.5 h en una RTX 3060 Laptop).
#   Paso A: modelo base SIN CBAM a 256 px (55 epocas), se compara contra completo.
#   Paso B: ajuste fino a 384 px con la receta final, se compara contra mejorado_384.
# Desde la raiz del repo, con el venv activo:
#   powershell -ExecutionPolicy Bypass -File scripts/correr_sin_cbam.ps1
# Todo queda tambien en resultados/sin_cbam/noche.log

New-Item -ItemType Directory -Force -Path "resultados/sin_cbam" | Out-Null
$log = "resultados/sin_cbam/noche.log"
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

powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

$c256 = "configs/ablacion_sin_cbam.yaml"
$c384 = "configs/ablacion_sin_cbam_384.yaml"
$k256 = "checkpoints/sin_cbam/mejor_conjunto.pth"
$k384 = "checkpoints/sin_cbam_384/mejor_conjunto.pth"

Paso "A1/8 Entrenamiento base sin CBAM, 256 px (55 epocas)" "python scripts/train.py --config $c256 --nombre sin_cbam"
Copy-Item $k256 "pesos/sin_cbam.pth" -Force
"[ok] pesos/sin_cbam.pth copiado" | Tee-Object -FilePath $log -Append
Paso "A2/8 Diagnostico en validacion (base)" "python scripts/diagnostico_fragmentos.py --config $c256 --ckpt $k256"
Paso "A3/8 Evaluacion en test (base)" "python scripts/evaluar.py --config $c256 --ckpt $k256 --degradacion 0"
Paso "A4/8 Comparacion sin_cbam vs completo" "python scripts/comparar_experimentos.py --nuevo sin_cbam --base completo"

Paso "B1/8 Ajuste fino sin CBAM, 384 px (20 epocas)" "python scripts/train.py --config $c384 --nombre sin_cbam_384"
Copy-Item $k384 "pesos/sin_cbam_384.pth" -Force
"[ok] pesos/sin_cbam_384.pth copiado" | Tee-Object -FilePath $log -Append
Paso "B2/8 Diagnostico en validacion (384)" "python scripts/diagnostico_fragmentos.py --config $c384 --ckpt $k384"
Paso "B3/8 Evaluacion en test (384)" "python scripts/evaluar.py --config $c384 --ckpt $k384 --degradacion 0"
Paso "B4/8 Comparacion sin_cbam_384 vs mejorado_384" "python scripts/comparar_experimentos.py --nuevo sin_cbam_384 --base mejorado_384"
"`nTodo listo $(Get-Date -Format 'HH:mm')." | Tee-Object -FilePath $log -Append
