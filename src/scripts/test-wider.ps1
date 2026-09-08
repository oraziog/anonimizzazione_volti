# test-wider.ps1 - avvia il server, carica WIDER_val.zip, raccoglie i timing
$ErrorActionPreference = "Stop"
$Root = "C:\Users\Admin\Music\anonimizzazione_volti\src"
Set-Location $Root

# Detector: 'yolo' (default) o 'retinaface' (sovrascrivibile via env).
if (-not $env:DETECTOR_MODE) { $env:DETECTOR_MODE = "yolo" }

# Env dal .env
$env:BIND_ADDR = "127.0.0.1:8080"
$env:MAX_CONCURRENT_IMAGES = "2"
$env:JPEG_QUALITY = "80"
$env:YOLO_MODEL_URL = "https://github.com/yakhyo/yolov8-face-onnx-inference/releases/download/weights/yolov8n-face.onnx"
$env:CLASSIFIER_MODEL_URL = ""
$env:MODEL_CACHE_DIR = "$Root\models_cache\"
$env:MODEL_YOLO_SHA256 = "33f3951af7fc0c4d9b321b29cdcd8c9a59d0a29a8d4bdc01fcb5507d5c714809"
$env:MODEL_CLASSIFIER_SHA256 = ""
$env:DATA_DIR = "$Root\testdata\"
$env:DATASET_FP_DIR = "$Root\testfp\"
$env:DATASET_SEED_REAL_FACES_DIR = "$Root\testseed\real_faces\"
$env:MODELS_BACKUP_DIR = "$Root\testmodels\backup\"
$env:YOLO_CONF_THRESHOLD = "0.05"
$env:YOLO_NMS_IOU = "0.45"
$env:DETECTOR_MODE = "retinaface"
$env:FP_CROP_CONF_MAX = "0.50"
$env:INITIAL_BLUR_SIGMA = "20.0"
$env:ANON_MODE = "blur"
$env:LEARNING_DAYS = "30"
$env:RETENTION_ENABLED = "false"
$env:RETENTION_MAX_DAYS = "0"
$env:RETENTION_MAX_GB = "0"
# Abilita il target 'perf' (debug) + info per i job
$env:RUST_LOG = "info,anonimizzazione_volti=info"

New-Item -ItemType Directory -Force -Path $env:DATA_DIR | Out-Null

$exe = "$Root\target\release\anonimizzazione_volti.exe"
$logOut = "$Root\test-wider.log"
$logErr = "$Root\test-wider.err.log"
Remove-Item $logOut, $logErr -ErrorAction SilentlyContinue

Write-Host "==> Avvio server su 127.0.0.1:8080..."
$proc = Start-Process -FilePath $exe -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru -WindowStyle Hidden

$ok = $false
for ($t = 0; $t -lt 120; $t++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8080/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
    if ($proc.HasExited) { break }
}
if (-not $ok) {
    Write-Host "Server non partito:" -ForegroundColor Red
    Get-Content $logOut -Tail 20 -ErrorAction SilentlyContinue
    Get-Content $logErr -Tail 20 -ErrorAction SilentlyContinue
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "    OK - /health risponde (PID $($proc.Id))"

# Upload + cronometrare
$start = Get-Date
Write-Host "==> Upload WIDER_val.zip (il tempo parte ora)..."
& curl.exe -fsS -m 3600 -D - -F "file=@$Root\WIDER_val.zip" "http://127.0.0.1:8080/anonymize" -o "$Root\WIDER_val_elaborato.zip"
$elapsed = (Get-Date) - $start
Write-Host ""
Write-Host "=== TEMPO TOTALE: $([math]::Round($elapsed.TotalSeconds,1)) s ===" -ForegroundColor Green

# Ferma il server
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500

# Pulisci gli errori di stdin chiuso nel log di stderr
Write-Host ""
Write-Host "==> Riepilogo job da log (info):"
Get-Content $logOut | Select-String -Pattern "job '" | ForEach-Object { $_.Line }

Write-Host ""
Write-Host "==> Statistiche timing (target perf, per immagine):"
$raw = Get-Content $logOut -Raw
# I log tracing includono sequenze ANSI: le rimuoviamo prima del match.
$clean = [regex]::Replace($raw, '\x1b\[[0-9;]*m', '')
$matches2 = [regex]::Matches($clean, 'decode_ms=([\d.]+)\s+process_ms=([\d.]+)\s+encode_ms=([\d.]+)\s+total_ms=([\d.]+)')
Write-Host "Righe timing: $($matches2.Count)"
if ($matches2.Count -gt 0) {
    $decode = @(); $process = @(); $encode = @(); $total = @(); $detect = @()
    foreach ($m in $matches2) {
        $decode += [double]$m.Groups[1].Value
        $process += [double]$m.Groups[2].Value
        $encode += [double]$m.Groups[3].Value
        $total += [double]$m.Groups[4].Value
    }
    $mdet = [regex]::Matches($clean, 'detect=(\d+)')
    foreach ($m in $mdet) { $detect += [int]$m.Groups[1].Value }
    $avg = { param($a) ($a | Measure-Object -Average).Average }
    Write-Host ("detect  avg: {0:N1} detezioni" -f (& $avg $detect))
    Write-Host ("decode  avg: {0:N1} ms" -f (& $avg $decode))
    Write-Host ("process avg: {0:N1} ms   (DETECTOR + blur)" -f (& $avg $process))
    Write-Host ("encode  avg: {0:N1} ms" -f (& $avg $encode))
    Write-Host ("total   avg: {0:N1} ms" -f (& $avg $total))
}

Write-Host ""
Write-Host "Output: $Root\WIDER_val_elaborato.zip"
