# test-wider.ps1 - test locale in ACTIVE mode: YOLOv8-Face (soglia ACTIVE bassa)
# + classificatore binario (second check p_face >= 0.5) + maschere MediaPipe +
# head-fallback COCO, esattamente come il bench python bench_classifier.py.
#
# Cosa fa, in ordine:
#   1. carica TUTTE le variabili dal .env del progetto (niente export a mano)
#   2. avvia un mini http.server su 127.0.0.1:8765 che serve le model ONNX
#      della MODEL_CACHE_DIR (serve per CLASSIFIER_MODEL_URL del .env)
#   3. compila la build release (disattivabile con -NoBuild)
#   4. costruisce un archivio CAM_001_frame_XXXX.jpg da WIDER-val
#      (rename, cosi' esiste UNA sola camera: CAM_001)
#   5. upload warmup (registra geometria telaio) -> forza ACTIVE con
#      POST /operator/cameras/CAM_001/roi {"type":"full"}
#   6. stampa /operator/classifier e /operator/cameras/CAM_001 (verifica gate)
#   7. upload + cronometra l'archivio completo (run 2, ormai ACTIVE)
#   8. riassume stato camera, job e timing per immagine (target perf)
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File scripts\test-wider.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\test-wider.ps1 -MaxImages 300 -NoBuild
#
# File ASCII-only come gli altri script (PS 5.1 legge senza BOM come ANSI).
param(
    [string]$Zip = (Join-Path (Split-Path -Parent $PSScriptRoot) "WIDER_val.zip"),
    [int]$MaxImages = 100,
    [int]$Port = 8080,
    [switch]$NoBuild,
    [switch]$KeepRunning,
    [string]$ModelsPort = "8765",
    [string]$Set = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# ─── 1. .env automatico ──────────────────────────────────────────────────────
function Load-EnvFile([string]$Path) {
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $t = $line.Trim()
        if ([string]::IsNullOrEmpty($t) -or $t.StartsWith("#")) { continue }
        $i = $t.IndexOf("=")
        if ($i -lt 0) { continue }
        $key = $t.Substring(0, $i).Trim()
        $val = $t.Substring($i + 1).Trim()
        if ($key -match "^[A-Za-z_][A-Za-z0-9_]*$") { Set-Item -Path "Env:$key" -Value $val }
    }
}
$envPath = "$Root\.env"
if (-not (Test-Path -LiteralPath $envPath)) { throw "Manca $envPath" }
Load-EnvFile $envPath
Write-Host "==> .env caricato (modelli, soglie, URL classificatore)."

# ─── 2. dir di test isolate (mai nei dati reali) ─────────────────────────────
$RunDir = "$Root\test-active"
if (Test-Path -LiteralPath $RunDir) {
    Write-Host "==> Pulisco $RunDir (dir di test dedicata)..."
    Remove-Item -LiteralPath $RunDir -Recurse -Force
}
foreach ($d in @("$RunDir\data", "$RunDir\fp", "$RunDir\seed\real_faces", "$RunDir\models-backup")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
$env:BIND_ADDR = "127.0.0.1:$Port"
$env:DATA_DIR = "$RunDir\data\"
$env:DATASET_FP_DIR = "$RunDir\fp\"
$env:DATASET_SEED_REAL_FACES_DIR = "$RunDir\seed\real_faces\"
$env:MODELS_BACKUP_DIR = "$RunDir\models-backup\"
if (-not $env:OPERATOR_API_KEY) { $env:OPERATOR_API_KEY = "test-key" }
$env:RUST_LOG = "info,anonimizzazione_volti=info,perf=info"

# Override espliciti per gli sweep: -Set "KEY=VAL,KEY=VAL" (dopo il .env).
if ($Set) {
    foreach ($pair in $Set.Split(",")) {
        $kv = $pair.Trim().Split("=", 2)
        if ($kv.Length -ne 2 -or -not $kv[0].Trim()) { continue }
        Set-Item -Path "Env:$($kv[0].Trim())" -Value $kv[1].Trim()
        Write-Host "==> override env: $($kv[0].Trim())=$($kv[1].Trim())"
    }
}
Write-Host "==> Data dirs isolate: $RunDir  (operator key: $env:OPERATOR_API_KEY)"

# ─── 3. mini server modelli (per la CLASSIFIER_MODEL_URL del .env) ───────────
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) {
    $alt = "$Root\.venv-train\Scripts\python.exe"
    if (Test-Path -LiteralPath $alt) { $py = $alt }
}
if (-not $py) { throw "python non trovato (serve per il mini http.server dei modelli)" }
$httpProc = $null
if ($env:CLASSIFIER_MODEL_URL -match "127\.0\.0\.1:${ModelsPort}|localhost:${ModelsPort}") {
    Write-Host "==> Avvio http.server modelli su :$ModelsPort (serve $env:MODEL_CACHE_DIR)..."
    $httpProc = Start-Process -FilePath $py -ArgumentList "-m", "http.server", $ModelsPort, "--bind", "127.0.0.1", "--directory", $env:MODEL_CACHE_DIR -RedirectStandardOutput "$RunDir\http-models.log" -RedirectStandardError "$RunDir\http-models.err.log" -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 1
    try { Invoke-WebRequest -Uri "http://127.0.0.1:$ModelsPort/classifier_manual.onnx" -Method Head -TimeoutSec 3 -UseBasicParsing | Out-Null; Write-Host "    OK - modello raggiungibile." } catch { Write-Host "    (il file modello non e' servito, ma la cache potrebbe bastare)" }
}

# ─── 4. build ────────────────────────────────────────────────────────────────
# La crate vive in un workspace (../Cargo.toml): cargo scrive gli artefatti in
# <workspace>/target, NON in <crate>/target. Interroga `cargo metadata` per il
# percorso reale, con fallback alle due posizioni note.
function Resolve-BinPath([string]$CrateRoot, [string]$Profile) {
    $name = "anonimizzazione_volti.exe"
    $candidates = @()
    try {
        $meta = (& cargo metadata --no-deps --format-version 1 2>$null) | ConvertFrom-Json
        if ($meta.target_directory) {
            $candidates += (Join-Path $meta.target_directory "$Profile\$name")
        }
    } catch { }
    $candidates += (Join-Path $CrateRoot "target\$Profile\$name")
    foreach ($c in $candidates) { if (Test-Path -LiteralPath $c) { return $c } }
    return $candidates[0]
}
$exe = Resolve-BinPath $Root "release"
if (-not $NoBuild) {
    Write-Host "==> cargo build --release ..."
    cargo build --release | Out-Host
    if (-not $?) { throw "build fallita" }
}
if (-not (Test-Path -LiteralPath $exe)) { throw "Binario mancante: $exe (build senza -NoBuild?)" }

# ─── 5. archivi di test: rename WIDER -> CAM_001_frame_XXXX.jpg ─────────────
if (-not (Test-Path -LiteralPath $Zip)) { throw "Zip sorgente mancante: $Zip" }
Add-Type -AssemblyName System.IO.Compression.FileSystem
function New-CameraZip([string]$SrcZip, [string]$DstZip, [int]$Max) {
    $zin = [System.IO.Compression.ZipFile]::OpenRead($SrcZip)
    $entries = @($zin.Entries | Where-Object { $_.Name -match "\.(jpe?g|png)$" } | Select-Object -First $Max)
    $zout = [System.IO.Compression.ZipFile]::Open($DstZip, [System.IO.Compression.ZipArchiveMode]::Create)
    $i = 0
    foreach ($e in $entries) {
        $i++
        $newName = ("CAM_001_frame_{0:d4}_{1}" -f $i, $e.Name)
        $src = $e.Open()
        $dst = $zout.CreateEntry($newName).Open()
        $src.CopyTo($dst)
        $dst.Dispose(); $src.Dispose()
    }
    $zout.Dispose(); $zin.Dispose()
    return $i
}
$inZip = "$RunDir\wider_active_in.zip"
$warmZip = "$RunDir\warmup.zip"
Remove-Item $inZip, $warmZip -ErrorAction SilentlyContinue
$nWarm = New-CameraZip $Zip $warmZip 1
$nMain = New-CameraZip $Zip $inZip $MaxImages
Write-Host "==> Archivi pronti: warmup=$nWarm immagine, main=$nMain immagini (camera CAM_001)"
if ($nWarm -lt 1) { throw "Nessuna immagine trovata nello zip sorgente" }

# ─── 6. avvio servizio ───────────────────────────────────────────────────────
$logOut = "$RunDir\server.log"
$logErr = "$RunDir\server.err.log"
Remove-Item $logOut, $logErr -ErrorAction SilentlyContinue
Write-Host "==> Avvio server su 127.0.0.1:$Port ..."
$proc = Start-Process -FilePath $exe -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru -WindowStyle Hidden
$ok = $false
for ($t = 0; $t -lt 240; $t++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
    if ($proc.HasExited) { break }
}
if (-not $ok) {
    Write-Host "Server non partito:" -ForegroundColor Red
    Get-Content $logOut -Tail 25 -ErrorAction SilentlyContinue
    Get-Content $logErr -Tail 25 -ErrorAction SilentlyContinue
    if ($httpProc) { Stop-Process -Id $httpProc.Id -Force -ErrorAction SilentlyContinue }
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "    OK - /health risponde (PID $($proc.Id))"

# ─── 7. warmup (geometria) + força ACTIVE via operator ──────────────────────
Write-Host "==> Upload warmup (registra geometria CAM_001)..."
& curl.exe -fsS -m 120 -F "file=@$warmZip" "http://127.0.0.1:$Port/anonymize" -o "$RunDir\warmup_out.zip"
Write-Host "    Warmup ok. Forzo ACTIVE (ROI full-frame)..."
$body = '{"type":"full"}'
$act = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/operator/cameras/CAM_001/roi" -Method Post -ContentType "application/json; charset=utf-8" -Headers @{ "X-Operator-Key" = $env:OPERATOR_API_KEY } -Body $body
$act = $act.error
Write-Host "    $act"
if ($act -notmatch "ACTIVE") { Write-Host "ATTENZIONE: la camera non risulta ACTIVE dopo il comando operator." -ForegroundColor Red }

Write-Host "==> Stato gate classificatore (prima della run):"
$clf = & curl.exe -sS -H "X-Operator-Key: $env:OPERATOR_API_KEY" "http://127.0.0.1:$Port/operator/classifier"
Write-Host "    $clf"
$cam = & curl.exe -sS -H "X-Operator-Key: $env:OPERATOR_API_KEY" "http://127.0.0.1:$Port/operator/cameras/CAM_001"
Write-Host "    CAM_001: $cam"

# ─── 8. run principale cronometrata ─────────────────────────────────────────
$outZip = "$RunDir\wider_active_elaborato.zip"
$start = Get-Date
Write-Host "==> Upload main ($nMain immagini, ACTIVE, tempo da adesso)..."
& curl.exe -fsS -m 1800 -D - -F "file=@$inZip" "http://127.0.0.1:$Port/anonymize" -o $outZip
$elapsed = (Get-Date) - $start
Write-Host ""
Write-Host "=== TEMPO TOTALE: $([math]::Round($elapsed.TotalSeconds,1)) s ===" -ForegroundColor Green

# ─── 9. riepilogo ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==> Stato camera e classificatore (dopo la run):"
$cam = & curl.exe -sS -H "X-Operator-Key: $env:OPERATOR_API_KEY" "http://127.0.0.1:$Port/operator/cameras/CAM_001"
$clf = & curl.exe -sS -H "X-Operator-Key: $env:OPERATOR_API_KEY" "http://127.0.0.1:$Port/operator/classifier"
Write-Host "    CAM_001: $cam"
Write-Host "    classifier: $clf"

if (-not $KeepRunning) {
    Write-Host "==> Fermo server e http-server modelli..."
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    if ($httpProc) { Stop-Process -Id $httpProc.Id -Force -ErrorAction SilentlyContinue }
}

Write-Host ""
Write-Host "==> Riepilogo job da log:"
Get-Content $logOut | Select-String -Pattern "job '|branch" | ForEach-Object { $_.Line }

Write-Host ""
Write-Host "==> Statistiche timing (target perf, per immagine):"
$raw = Get-Content $logOut -Raw
$clean = [regex]::Replace($raw, "\x1b\[[0-9;]*m", "")
$matches2 = [regex]::Matches($clean, "decode_ms=([\d.]+)\s+process_ms=([\d.]+)\s+encode_ms=([\d.]+)\s+total_ms=([\d.]+)")
Write-Host "Righe timing: $($matches2.Count)"
if ($matches2.Count -gt 0) {
    $decode = @(); $process = @(); $encode = @(); $total = @(); $detect = @()
    foreach ($m in $matches2) {
        $decode += [double]$m.Groups[1].Value
        $process += [double]$m.Groups[2].Value
        $encode += [double]$m.Groups[3].Value
        $total += [double]$m.Groups[4].Value
    }
    foreach ($m in [regex]::Matches($clean, "detect=(\d+)")) { $detect += [int]$m.Groups[1].Value }
    $avg = { param($a) ($a | Measure-Object -Average).Average }
    Write-Host ("detect  avg: {0:N1} detezioni" -f (& $avg $detect))
    Write-Host ("decode  avg: {0:N1} ms" -f (& $avg $decode))
    Write-Host ("process avg: {0:N1} ms   (DETECTOR+classificatore+maschera)" -f (& $avg $process))
    Write-Host ("encode  avg: {0:N1} ms" -f (& $avg $encode))
    Write-Host ("total   avg: {0:N1} ms" -f (& $avg $total))
}

Write-Host ""
Write-Host "Output: $outZip"