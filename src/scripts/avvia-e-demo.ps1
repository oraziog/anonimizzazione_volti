# avvia-e-demo.ps1 - compila, avvia il server locale, carica un test.zip con
# foto di volti, mostra header + log e apre un'anteprima prima/dopo.
#
# Uso (da PowerShell, nella root del progetto):
#   powershell -ExecutionPolicy Bypass -File scripts\avvia-e-demo.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\avvia-e-demo.ps1 -Port 18080 -Release -AnonMode pixelate
#
# Parametri:
#   -Port        porta HTTP del server (default 8080)
#   -Release     usa la build release (piu veloce, build piu lunga)
#   -TestZip     percorso dello ZIP da caricare (se manca, lo crea con 3 foto)
#   -AnonMode    blur | pixelate  (vedi ANON_MODE nel servizio)
#   -KeepRunning non fermare il server alla fine
#   -NoPreview   non aprire il browser con il confronto
#
# NB: file volutamente ASCII-only (PowerShell 5.1 legge i file senza BOM come
# ANSI e i caratteri non-ASCII rompono il parsing delle stringhe).

param(
    [int]$Port = 8080,
    [switch]$Release,
    [string]$TestZip = "",
    [ValidateSet("blur", "pixelate")][string]$AnonMode = "blur",
    [switch]$KeepRunning,
    [switch]$NoPreview
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

# --- 1. Compila --------------------------------------------------------------
Write-Host "==> Build ($(if ($Release) { 'release' } else { 'debug' }))..."
if ($Release) { cargo build --release | Out-Host } else { cargo build | Out-Host }
$exe = Join-Path $Root "target\$(if ($Release) { 'release' } else { 'debug' })\anonimizzazione_volti.exe"
if (-not (Test-Path $exe)) { throw "Binario non trovato: $exe" }

# --- 2. Prepara lo ZIP di test (se non esiste) -------------------------------
if (-not $TestZip) { $TestZip = Join-Path $Root "test.zip" }
if (-not (Test-Path $TestZip)) {
    Write-Host "==> Creo test.zip con 3 foto demo..."
    $work = Join-Path $env:TEMP "av-demo-$PID"
    New-Item -ItemType Directory -Force -Path (Join-Path $work "in\CAM_001") | Out-Null
    $urls = @(
        "https://ultralytics.com/images/bus.jpg",
        "https://ultralytics.com/images/zidane.jpg",
        "https://ultralytics.com/images/face.jpg"
    )
    $i = 0
    foreach ($u in $urls) {
        $i++
        # Nome con prefisso camera (CAM_001_frame_00X.jpg) cosi lo ZIP piazzato
        # a root viene riconosciuto nel layout "prefix style".
        $dst = Join-Path $work "CAM_001_frame_$('{0:000}' -f $i).jpg"
        try {
            curl.exe -fsSL -m 40 -o $dst $u 2>$null
            if ((Get-Item $dst).Length -lt 5000) { Remove-Item $dst -ErrorAction SilentlyContinue }
        } catch { Remove-Item $dst -ErrorAction SilentlyContinue }
    }
    # Fallback: se nessun download e riuscito, genera 3 immagini a tinta unita
    $have = @(Get-ChildItem $work -Filter *.jpg -ErrorAction SilentlyContinue)
    if ($have.Count -lt 3) {
        Add-Type -AssemblyName System.Drawing
        for ($k = 1; $k -le 3; $k++) {
            $dst = Join-Path $work "CAM_001_frame_$('{0:000}' -f $k).jpg"
            $bmp = New-Object System.Drawing.Bitmap(640, 480)
            $g = [System.Drawing.Graphics]::FromImage($bmp)
            $g.Clear([System.Drawing.Color]::FromArgb(60 + 40 * $k, 90, 130))
            $bmp.Save($dst, [System.Drawing.Imaging.ImageFormat]::Jpeg)
            $g.Dispose(); $bmp.Dispose()
        }
    }
    Compress-Archive -Path (Join-Path $work "*.jpg") -DestinationPath $TestZip
    Remove-Item $work -Recurse -Force
    Write-Host "    ZIP creato: $TestZip"
}

# --- 3. Avvia il server ------------------------------------------------------
$logOut = Join-Path $Root "server-demo.log"
$logErr = Join-Path $Root "server-demo.err.log"
Remove-Item $logOut, $logErr -ErrorAction SilentlyContinue

Write-Host "==> Avvio server su 127.0.0.1:$Port (ANON_MODE=$AnonMode)..."
# Le variabili d'ambiente vengono ereditate dal processo figlio
$env:BIND_ADDR = "127.0.0.1:$Port"
$env:DATA_DIR = (Join-Path $Root "data-demo")
$env:DATASET_FP_DIR = (Join-Path $Root "fp-demo")
$env:MODEL_CACHE_DIR = (Join-Path $Root "models-cache")
$env:ANON_MODE = $AnonMode
New-Item -ItemType Directory -Force -Path $env:DATA_DIR | Out-Null

$proc = Start-Process -FilePath $exe -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru -WindowStyle Hidden

$ok = $false
for ($t = 0; $t -lt 60; $t++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
    if ($proc.HasExited) { break }
}
if (-not $ok) {
    Write-Host "Server non partito. Ultime righe di log:" -ForegroundColor Red
    Get-Content $logOut -Tail 20 -ErrorAction SilentlyContinue
    Get-Content $logErr -Tail 20 -ErrorAction SilentlyContinue
    throw "Server non raggiungibile su :$Port"
}
Write-Host "    OK - /health risponde (PID $($proc.Id))"

# --- 4. Upload + header ------------------------------------------------------
$outDir = [System.IO.Path]::GetDirectoryName($TestZip)
$outBase = [System.IO.Path]::GetFileNameWithoutExtension($TestZip)
$outZip = [System.IO.Path]::Combine($outDir, "${outBase}_elaborato.zip")
Write-Host "==> Upload $TestZip ..."
& curl.exe -fsS -m 300 -D - -F "file=@$TestZip" "http://127.0.0.1:$Port/anonymize" -o $outZip
Write-Host ""
Write-Host "==> Header della risposta (sopra) - output salvato in $outZip"
if (-not (Test-Path $outZip)) { throw "Upload fallito: nessun output" }

# --- 5. Log del server -------------------------------------------------------
Write-Host ""
Write-Host "==> Log del server (righe di job/camera):"
Get-Content $logOut -Tail 60 | Select-String -Pattern "job |camera |recovered|skipping" | ForEach-Object { Write-Host $_.Line }

# --- 6. Anteprima prima/dopo -------------------------------------------------
if (-not $NoPreview) {
    Write-Host "==> Preparo anteprima prima/dopo..."
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $tmp = Join-Path $env:TEMP "av-preview-$PID"
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    $zin = [System.IO.Compression.ZipFile]::OpenRead($TestZip)
    $zout = [System.IO.Compression.ZipFile]::OpenRead($outZip)
    # prima immagine JPG dentro lo ZIP di input
    $first = $zin.Entries | Where-Object { $_.FullName -match "\.(jpg|jpeg)$" } | Select-Object -First 1
    if ($first) {
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($first, (Join-Path $tmp "before.jpg"), $true)
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($first, (Join-Path $tmp "after.jpg"), $true)
        $zin.Dispose(); $zout.Dispose()
        $b64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes((Join-Path $tmp "before.jpg")))
        $b64b = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes((Join-Path $tmp "after.jpg")))
        $html = @"
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Anteprima anonimizzazione</title>
<style>body{font-family:sans-serif;background:#111;color:#ddd;padding:20px}
.pair{display:flex;gap:16px;flex-wrap:wrap}.card{background:#1c1c1c;border-radius:8px;padding:10px}
p{margin:4px 0;font-size:13px;color:#9ac}img{max-width:640px;max-height:480px;border-radius:4px}</style></head>
<body><h1>Prima / Dopo - $TestZip (ANON_MODE=$AnonMode)</h1><div class="pair">
<div class="card"><p>Originale</p><img src="data:image/jpeg;base64,$b64"></div>
<div class="card"><p>Elaborato</p><img src="data:image/jpeg;base64,$b64b"></div>
</div></body></html>
"@
        $htmlPath = Join-Path $Root "anteprima-demo.html"
        [System.IO.File]::WriteAllText($htmlPath, $html, [System.Text.Encoding]::UTF8)
        Write-Host "    Anteprima: $htmlPath (apertura browser...)"
        Start-Process $htmlPath
    }
}

# --- 7. Cleanup --------------------------------------------------------------
if (-not $KeepRunning) {
    Write-Host "==> Fermo il server (PID $($proc.Id))..."
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "==> Server lasciato attivo su http://127.0.0.1:$Port (log: $logOut)"
}
Write-Host ""
Write-Host "Fatto. Output: $outZip"