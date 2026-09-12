# anonimizza-cartella.ps1 - trasforma una cartella di foto in un archivio ZIP,
# lo invia al server di anonimizzazione e salva l'output pronto.
#
# Uso (una riga, da qualsiasi cartella):
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\anonimizza-cartella.ps1 -Cartella "C:\percorso\delle\foto"
#
# Opzioni:
#   -Camera    id della telecamera per queste foto (default CAM_010).
#              ATENZIONE: e' l'identita' della camera nel sistema - usa lo stesso
#              id solo per la stessa telecamera reale, cosi' il sistema impara.
#   -Output    percorso del file ZIP anonimizzato prodotto (default:
#              <cartella>\CAM_xxx_elaborato.zip accanto alle foto originali)
#   -ServerUrl base del server (default http://localhost:8080)
#   -Ricorsiva include le foto anche nelle sottocartelle (default: solo primo livello)

param(
    [Parameter(Mandatory = $true)] [string] $Cartella,
    [string] $Camera = "CAM_010",
    [string] $Output = "",
    [string] $ServerUrl = "http://localhost:8080",
    [switch] $Ricorsiva
)

$ErrorActionPreference = "Stop"

# --- 0. Verifiche preliminari ------------------------------------------------
if (-not (Test-Path -LiteralPath $Cartella -PathType Container)) {
    Write-Host "Cartella non trovata: $Cartella" -ForegroundColor Red
    exit 1
}
if ($Camera -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$') {
    Write-Host "Id camera non valido: '$Camera' (usa lettere, numeri, - e _; es. CAM_010)" -ForegroundColor Red
    exit 1
}

$serverProc = Get-Process -Name "anonimizzazione_volti" -ErrorAction SilentlyContinue
if (-not $serverProc) {
    Write-Host "Il server non e' in esecuzione. Avvialo prima con doppio click su start-server.cmd" -ForegroundColor Red
    exit 1
}

# --- 1. Raccogli le foto -----------------------------------------------------
$extValid = @(".jpg", ".jpeg", ".png")
$gciArgs = @{ LiteralPath = $Cartella; File = $true }
if ($Ricorsiva) { $gciArgs.Recurse = $true }
$tutte = @(Get-ChildItem @gciArgs)
$foto = @($tutte | Where-Object { $extValid -contains $_.Extension.ToLower() } | Sort-Object FullName)

if ($foto.Count -eq 0) {
    Write-Host "Nessuna foto JPG/PNG trovata in $Cartella" -ForegroundColor Red
    exit 1
}
$altreFile = $tutte.Count - $foto.Count

# --- 2. Crea lo ZIP (nomi normalizzati CAMERA_N.ext, riconosciuti dal server) -
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zipPath = Join-Path $env:TEMP "av_cartella_$stamp.zip"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    for ($i = 0; $i -lt $foto.Count; $i++) {
        $entry = "{0}_{1}{2}" -f $Camera, ($i + 1), $foto[$i].Extension.ToLower()
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $foto[$i].FullName, $entry, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
} finally {
    $zip.Dispose()
}
$zipSize = (Get-Item -LiteralPath $zipPath).Length

# --- 3. Invia al server -------------------------------------------------------
# Copia lo zip come 'upload.zip' in una cartella temporanea senza spazi: cosi'
# la riga di curl non ha mai problemi di quoting, qualsiasi sia il percorso.
$outPath = if ($Output) { $Output } else { Join-Path $Cartella ($Camera + "_elaborato.zip") }
$tmpDir = Join-Path $env:TEMP ("av_upload_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $tmpDir | Out-Null
$hdrFile = Join-Path $tmpDir "headers.txt"
Copy-Item -LiteralPath $zipPath -Destination (Join-Path $tmpDir "upload.zip")

Write-Host ("Invio di {0} foto (camera {1}, {2:N1} MB) al server..." -f $foto.Count, $Camera, ($zipSize / 1MB)) -ForegroundColor Cyan
$t0 = Get-Date
Push-Location $tmpDir
try {
    $code = & curl.exe -s -S -X POST "$ServerUrl/anonymize" -F "file=@upload.zip" -o "$outPath" -D "$hdrFile" -w "%{http_code}"
    $curlErr = $LASTEXITCODE
} finally {
    Pop-Location
}
$elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

# Legge X-Processing-Errors dagli header PRIMA di cancellare la cartella temporanea
$procErrors = "?"
if (Test-Path -LiteralPath $hdrFile) {
    $line = (Get-Content -LiteralPath $hdrFile) | Where-Object { $_ -match '^X-Processing-Errors:\s*(\d+)' } | Select-Object -First 1
    if ($line -match '^X-Processing-Errors:\s*(\d+)') { $procErrors = $Matches[1] }
}
Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue

if ($curlErr -ne 0 -or $code -ne "200") {
    Write-Host "Invio fallito (HTTP $code, curl exit $curlErr). Controlla i log in logs\server.log" -ForegroundColor Red
    exit 1
}

$ignorati = ""
if ($altreFile -gt 0) { $ignorati = "  ($altreFile file non-JPG/PNG ignorati)" }
Write-Host ""
Write-Host "FATTO in $elapsed secondi." -ForegroundColor Green
Write-Host "  Foto inviate         : $($foto.Count)$ignorati"
Write-Host "  Camera               : $Camera"
Write-Host "  Errori di elaborazione: $procErrors"
Write-Host "  Archivio anonimizzato: $outPath"
Write-Host ""
Write-Host "Nota: con una camera nuova il sistema e' in fase LEARNING e sfoca tutto il riquadro"
Write-Host "del volto; dopo LEARNING_DAYS giorni di dati passera' in ACTIVE con maschere surgicali."
