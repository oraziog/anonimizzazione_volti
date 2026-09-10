# push-pop.ps1 - prova end-to-end "push/pop" in stile curl verso il servizio
# (funziona sia in locale sia rimontato in Docker su Proxmox).
#
# Push  : curl -F file=@xxx.zip HOST/anonymize  -> zip elaborato (con intestazioni
#         X-Processing-* : stato, conteggi, errori).
# Pop   : il file scaricato E' l'archivio elaborato (le immagini anonimizzate).
#         Per la variante S3 usare /anonymize/s3 + /status/:job_id + download
#         dalla bucket output (vedi scripts/s3_tools.sh).
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File scripts\push-pop.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\push-pop.ps1 -Base http://192.168.1.50:8080 -Zip C:\test\wider_active_in.zip -Max 200
param(
    [string]$Base = "http://127.0.0.1:8080",
    [string]$Zip = (Join-Path (Split-Path -Parent $PSScriptRoot) "test-active\wider_active_in.zip"),
    [string]$Out = "result.zip",
    [int]$Max = $null,
    [switch]$KeepRunning
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

# Se il file non esiste, ripiega sull'archivio di test del test-active locale.
if (-not (Test-Path -LiteralPath $Zip)) {
    $cand = Join-Path $Root "test-active\wider_active_in.zip"
    if (-not (Test-Path -LiteralPath $cand)) { throw "Zip di input non trovato. Generane uno con scripts\test-wider.ps1 oppure passa -Zip." }
    $Zip = $cand
}

$MainZip = "$Zip"
if ($Max) {
    $MainZip = Join-Path (Split-Path $Zip) "pushpop_in.zip"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zin = [System.IO.Compression.ZipFile]::OpenRead($Zip)
    $entries = @($zin.Entries | Where-Object { $_.Name -match "\.(jpe?g|png)$" } | Select-Object -First $Max)
    $zout = [System.IO.Compression.ZipFile]::Open($MainZip, [System.IO.Compression.ZipArchiveMode]::Create)
    foreach ($e in $entries) {
        $src = $e.Open(); $dst = $zout.CreateEntry($e.Name).Open()
        $src.CopyTo($dst); $dst.Dispose(); $src.Dispose()
    }
    $zout.Dispose(); $zin.Dispose()
}

Write-Host "==> Health:"
try { $h = Invoke-WebRequest -Uri "$Base/health" -UseBasicParsing -TimeoutSec 5; Write-Host "    $($h.StatusCode)" } catch { Write-Host ("    K.O. {0}" -f $_.Exception.Message) -ForegroundColor Red }

Write-Host "==> PUSH $MainZip -> $Base/anonymize"
$marker = "$env:TEMP\pop_out_$PID"
& curl.exe -fsS -m 1800 -D - -F "file=@$MainZip" "$Base/anonymize" -o $Out | Select-String -Pattern "HTTP/|x-processing|content"
Write-Host ""
Write-Host "==> POP: salvato $Out ($(Get-Item $Out -ErrorAction SilentlyContinue).Length byte)"

$pop = Join-Path (Split-Path $Out) "pop_estratto"
Remove-Item $pop -Recurse -ErrorAction SilentlyContinue
Expand-Archive -LiteralPath $Out -DestinationPath $pop -ErrorAction SilentlyContinue
$img = Get-ChildItem -Path $pop -Recurse -File | Where-Object { $_.Name -match "\.(jpe?g|png)$" }
$err = Get-ChildItem -Path $pop -Recurse -File | Where-Object { $_.Name -match "_error\.txt$" }
Write-Host "Immagini elaborate estratte: $($img.Count)   file errore: $($err.Count)"