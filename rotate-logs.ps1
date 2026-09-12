# rotate-logs.ps1 - comprime i log vecchi della cartella logs\ in un archivio
# zip settimanale e mantiene solo gli ultimi 4 zip (cancellando i piu' vecchi).
#
# Uso (una riga, da qualsiasi cartella):
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\rotate-logs.ps1
#
# Rotazione automatica settimanale (lunedi' 07:30) gia' registrata come task
# "AnonVolt_RotateLogs": per eliminarla:
#   powershell -NoProfile -ExecutionPolicy Bypass -Command "Unregister-ScheduledTask -TaskName AnonVolt_RotateLogs -Confirm:0"
# per riregistrarla:
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\rotate-logs.ps1 -RegistraTask

param(
    [switch] $RegistraTask
)

$ErrorActionPreference = "Stop"
$logsDir = Join-Path $PSScriptRoot "logs"
$keep = 4

if ($RegistraTask) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\rotate-logs.ps1`""
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 07:30
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
    Register-ScheduledTask -TaskName "AnonVolt_RotateLogs" -Action $action `
        -Trigger $trigger -Settings $settings -Description "Rotazione settimanale dei log di Anonimizzazione Volti" -Force | Out-Null
    Write-Host "Task pianificato 'AnonVolt_RotateLogs' registrato (lunedi' 07:30)." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path -LiteralPath $logsDir)) {
    Write-Host "Cartella log non trovata: $logsDir" -ForegroundColor Red
    exit 1
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

# File candidati alla rotazione: log e pid NON attivi (il server scrive solo su
# server.log / server.err correnti; tutto il resto si puo' archiviare).
$candidati = Get-ChildItem -LiteralPath $logsDir -File |
    Where-Object { $_.Name -match '\.(log|err|pid)$' -and $_.Name -notin @('server.log', 'server.err', 'server.pid') }

if (-not $candidati) {
    Write-Host "Nessun log vecchio da ruotare."
    exit 0
}

$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$zipPath = Join-Path $logsDir "logs-$stamp.zip"
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($f in $candidati) {
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $f.Name, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
} finally {
    $zip.Dispose()
}

foreach ($f in $candidati) { Remove-Item -LiteralPath $f.FullName -Force }

# Conserva solo gli ultimi $keep archivi.
$vectimi = Get-ChildItem -LiteralPath $logsDir -Filter "logs-*.zip" |
    Sort-Object LastWriteTime -Descending | Select-Object -Skip $keep
foreach ($v in $vectimi) {
    Remove-Item -LiteralPath $v.FullName -Force
    Write-Host "Archivio obsoleto cancellato: $($v.Name)" -ForegroundColor Yellow
}

$sizeKb = [math]::Round((Get-Item -LiteralPath $zipPath).Length / 1KB, 1)
Write-Host "Rotazione completata: $($candidati.Count) file -> $(Split-Path -Leaf $zipPath) ($sizeKb KB), conservati gli ultimi $keep archivi." -ForegroundColor Green
