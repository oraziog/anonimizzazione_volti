# stop-server.ps1 - ferma il server avviato con start-server.ps1
# (o da console: powershell -NoProfile -ExecutionPolicy Bypass -File .\stop-server.ps1)

$ErrorActionPreference = "SilentlyContinue"

$pidFile = Join-Path $PSScriptRoot "logs\server.pid"
$stopped = $false

if (Test-Path -LiteralPath $pidFile) {
    $p = Get-Content $pidFile -ErrorAction SilentlyContinue
    if ($p -match '^\d+$') {
        $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
        if ($proc -and $proc.ProcessName -eq "anonimizzazione_volti") {
            Stop-Process -Id $p -Force
            $stopped = $true
            Write-Host "Server fermato (PID $p)." -ForegroundColor Green
        }
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

if (-not $stopped) {
    $any = Get-Process -Name "anonimizzazione_volti" -ErrorAction SilentlyContinue
    if ($any) {
        $any | Stop-Process -Force
        Write-Host "Server fermato." -ForegroundColor Green
    } else {
        Write-Host "Nessun server in esecuzione." -ForegroundColor Yellow
    }
}
