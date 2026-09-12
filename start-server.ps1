# start-server.ps1 - avvia il servizio di anonimizzazione con un doppio click
# (o da console: powershell -NoProfile -ExecutionPolicy Bypass -File .\start-server.ps1)
#
# Cosa fa:
#   1. carica le variabili di src\.env nel processo (il binario NON legge .env da solo)
#   2. lancia target\release\anonimizzazione_volti.exe come processo indipendente,
#      tramite logs\run-server.cmd (generato qui): redirezione stdout/stderr inclusa
#   3. log su logs\server.log (stdout) e logs\server.err (stderr); PID in logs\server.pid
#
# Per fermare il server:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\stop-server.ps1
#   (oppure doppio click su stop-server.cmd)

$ErrorActionPreference = "Stop"

$root    = $PSScriptRoot
$exe     = Join-Path $root "target\release\anonimizzazione_volti.exe"
$envFile = Join-Path $root "src\.env"
$loader  = Join-Path $root "src\scripts\load-env.ps1"
$logDir  = Join-Path $root "logs"
$outLog  = Join-Path $logDir "server.log"
$errLog  = Join-Path $logDir "server.err"
$pidFile = Join-Path $logDir "server.pid"
$runCmd  = Join-Path $logDir "run-server.cmd"
$srcDir  = Join-Path $root "src"
$port    = 8080   # deve corrispondere a BIND_ADDR in src\.env

if (-not (Test-Path -LiteralPath $exe)) {
    throw "Eseguibile non trovato: $exe - compila prima con: cargo build --release (dentro $root)"
}

# Evita doppioni: se c'e' gia' un'istanza, esce senza fare nulla.
$existing = Get-Process -Name "anonimizzazione_volti" -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Il server e' gia' in esecuzione (PID $($existing.Id -join ', '))." -ForegroundColor Yellow
    Write-Host "Per fermarlo: powershell -NoProfile -ExecutionPolicy Bypass -File .\stop-server.ps1"
    exit 0
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Ruota i log della run precedente (server.log -> server.previous.log,
# server.err -> server.previous.err)
if (Test-Path -LiteralPath $outLog) { Move-Item -Force -Path $outLog -Destination (Join-Path $logDir "server.previous.log") }
if (Test-Path -LiteralPath $errLog) { Move-Item -Force -Path $errLog -Destination (Join-Path $logDir "server.previous.err") }

# Genera il launcher: i percorsi con spazi restano dentro il file, nessun
# quoting complesso da riga di comando.
$nl = [Environment]::NewLine
Set-Content -Path $runCmd -Encoding ASCII -Value (
    "@echo off" + $nl +
    "cd /d ""$srcDir""" + $nl +
    """$exe"" > ""$outLog"" 2> ""$errLog""" + $nl
)

# Carica il .env nel processo corrente: l'exe lo ereditera' all'avvio.
& $loader -Path $envFile -Force

# Avvia il launcher nascosto (la finestra cmd resta invisibile).
Start-Process -FilePath $runCmd -WindowStyle Hidden | Out-Null
Write-Host "Avvio in corso..." -ForegroundColor Green

# Individua il PID del processo del server per stop-server.ps1 (max ~10s)
$serverPid = $null
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Milliseconds 500
    $srv = Get-Process -Name "anonimizzazione_volti" -ErrorAction SilentlyContinue
    if ($srv) { $serverPid = $srv[0].Id; break }
}
if ($serverPid) {
    Set-Content -Path $pidFile -Value $serverPid
    Write-Host "Server avviato con PID $serverPid." -ForegroundColor Green
} else {
    Write-Host "ATTENZIONE: processo del server non individuato." -ForegroundColor Yellow
}

# Attende che la porta risponda (test TCP diretto, max ~45s): una connessione
# accettata significa che il server e' online, a prescindere dalla risposta HTTP.
$ok = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Milliseconds 500
    if (-not (Get-Process -Name "anonimizzazione_volti" -ErrorAction SilentlyContinue)) { break }
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync("127.0.0.1", $port)
        if ($task.Wait(800) -and $client.Connected) { $ok = $true; break }
    } catch { } finally { $client.Close() }
}

if ($ok) {
    Write-Host "Server ONLINE su http://localhost:$port" -ForegroundColor Green
    Write-Host "Log: $outLog"
    Write-Host "Test rapido (una riga, da qualsiasi cartella):"
    Write-Host ('  curl -X POST http://localhost:8080/anonymize -F "file=@C:\Users\Admin\Music\anonimizzazione_volti\test_input.zip" -o C:\Users\Admin\Music\anonimizzazione_volti\test_zip\my_output.zip')
} elseif (-not (Get-Process -Name "anonimizzazione_volti" -ErrorAction SilentlyContinue)) {
    Write-Host "Il processo e' terminato subito. Ultime righe dell'errore:" -ForegroundColor Red
    Get-Content $errLog -Tail 15 -ErrorAction SilentlyContinue
    Get-Content $outLog -Tail 15 -ErrorAction SilentlyContinue
} else {
    Write-Host "Il server non risponde ancora su http://localhost:$port (controlla $outLog)" -ForegroundColor Yellow
}
