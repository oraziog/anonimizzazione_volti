# riassunto-log.ps1 - cerca nei log solo gli ERROR e le transizioni di stato
# delle telecamere (INITIAL/LEARNING/ACTIVE, estrazioni ROI, promozioni),
# con riepilogo raggruppato per giorno.
#
# Uso (una riga, da qualsiasi cartella):
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\riassunto-log.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\riassunto-log.ps1 -Giorni 7 -Dettaglio

param(
    [int] $Giorni = 30,
    [switch] $Dettaglio
)

$logsDir = Join-Path $PSScriptRoot "logs"
$file = Join-Path $logsDir "server.log"
if (-not (Test-Path -LiteralPath $file)) {
    Write-Host "Log non trovato: $file" -ForegroundColor Red
    exit 1
}

$cutoff = (Get-Date).AddDays(-$Giorni)
$righe = Get-Content -LiteralPath $file

# Pattern: timestamp ISO + livello + modulo + messaggio
$pattern = '^(?<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+(?<lvl>INFO|WARN|ERROR)\s+(?<mod>\S+):\s+(?<msg>.*)$'

$eventi = foreach ($r in $righe) {
    if ($r -match $pattern) {
        # Salva subito i campi: i -match successivi sovrascrivono $Matches.
        $ts = [datetime]::Parse($Matches.ts, [cultureinfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::AssumeUniversal -bor [System.Globalization.DateTimeStyles]::AdjustToUniversal)
        $lvl = $Matches.lvl
        $mod = $Matches.mod
        $msg = $Matches.msg
        if ($ts -ge $cutoff) {
            $interessa = $false
            if ($lvl -eq 'ERROR') { $interessa = $true }
            # Transizioni e promozioni FSM / ROI (usi le phrasing reali del servizio)
            if ($msg -match 'camera .+ (INITIAL|LEARNING|ACTIVE)') { $interessa = $true }
            if ($msg -match 'ROI (extracted|validation failed|re-adopted|kept|replaced)|not enough detection data|ROI finalization|to LEARNING') { $interessa = $true }
            if ($msg -match 'operator reset camera') { $interessa = $true }
            if ($interessa) {
                [pscustomobject]@{ Giorno = $ts.ToString('yyyy-MM-dd'); Livello = $lvl; Modulo = $mod; Messaggio = $msg; Ts = $ts }
            }
        }
    }
}

$eventi = @($eventi)
if ($eventi.Count -eq 0) {
    Write-Host "Nessun ERROR ne' transizione di stato negli ultimi $Giorni giorni."
    exit 0
}

Write-Host "=== ERROR e transizioni telecamere - ultimi $Giorni giorni ===" -ForegroundColor Cyan
$eventi | Group-Object Giorno | Sort-Object Name | ForEach-Object {
    $errCount = @($_.Group | Where-Object Livello -eq 'ERROR').Count
    Write-Host ("{0}: {1} evento/i ({2} ERROR)" -f $_.Name, $_.Count, $errCount) -ForegroundColor Yellow
    if ($Dettaglio) {
        $_.Group | Sort-Object Ts | ForEach-Object {
            Write-Host ("   {0} [{1}] {2}" -f $_.Ts.ToString('HH:mm:ss'), $_.Livello, $_.Messaggio)
        }
    } else {
        $_.Group | Sort-Object Ts | Select-Object -First 5 | ForEach-Object {
            Write-Host ("   {0} [{1}] {2}" -f $_.Ts.ToString('HH:mm:ss'), $_.Livello, $_.Messaggio)
        }
        if ($_.Count -gt 5) { Write-Host ("   ... e altri {0}" -f ($_.Count - 5)) }
    }
}
