# load-env.ps1 - carica un file .env (variabili chiave=valore) nell'ambiente
# della sessione PowerShell corrente. Script GENERICO: nessun percorso del
# progetto, funziona da qualunque directory. Pensato per gli utenti Windows
# che non vogliono esportare ogni campo a mano (il binario Rust NON legge .env
# da solo: le variabili vanno nell'ambiente del processo che lo avvia).
#
# Uso (PowerShell, da qualsiasi posizione):
#   . .\scripts\load-env.ps1                        # carica .\...\sunse
#   . ".\scripts\load-env.ps1" -Path .\.env         # file esplicito
#   . ".\scripts\load-env.ps1" -Path .\.env -Prefix APP_
#   . ".\scripts\load-env.ps1" -Force               # sovrascrive le variabili
#   . ".\scripts\load-env.ps1" -Show                # stampa le chiavi caricate
#
# NB: si usa la dot-source (la sintassi ". <percorso>") perche' cosi' le
# variabili finiscono nella SESSIONE corrente e restano disponibili al
# comando successivo (es. l'avvio del server). Se lo eseguite con & da solo,
# le variabili morirebbero con il child process.
#
# Formato riconosciuto (stesso di .env.example):
#   KEY=value          -> chiave semplice
#   KEY="value"        -> virgolette doppie tolte
#   KEY='value'        -> virgolette singole tolte
#   KEY=value # comment -> commento dopo lo spazio rimosso
#   # commento          -> riga ignorata
#   la riga dentro e' l'ultima; un = mancante o chiave invalida -> riga ignorata.
# Script volutamente ASCII-only (PowerShell 5.1 legge i file senza BOM come
# ANSI; i caratteri non-ASCII nei commenti romperebbero il parsing).

param(
    [string]$Path = ".env",
    [string]$Prefix = "",
    [switch]$Force,
    [switch]$Show
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Path)) {
    throw "File .env non trovato: $Path"
}

$global:envLoad = @{}
foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
    $t = $line.Trim()
    if ([string]::IsNullOrEmpty($t) -or $t.StartsWith("#")) { continue }
    $i = $t.IndexOf("=")
    if ($i -lt 0) { continue }
    $key = $t.Substring(0, $i).Trim()
    if ($key -notmatch "^[A-Za-z_][A-Za-z0-9_]*$") { continue }
    $val = $t.Substring($i + 1).Trim()
    # Commento finale (solo se preceduto da spazio) e virgolette circostanti.
    if ($val.Length -ge 2 -and $val.StartsWith("`"") -and $val.EndsWith("`"")) {
        $val = $val.Substring(1, $val.Length - 2)
    } elseif ($val.Length -ge 2 -and $val.StartsWith("'") -and $val.EndsWith("'")) {
        $val = $val.Substring(1, $val.Length - 2)
    } else {
        $c = $val.IndexOf(" #")
        if ($c -ge 0) { $val = $val.Substring(0, $c).TrimEnd() }
        $val = $val.Trim()
    }
    $target = "${Prefix}${key}"
    $already = [Environment]::GetEnvironmentVariable($target, "Process") -ne $null
    if (-not $already -or $Force) {
        Set-Item -Path "Env:$target" -Value $val
    }
    $global:envLoad[$target] = $val
}

Write-Host "==> .env caricato: $($global:envLoad.Count) variabili (prefix '$Prefix')" -ForegroundColor Green
if ($Show) {
    foreach ($k in ($global:envLoad.Keys | Sort-Object)) {
        $v = $global:envLoad[$k]
        if ($k -match "KEY|SECRET|PASS|TOKEN") { $v = "(nascosto)" }
        Write-Host ("    {0}={1}" -f $k, $v)
    }
}