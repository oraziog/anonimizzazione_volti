# report-camere.ps1 - report giornaliero dello stato delle telecamere:
# legge /operator/cameras e /operator/jobs dal server e scrive un report
# leggibile in reports\camere-YYYY-MM-DD.txt (una sezione per ogni esecuzione).
#
# Uso (una riga, da qualsiasi cartella):
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\report-camere.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\report-camere.ps1 -Stampa
#
# Report automatico ogni giorno (08:00) gia' registrabile come task
# "AnonVolt_ReportCamere":
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\report-camere.ps1 -RegistraTask
# per eliminarlo:
#   powershell -NoProfile -ExecutionPolicy Bypass -Command "Unregister-ScheduledTask -TaskName AnonVolt_ReportCamere -Confirm:0"

param(
    [string] $ServerUrl = "http://localhost:8080",
    [switch] $Stampa,
    [switch] $RegistraTask
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# Chiave operatore: letta dal .env (riga OPERATOR_API_KEY=...), mai scritta nel report.
$envFile = Join-Path $root "src\.env"
$key = $null
if (Test-Path -LiteralPath $envFile) {
    $line = Get-Content -LiteralPath $envFile | Where-Object { $_ -match '^OPERATOR_API_KEY=(.+)\s*$' } | Select-Object -First 1
    if ($line -match '^OPERATOR_API_KEY=(.+)\s*$') { $key = $Matches[1].Trim() }
}
if (-not $key) {
    Write-Host "OPERATOR_API_KEY non trovata in $envFile" -ForegroundColor Red
    exit 1
}

if ($RegistraTask) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$root\report-camere.ps1`""
    $trigger = New-ScheduledTaskTrigger -Daily -At 08:00
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
    Register-ScheduledTask -TaskName "AnonVolt_ReportCamere" -Action $action `
        -Trigger $trigger -Settings $settings -Description "Report giornaliero stato telecamere Anonimizzazione Volti" -Force | Out-Null
    Write-Host "Task pianificato 'AnonVolt_ReportCamere' registrato (tutti i giorni 08:00)." -ForegroundColor Green
    exit 0
}

$h = @{ "X-Operator-Key" = $key }

# --- 1. Telecamere -----------------------------------------------------------
try {
    $camere = Invoke-RestMethod -Uri "$ServerUrl/operator/cameras" -Headers $h -TimeoutSec 15
} catch {
    Write-Host "Server non raggiungibile su $ServerUrl ($_ )" -ForegroundColor Red
    exit 1
}

# --- 2. Job recenti -----------------------------------------------------------
$jobs = $null
try {
    $jobs = Invoke-RestMethod -Uri "$ServerUrl/operator/jobs?limit=20" -Headers $h -TimeoutSec 15
} catch {
    $jobs = $null  # non blocca il report
}

$adesso = Get-Date
$perStato = $camere | Group-Object state

$righe = @()
$righe += "==================================================================="
$righe += "REPORT TELECAMERE - " + $adesso.ToString("yyyy-MM-dd HH:mm:ss")
$righe += "Server: $ServerUrl"
$righe += "==================================================================="
$righe += ""
$righe += "TELECAMERE: $($camere.Count) totali  ( " + (($perStato | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join "  ") + " )"
$righe += ""
$righe += "ID               STATO      LEARNING DAL          ROI      FRAME"
foreach ($c in $camere) {
    $learning = if ($c.learning_started_at) { ([datetime]$c.learning_started_at).ToLocalTime().ToString("yyyy-MM-dd HH:mm") } else { "-" }
    $roi = if ($c.roi) { "si" } else { "no" }
    $frame = if ($c.frame_width) { "$($c.frame_width)x$($c.frame_height)" } else { "?" }
    $righe += ("{0,-16} {1,-10} {2,-20} {3,-8} {4}" -f $c.id, $c.state, $learning, $roi, $frame)
}
$righe += ""

if ($jobs) {
    $lista = if ($jobs.jobs) { $jobs.jobs } elseif ($jobs -is [array]) { $jobs } else { $null }
    if ($lista) {
        $righe += "ULTIMI JOB (max 20):"
        $righe += "ARCHIVIO                              IMG      ERRORI    DURATA(s)  QUANDO"
        foreach ($j in ($lista | Select-Object -First 20)) {
            $nome = $j.input; if (-not $nome) { $nome = "?" }
            $img = $j.processed_images; if ($null -eq $img) { $img = "?" }
            $err = $j.error_count; if ($null -eq $err) { $err = "?" }
            $dur = ""
            if ($j.duration_ms) { $dur = [math]::Round($j.duration_ms / 1000.0, 1) }
            $quando = ""
            if ($j.finished_at) { $quando = ([datetime]$j.finished_at).ToLocalTime().ToString("yyyy-MM-dd HH:mm") }
            elseif ($j.started_at) { $quando = ([datetime]$j.started_at).ToLocalTime().ToString("yyyy-MM-dd HH:mm") }
            $righe += ("{0,-38} {1,-8} {2,-9} {3,-10} {4}" -f ([string]$nome), $img, $err, $dur, $quando)
        }
        $righe += ""
    }
}

# --- 3. Audit retraining ------------------------------------------------------
try {
    $audit = Invoke-RestMethod -Uri "$ServerUrl/operator/retrain-audit" -Headers $h -TimeoutSec 15
    if ($audit -and -not $audit.error) {
        $righe += "RETRAIN AUDIT:"
        $quando = if ($audit.finished_at) { ([datetime]$audit.finished_at).ToLocalTime().ToString("yyyy-MM-dd HH:mm") } elseif ($audit.timestamp) { ([datetime]$audit.timestamp).ToLocalTime().ToString("yyyy-MM-dd HH:mm") } else { "?" }
        $esito = if ($audit.outcome) { $audit.outcome } elseif ($audit.status) { $audit.status } else { "?" }
        $acc = if ($null -ne $audit.accuracy) { $audit.accuracy } elseif ($null -ne $audit.metrics) { $audit.metrics } else { "?" }
        $righe += "  ultima esecuzione: $quando   esito: $esito   metriche: $acc"
        foreach ($p in ($audit.PSObject.Properties | Where-Object { $_.Name -notin @('finished_at','timestamp','outcome','status','accuracy','metrics') })) {
            $v = if ($null -ne $p.Value) { $p.Value } else { "-" }
            $righe += "  $($p.Name): $v"
        }
        $righe += ""
    } else {
        $righe += "RETRAIN AUDIT: mai eseguito (il retraining notturno non ha ancora prodotto un audit)"
        $righe += ""
    }
} catch {
    $code = $null
    try { $code = [int]$_.Exception.Response.StatusCode } catch { }
    if ($code -eq 404) {
        $righe += "RETRAIN AUDIT: mai eseguito (nessun audit registrato finora)"
    } else {
        $msg = $_.Exception.Message
        try { $errObj = $_.ErrorDetails.Message | ConvertFrom-Json; if ($errObj.error) { $msg = $errObj.error } } catch { }
        $righe += "RETRAIN AUDIT: $msg"
    }
    $righe += ""
}

# --- 4. Salute classificatore -------------------------------------------------
try {
    $clf = Invoke-RestMethod -Uri "$ServerUrl/operator/classifier" -Headers $h -TimeoutSec 15
    $stato = if ($clf.loaded) { "CARICATO" } else { "non caricato" }
    $righe += "CLASSIFICATORE: $stato"
    $modello = if ($clf.active_onnx) { $clf.active_onnx } elseif ($clf.persisted) { $clf.persisted } else { "-" }
    $righe += "  modello attivo : $modello"
    $righe += "  enforce        : $($(if ($clf.classifier_enforce) {'ON'} else {'OFF'}))   soglia conferma: $($clf.confirm_threshold)   min accuracy swap: $([math]::Round($clf.min_accuracy_for_swap,2))"
    if ($clf.inference) {
        $righe += "  inference      : $($clf.inference.count) valutazioni, media $([math]::Round($clf.inference.avg_ms,1)) ms"
    }
    if (-not $clf.loaded) { $righe += "  (le cam ACTIVE sfocano ogni detection in-ROI finche' il classificatore non e' caricato)" }
    $righe += ""
} catch {
    $righe += "CLASSIFICATORE: non disponibile ($($_.Exception.Message))"
    $righe += ""
}

# --- 5. Storico accuracy del classificatore -----------------------------------
# Fonti: models\cache\metrics_*.json (accuracy Python del candidato ad ogni
# retraining) e data\classifier_state.json (accuracy validata Rust dell'ultimo
# swap). Grafico testuale: '#' piu' lunghi = accuracy piu' alta.
$storia = @()
$metricsFiles = Get-ChildItem -LiteralPath (Join-Path $root "models\cache") -Filter "metrics_*.json" -ErrorAction SilentlyContinue
foreach ($m in $metricsFiles) {
    try {
        $j = Get-Content -LiteralPath $m.FullName -Raw | ConvertFrom-Json
        if ($null -ne $j.val_accuracy) {
            $storia += [pscustomobject]@{
                quando = $j.exported_at
                fonte  = "candidato (Python)"
                acc    = [double]$j.val_accuracy
                extra  = ("real {0} / fp {1}" -f $j.real_samples_used, $j.fp_samples_used)
            }
        }
    } catch { }
}
$stateFile = Join-Path $root "data\classifier_state.json"
if (Test-Path -LiteralPath $stateFile) {
    try {
        $s = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        $storia += [pscustomobject]@{
            quando = $s.updated_at
            fonte  = "attivo (validato Rust)"
            acc    = [double]$s.accuracy
            extra  = (Split-Path -Leaf $s.active_onnx)
        }
    } catch { }
}
$storia = @($storia | Where-Object { $_.quando } | Sort-Object quando)
if ($storia.Count -gt 0) {
    $storia = @($storia | Select-Object -Last 12)
    $righe += "STORICO ACCURACY CLASSIFICATORE (ultimi $($storia.Count), dal piu' vecchio):"
    $stats = $storia | Measure-Object -Property acc -Minimum -Maximum
    $minA = [math]::Floor($stats.Minimum * 20) / 20
    $maxA = [math]::Ceiling($stats.Maximum * 20) / 20
    if (($maxA - $minA) -lt 0.05) { $minA = [math]::Max(0.0, $maxA - 0.05) }
    $span = [math]::Max($maxA - $minA, 0.0001)
    $righe += ("  scala: {0:P1} - {1:P1}   (minimo per lo swap: 85,0%)" -f $minA, $maxA)
    foreach ($p in $storia) {
        $q = "?"
        try { $q = ([datetime]$p.quando).ToLocalTime().ToString("yyyy-MM-dd HH:mm") } catch { }
        $barLen = [int][math]::Round((($p.acc - $minA) / $span) * 40)
        $barLen = [math]::Max(1, [math]::Min(40, $barLen))
        $righe += ("  {0}  {1,7:P1}  {2,-24} {3,-28} {4}" -f $q, $p.acc, $p.fonte, $p.extra, ("#" * $barLen))
    }
    $righe += ""
}

# Avvisi operativi (heuristiche utili)
$avvisi = @()
$alertPresente = Join-Path $root "alerts\RETRAINING-ATTENTION.txt"
if (Test-Path -LiteralPath $alertPresente) {
    $avvisi += "ALERT retraining attivo: leggere alerts\RETRAINING-ATTENTION.txt (esito failed/rejected)"
}
foreach ($c in $camere) {
    if ($c.state -eq "LEARNING" -and $c.learning_started_at) {
        $giorni = ((Get-Date) - ([datetime]$c.learning_started_at)).TotalDays
        if ($giorni -ge 1) {
            $avvisi += "Camera $($c.id): in LEARNING da $([math]::Round($giorni,1)) giorni (LEARNING_DAYS corrente: vedi .env)"
        }
    }
}
if ($avvisi.Count -gt 0) {
    $righe += "AVVISI:"
    foreach ($a in $avvisi) { $righe += "  - $a" }
    $righe += ""
}

$reportDir = Join-Path $root "reports"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$file = Join-Path $reportDir ("camere-" + $adesso.ToString("yyyy-MM-dd") + ".txt")
Add-Content -LiteralPath $file -Value ($righe -join [Environment]::NewLine) -Encoding UTF8

if ($Stampa) { $righe | ForEach-Object { Write-Host $_ } }
Write-Host "Report scritto: $file" -ForegroundColor Green
