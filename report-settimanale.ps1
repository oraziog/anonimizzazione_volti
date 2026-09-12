# report-settimanale.ps1 - trend settimanale dell'accuracy del classificatore.
# Accumula (deduplicati per timestamp) tutti i punti della storia:
#   models\cache\metrics_*.json          -> accuracy del candidato (Python)
#   data\classifier_state.json           -> accuracy del modello attivo (Rust)
# e scrive/aggiorna UN SOLO grafico Markdown per settimana in reports\:
#   reports\trend-accuracy-AAAA-Www.md
#
# Uso (una riga, da qualsiasi cartella):
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\report-settimanale.ps1
#   opzioni: -Stampa (mostra il grafico anche a terminale)
# Task settimanale (lunedi' 08:10):
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\report-settimanale.ps1 -RegistraTask
#   powershell -NoProfile -ExecutionPolicy Bypass -Command "Unregister-ScheduledTask -TaskName AnonVolt_TrendAccuracy -Confirm:0"

param(
    [switch] $Stampa,
    [switch] $RegistraTask
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if ($RegistraTask) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$root\report-settimanale.ps1`""
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 08:10
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
    Register-ScheduledTask -TaskName "AnonVolt_TrendAccuracy" -Action $action `
        -Trigger $trigger -Settings $settings `
        -Description "Trend settimanale accuracy classificatore (Markdown in reports\)" -Force | Out-Null
    Write-Host "Task pianificato 'AnonVolt_TrendAccuracy' registrato (lunedi' 08:10)." -ForegroundColor Green
    exit 0
}

$punti = @()

# 1) candidati Python: models\cache\metrics_*.json
$metricsFiles = Get-ChildItem -LiteralPath (Join-Path $root "models\cache") -Filter "metrics_*.json" -ErrorAction SilentlyContinue
foreach ($m in $metricsFiles) {
    try {
        $j = Get-Content -LiteralPath $m.FullName -Raw | ConvertFrom-Json
        if ($null -ne $j.val_accuracy -and $j.exported_at) {
            $punti += [pscustomobject]@{
                quando = [datetime]$j.exported_at
                tipo   = "candidato (Python)"
                acc    = [double]$j.val_accuracy
                nota   = ("real {0} / fp {1}" -f $j.real_samples_used, $j.fp_samples_used)
            }
        }
    } catch { }
}

# 2) modello attivo validato Rust: data\classifier_state.json
$stateFile = Join-Path $root "data\classifier_state.json"
$modelloAttivo = "-"
if (Test-Path -LiteralPath $stateFile) {
    try {
        $s = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        $punti += [pscustomobject]@{
            quando = [datetime]$s.updated_at
            tipo   = "attivo (Rust)"
            acc    = [double]$s.accuracy
            nota   = (Split-Path -Leaf $s.active_onnx)
        }
        $modelloAttivo = Split-Path -Leaf $s.active_onnx
    } catch { }
}

# 3) ultimo esito retraining (per la riga di stato)
$ultimoEsito = "n/d"
$auditFile = Join-Path $root "data\retrain_audit.json"
if (Test-Path -LiteralPath $auditFile) {
    try {
        $a = Get-Content -LiteralPath $auditFile -Raw | ConvertFrom-Json
        $ultimoEsito = "{0} ({1})" -f $a.status, ([datetime]$a.timestamp).ToLocalTime().ToString("yyyy-MM-dd HH:mm")
    } catch { }
}

if ($punti.Count -eq 0) {
    Write-Host "Nessun dato di accuracy disponibile (nessun retraining ancora eseguito)."
    exit 0
}

# Deduplica (stesso istante + stesso tipo) e ordine cronologico
$punti = @($punti | Sort-Object quando, tipo -Unique)

$adesso = Get-Date
$cal = $adesso
$weekNum = [System.Globalization.CultureInfo]::InvariantCulture.Calendar.GetWeekOfYear(
    $cal, [System.Globalization.CalendarWeekRule]::FirstFourDayWeek, [DayOfWeek]::Monday)
$settimana = "{0}-W{1:d2}" -f $cal.Year, $weekNum

# Grafico: barra 0..100% su 30 caratteri
$righe = @()
$righe += "# Trend accuracy classificatore - settimana $settimana"
$righe += ""
$righe += "_Aggiornato: $($adesso.ToString('yyyy-MM-dd HH:mm')) | modello attivo: $modelloAttivo | ultimo retraining: $ultimoEsito | minimo swap: 85%_"
$righe += ""
$righe += '```text'
$righe += "accuracy  | grafico (0%..100%)                    | quando (locale)  | tipo / dettaglio"
$righe += "----------|---------------------------------------|------------------|------------------"
foreach ($p in $punti) {
    $barLen = [int][math]::Round($p.acc * 30)
    $bar = "#" * [math]::Max(1, [math]::Min(30, $barLen))
    $righe += ("{0,7:P1} | {1,-30} | {2} | {3} - {4}" -f $p.acc, $bar,
        $p.quando.ToLocalTime().ToString("yyyy-MM-dd HH:mm"), $p.tipo, $p.nota)
}
$righe += '```'
$righe += ""
$righe += ("Punti totali nella storia: **{0}** (da {1} a {2})." -f $punti.Count,
    ($punti | Select-Object -First 1).quando.ToLocalTime().ToString("yyyy-MM-dd"),
    ($punti | Select-Object -Last 1).quando.ToLocalTime().ToString("yyyy-MM-dd"))
if ($ultimoEsito -match "^(failed|rejected)") {
    $righe += ""
    $righe += "> ATTENZIONE: ultimo retraining con esito $($ultimoEsito.Split(' ')[0]) - vedi alerts\RETRAINING-ATTENTION.txt"
}

$reportDir = Join-Path $root "reports"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$file = Join-Path $reportDir ("trend-accuracy-" + $settimana + ".md")
Set-Content -LiteralPath $file -Value ($righe -join [Environment]::NewLine) -Encoding UTF8

if ($Stampa) { $righe | ForEach-Object { Write-Host $_ } }
Write-Host "Trend scritto: $file" -ForegroundColor Green
