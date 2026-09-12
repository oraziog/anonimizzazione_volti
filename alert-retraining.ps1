# alert-retraining.ps1 - controlla l'esito del retraining notturno e genera un
# avviso (file di segnalazione) quando l'ultimo audit ha esito "failed" o
# "rejected". Pensato per girare dopo ogni finestra notturna (03:00 + margine):
# task pianificato "AnonVolt_RetrainAlert" alle 04:00, ogni giorno.
#
# Uso (una riga, da qualsiasi cartella):
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\alert-retraining.ps1
#   opzioni: -Stampa (mostra l'esito anche quando tutto va bene)
# Registra/elimina il task:
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\alert-retraining.ps1 -RegistraTask
#   powershell -NoProfile -ExecutionPolicy Bypass -Command "Unregister-ScheduledTask -TaskName AnonVolt_RetrainAlert -Confirm:0"

param(
    [switch] $Stampa,
    [switch] $RegistraTask
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if ($RegistraTask) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$root\alert-retraining.ps1`""
    $trigger = New-ScheduledTaskTrigger -Daily -At 04:00
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
    Register-ScheduledTask -TaskName "AnonVolt_RetrainAlert" -Action $action `
        -Trigger $trigger -Settings $settings `
        -Description "Alert se il retraining notturno fallisce o viene rifiutato" -Force | Out-Null
    Write-Host "Task pianificato 'AnonVolt_RetrainAlert' registrato (tutti i giorni 04:00)." -ForegroundColor Green
    exit 0
}

$dataDir = Join-Path $root "data"
$auditFile = Join-Path $dataDir "retrain_audit.json"
$alertDir  = Join-Path $root "alerts"
$alertFile = Join-Path $alertDir "RETRAINING-ATTENTION.txt"

if (-not (Test-Path -LiteralPath $auditFile)) {
    if ($Stampa) { Write-Host "Nessun audit del retraining presente (il retraining non ha mai girato)." }
    exit 0
}

$audit = Get-Content -LiteralPath $auditFile -Raw | ConvertFrom-Json
$status = $audit.status
$when = $audit.timestamp

if ($Stampa -or $status -in @("failed", "rejected")) {
    $acc = ""
    if ($null -ne $audit.rust_validation_accuracy) { $acc = " rust_acc=$($audit.rust_validation_accuracy)" }
    elseif ($null -ne $audit.python_val_accuracy) { $acc = " python_val_acc=$($audit.python_val_accuracy)" }
    Write-Host ("Retraining audit: status={0} ({1}){2}" -f $status, $when, $acc)
    if ($audit.reason) { Write-Host ("  motivo: " + $audit.reason) }
}

if ($status -notin @("failed", "rejected")) {
    # Esito ok (swapped/skipped): l'alert precedente, se esiste, non e' piu' attuale.
    if (Test-Path -LiteralPath $alertFile) {
        Remove-Item -LiteralPath $alertFile -Force
        Write-Host "Alert precedente rimosso (l'ultimo retraining non richiede piu' attenzione)."
    }
    exit 0
}

# Esito failed/rejected: scrivi (o aggiorna) il file di segnalazione.
New-Item -ItemType Directory -Force -Path $alertDir | Out-Null
$righe = @()
$righe += "==============================================================="
$righe += "RITENTA OPERATORE: retraining notturno con esito '$status'"
$righe += "Audit del: $when   (controllato il: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))"
$righe += "==============================================================="
$righe += "Motivo:"
$righe += "  $(if ($audit.reason) { $audit.reason } else { '-' })"
if ($null -ne $audit.python_val_accuracy) { $righe += "Accuracy Python:  $($audit.python_val_accuracy)  (minimo richiesto: $($audit.min_accuracy))" }
if ($null -ne $audit.rust_validation_accuracy) { $righe += "Accuracy Rust:    $($audit.rust_validation_accuracy)  (minimo richiesto: $($audit.min_accuracy))" }
$righe += "Campioni: $($audit.real_samples) volti reali, $($audit.fp_samples) falsi positivi"
if ($audit.candidate_onnx) { $righe += "Candidato scartato: $($audit.candidate_onnx)" }
$righe += ""
$righe += "Cosa controllare:"
$righe += "  1) seed troppo piccolo o di cattiva qualita': dataset_seed\real_faces\"
$righe += "  2) falsi positivi pochi/tutti uguali: dataset_falsi_positivi\"
$righe += "  3) soglia troppo severa: RETRAIN_MIN_ACCURACY in src\.env"
$righe += "  4) dettaglio completo: data\retrain_audit.json e logs\server.log"
$righe += ""
$righe += "Il servizio NON e' degradato: il classificatore precedente resta attivo"
$righe += "(o, al primo giro, le cam ACTIVE sfocano tutto in-ROI: sempre GDPR-safe)."

Set-Content -LiteralPath $alertFile -Value ($righe -join [Environment]::NewLine) -Encoding UTF8
Write-Host "ALERT scritto: $alertFile" -ForegroundColor Yellow
exit 0
