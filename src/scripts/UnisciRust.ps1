# Script per unire tutti i file .rs in un unico file
# Salva questo script come "UnisciRust.ps1" nella cartella src

param(
    [string]$OutputFile = "Codice_rust_Unito.txt",
    [string]$RootDir = "."
)

# Ottieni tutti i file .rs ricorsivamente
$rustFiles = Get-ChildItem -Path $RootDir -Filter "*.rs" -Recurse | Sort-Object FullName

if ($rustFiles.Count -eq 0) {
    Write-Host "Nessun file .rs trovato!" -ForegroundColor Red
    exit
}

Write-Host "Trovati $($rustFiles.Count) file .rs da unire..." -ForegroundColor Green

# Crea il file di output (sovrascrive se esiste)
$outputPath = Join-Path $RootDir $OutputFile

# Usa un StreamWriter per prestazioni migliori
$writer = [System.IO.StreamWriter]::new($outputPath, $false, [System.Text.Encoding]::UTF8)

try {
    $fileCounter = 0
    $totalLines = 0
    $errorFiles = @()
    
    foreach ($file in $rustFiles) {
        $fileCounter++
        
        try {
            # Leggi il contenuto del file
            $content = Get-Content $file.FullName -Raw -ErrorAction Stop
            
            # Se il file è vuoto, usa una stringa vuota
            if ($null -eq $content) {
                $content = ""
            }
            
            # Calcola il numero di linee
            if ($content -eq "") {
                $lineCount = 0
            } else {
                $lineCount = $content.Split("`n").Count
            }
            
            # Scrivi l'intestazione
            $header = @"
// ======================================== 
// File: $($file.FullName)
// ========================================

"@
            
            $writer.WriteLine($header)
            $writer.WriteLine($content)
            
            # Aggiungi una linea vuota tra i file
            $writer.WriteLine()
            
            $totalLines += $lineCount
            Write-Host "Aggiunto: $($file.Name) ($lineCount linee)" -ForegroundColor Gray
        }
        catch {
            $errorFiles += $file.Name
            Write-Host "ERRORE durante la lettura di: $($file.Name) - $_" -ForegroundColor Red
            
            # Scrivi comunque l'intestazione con un messaggio di errore
            $header = @"
// ======================================== 
// File: $($file.FullName) - ERRORE DI LETTURA
// ========================================

"@
            $writer.WriteLine($header)
            $writer.WriteLine("// Impossibile leggere il contenuto del file")
            $writer.WriteLine()
        }
    }
    
    Write-Host "`nOperazione completata!" -ForegroundColor Green
    Write-Host "File creato: $outputPath" -ForegroundColor Yellow
    Write-Host "Totale file uniti: $fileCounter" -ForegroundColor Cyan
    Write-Host "Totale linee: $totalLines" -ForegroundColor Cyan
    
    if ($errorFiles.Count -gt 0) {
        Write-Host "`nFile con errori:" -ForegroundColor Red
        $errorFiles | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    }
}
finally {
    $writer.Dispose()
}

# Mostra le prime righe del file creato per verifica
if (Test-Path $outputPath) {
    Write-Host "`nAnteprima del file creato:" -ForegroundColor Green
    $previewLines = Get-Content $outputPath -Head 20 -ErrorAction SilentlyContinue
    if ($previewLines) {
        $previewLines | ForEach-Object { Write-Host $_ }
    } else {
        Write-Host "Il file creato è vuoto o non è stato possibile leggerlo." -ForegroundColor Yellow
    }
}