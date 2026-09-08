# Archivi ZIP / 7z / RAR: pitfall e comportamento del servizio

Questa pagina elenca i casi "strani" che si incontrano con archivi reali prodotti
da strumenti diversi (Windows, 7-Zip, WinRAR, macOS, Python…) e cosa fa il
servizio in ciascun caso. Riferimenti: `src/zip_worker.rs`, endpoint
`POST /anonymize`.

## Formati accettati

| Estensione | Come viene elaborato |
|---|---|
| `.zip` | Letto interamente in RAM (mai estratto su disco); decompressione per-entry con decoder indipendenti come fallback. |
| `.7z` | Decompresso da `sevenz-rust` (puro Rust, LZMA/LZMA2) in una cartella temporanea sotto `DATA_DIR`, poi pipeline identica allo ZIP; la cartella viene rimossa a fine job. |
| `.rar` | Decompresso da libunrar (sorgente C++ compilata dentro il binario — nessuna dipendenza di sistema) in una cartella temporanea sotto `DATA_DIR`, poi rimossa. RAR4 e RAR5, inclusi archivi solidi. |
| altro | **400** con messaggio chiaro (il multipart viene comunque letto e scartato). |

Il campo multipart deve chiamarsi `file`; l'estensione del nome file decide il
formato.

## Pitfall più comuni

### 1. Compressione LZMA (metodo ZIP 14) — es. ZIP creati con 7-Zip/Python

Il problema storico: il decoder puro-Rust del crate `zip` su certi flussi
reali si disallinea all'inizio dello stream e fallisce con
`LzmaError("LZ distance 1 is beyond output size 0")` (le entry vengono
scartate). **Comportamento del servizio:** l'entry viene riletta grezza e
ritentata con **liblzma** (`xz2`), accettando il risultato solo se **CRC32 e
dimensione coincidono** con i metadati dell'entry. Nel log compare una riga
`recovered with independent decoder`. Se anche liblzma fallisce, l'entry è
segnalata in `X-Processing-Errors-Detail` e in `<input>_error.txt`.

### 2. bzip2 / zstd / XZ / deflate64

Metodi non-Deflate usati da vari strumenti. Il servizio li decodifica con i
decoder del crate `zip` (libbz2, libzstd, liblzma); se la decodifica in-crate
fallisce, applica lo **stesso fallback** del caso LZMA: rilettura grezza +
decoder indipendente + verifica CRC32/dimensione.

### 3. Archivi solidi

Negli archivi solidi (7z/rar) ogni file dipende dal contenuto dei precedenti:
non si può "saltare" una entry. Il servizio decompone l'intero archivio in
ordine (estrazione sequenziale nel file temporaneo) e poi processa le
immagini in **parallelo** — la decompressione resta veloce, il collo di
bottiglia (inferenza YOLO) è parallelizzato come per gli ZIP.

### 4. Archivi protetti da password / crittografati

Non supportati: l'estrazione fallisce e il job viene **rifiutato con 400**
(nessuna immagine elaborata a metà). Nel log compare il motivo
(`invalid 7z archive` / `cannot read RAR header`). Non esiste un modo sicuro
di chiedere la password via API, quindi il consiglio è di rimuovere la
password prima dell'upload.

### 5. Cartelle annidate

La grammatica accettata per le immagini:

```
CAM_001/foto.jpg            (cartella = camera)
uploads/2024/CAM_007/x.jpeg (cartella foglia = camera)
CAM_001_foto.jpg            (prefisso CAM_xxx_ = camera)
```

Qualunque percorso che non rientri in queste forme (es. `a/b/c/foto.jpg` con
cartella intermedia non-camera, file dentro una cartella-camera con estensione
non immagine) è un errore per-entry: l'immagine non viene elaborata, la entry
compare in `_error.txt`, il resto del job prosegue.

### 6. Entry pericolose (path traversal)

Nomi come `../CAM_001/evil.jpg` o `C:\evil.jpg` vengono **rifiutati**: per
gli ZIP la classificazione li scarta; per 7z/rar l'estrazione ricostruisce i
percorsi segmento per segmento e abortisce l'intero job se incontra `..` o
`:` — nessun file può mai essere scritto fuori dalla cartella temporanea.

### 7. File non immagine dentro l'archivio

`.txt`, `.DS_Store`, `._foto.jpg` (macOS), log di servizio: scartati e
contati in `X-Processing-Errors`, mai un abort. I file non immagine dentro una
cartella-camera (es. `CAM_001/readme.txt`) sono anch'essi errori per-entry.

### 8. Archivi molto grandi / decine di migliaia di file

- L'upload viene **spoolato su disco** (mai in RAM) e l'output è scritto in
  **streaming su disco** (`DATA_DIR`) e rispedito al client entry per entry:
  la memoria scala con i buffer dei task concorrenti (semaforo =
  auto-detected: core ≤ 4 → core, core > 4 → core − 2), non con la
  dimensione di input/output.
- 7z/rar: la decompressione su disco è sequenziale ma veloce; il
  processamento è parallelo.
- L'output è sempre uno ZIP **Stored** (zero compressione): si evita il costo
  CPU della ricompressione di decine di migliaia di JPEG.
- Per volumi oltre `BODY_LIMIT_BYTES` (3,5 GB di default) esiste
  `POST /anonymize/batch`: più campi archivio nella stessa richiesta (o
  upload chunked), processati in sequenza senza limite di corpo, risposta in
  un unico `batch_elaborato.zip`.
- Gli output accumulati in `DATA_DIR` non riempiono il disco: un task
  periodico applica la **retention** (`RETENTION_MAX_DAYS` per età e/o
  `RETENTION_MAX_GB` per spazio, eliminando i più vecchi per primi;
  disattivabile con `RETENTION_ENABLED=false`). Gli output più recenti di
  `RETENTION_MIN_AGE_SECS` non vengono mai toccati (potrebbero essere in
  scrittura o in streaming verso un client).

### 9. Archivi multiparte (.part01.rar, .z01…) e volumi

Non supportati (serve l'intero set di volumi). Il file caricato viene
elencato/estratto come archivio singolo; se mancano i volumi l'estrazione
fallisce con un errore chiaro.

### 10. ZIP "pieni" di duplicati

Ogni entry con lo stesso nome viene scritta in sequenza nell'output (l'ultima
vince nei lettori che sovrascrivono). Nessun dedup automatico: la FSM per
camera accumula le detection di tutte le immagini, quindi i duplicati non
influenzano la correttezza dell'anonimizzazione.

In `/anonymize/batch`, invece, il merge di più archivi **deduplica** le entry
con lo stesso nome (vince la prima copia, con un warning nel log): è lo
scenario normale del doppio upload dello stesso archivio camera, che
altrimenti farebbe fallire l'intero batch (`Duplicate filename` dello scrittore
ZIP).

## Cosa riceve il chiamante in caso di errori

Risposta 200 (il job non abortisce mai per entry singole) con:

```
X-Processing-Errors: 2
X-Processing-Errors-Detail: %5B%7B%22entry%22%3A%22note.txt%22...%5D
```

- `X-Processing-Errors-Detail` = JSON percent-encoded (decodifica con
  `decodeURIComponent`), prime 50 entry: `[{"entry":"note.txt","error":"..."}]`
- `<input>_error.txt` dentro lo ZIP di output, es. `Mio_Test.zip` →
  `Mio_Test_error.txt`:

```
Errori durante l'elaborazione di 'Mio_Test.zip' — 2 file scartati o degradati:

note.txt: unsupported entry (not an image in a camera layout)
CAM_001/readme.txt: unsupported entry (not an image in a camera layout)
```

- Se l'intero archivio è illegibile (password, formato sconosciuto, volume
  mancante) la risposta è **400** con `{"error": "..."}`.

## Verifica rapida

```bash
# ZIP con LZMA (come fa 7-Zip) + file non immagine
curl -fsS -D - -F "file=@Mio_Test.zip" http://localhost:8080/anonymize \
  -o Mio_Test_elaborato.zip | grep -i "x-processing-errors"
# il dettaglio, decodificato:
python -c "import urllib.parse,sys; h=[l for l in open(0) if l.lower().startswith('x-processing-errors-detail')]; print(urllib.parse.unquote(h[0].split(': ',1)[1].strip())) if h else None"

# 7z / rar funzionano allo stesso modo, basta cambiare l'estensione
curl -fsS -F "file=@lotto.7z"  http://localhost:8080/anonymize -o lotto_elaborato.zip
curl -fsS -F "file=@lotto.rar" http://localhost:8080/anonymize -o lotto_elaborato.zip
```