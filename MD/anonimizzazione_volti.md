# SPECIFICA TECNICA: SISTEMA INDUSTRIAL-GRADE DI ANONIMIZZAZIONE VOLTI BATCH (RUST + ONNX)

## 1. VISIONE GENERALE E CONTESTO

Microservizio batch in Rust, containerizzato Docker, su server centrale. Anonimizza (sfoca) volti di pedoni e guidatori da immagini catturate da migliaia di telecamere stradali fisse (traffico e ZTL).

Obiettivo primario: conformità GDPR, zero falsi positivi visibili nell'immagine finale, 100% Recall su volti veri, precisione chirurgica tramite filtri secondari (classificatore binario).

Vedi Sezione 9 per requisiti hardware minimi legati al carico previsto.

## 2. MODALITÀ DI INGESTIONE E OUTPUT (I/O)

**Ingestione**: endpoint HTTP POST via `axum`, upload `.zip` multipart/form-data. Override del limite payload di default con `DefaultBodyLimit::max()` (minimo 3.5GB, vedi Sezione 9).

**Elaborazione in memoria**: il file ZIP va aperto ed elaborato interamente in RAM (`Cursor<Vec<u8>>`), byte grezzi. Vietato scompattare su disco. Elaborazione delle immagini in parallelo (`tokio::task::spawn_blocking`), concorrenza limitata da `Semaphore` per prevenire OOM (formula in Sezione 9).

**Elaborazione a singolo job**: dato il footprint di memoria (Sezione 9) e la frequenza attesa di 1 upload/ora, il server deve processare **un solo ZIP alla volta**. Implementare un lock globale (`tokio::sync::Mutex<()>` o flag atomico) che rifiuta nuovi upload con `HTTP 429 Too Many Requests` se un job è già in corso.

**Gestione errori**: file non supportati, corrotti o non immagine (es. `.DS_Store`, `.txt`) non interrompono il processo. Loggare, saltare, continuare. Risposta finale: ZIP elaborato + header `X-Processing-Errors: <count>`.

**Identificazione telecamere**: ID dalla cartella nello ZIP (`CAM_001/foto.jpg`) o dal prefisso file (`CAM_001_foto.jpg`). Parser robusto con fallback esplicito se nessuno dei due pattern matcha (loggare come errore, non panicare).

**Output**: ZIP con nome `<input>_elaborato.zip`, compressione `zip::CompressionMethod::Stored` (zero compressione).

## 3. GESTIONE MODELLI ONNX (DOWNLOAD RUNTIME)

I modelli (YOLOv8-Face, Classificatore Binario) NON sono forniti staticamente: vanno scaricati a runtime da URL configurabili.

**Configurazione (env var)**:
- `YOLO_MODEL_URL`, `CLASSIFIER_MODEL_URL`
- `MODEL_CACHE_DIR` (default `/app/models/cache/`, volume persistente)
- `MODEL_YOLO_SHA256`, `MODEL_CLASSIFIER_SHA256` (opzionali, per verifica integrità)

**Logica di avvio**:
1. Se esiste file in cache e il checksum combacia → skip download.
2. Altrimenti scarica via `reqwest`, retry con backoff esponenziale (max 3 tentativi).
3. Se il download fallisce e non esiste cache valida → **fail-fast**: log critico, exit code ≠ 0. Il servizio non deve avviarsi con modelli mancanti.

**Hot-reload**: le `ort::Session` devono essere condivise tramite `arc_swap::ArcSwap<Session>` (preferibile a `RwLock` per reload lock-free), per permettere la sostituzione del classificatore dopo il retraining notturno (Sezione 6) senza riavviare il servizio.

## 4. ARCHITETTURA DEL SOFTWARE E STATI (FSM)

Persistenza (ID telecamera, stato, ROI) via SQLite (`sqlx`). Ogni telecamera segue una FSM:

### INITIAL (fallback)
Telecamera non presente nel DB → creata con questo stato. Blur cautelativo standard sull'intera immagine (o zone fisse). Transizione immediata a LEARNING.

### LEARNING (primi 30 giorni, soglia configurabile)

**Pipeline (YOLOv8-Face)**: soglia confidenza ~0.20. Input `[1, 3, 640, 640]` RGB f32. Output `[1, X, 8400]` (X = numero classi+box params, da confermare in base al modello scaricato).

**Blur cautelativo rettangolare**: bounding box YOLO + margine fisso del 15% su ogni lato. `imageproc::filter::gaussian_blur_f32`, sigma proporzionale alla dimensione del box: `sigma = box_width / 8`, clampato in `[5, 50]`.

**Raccolta dati**: coordinate (X, Y) del centro di ogni detection salvate nel DB in background.

**Estrazione falsi positivi**: crop dei rilevamenti dubbi salvati su disco (`/app/dataset_falsi_positivi/{camera_id}/`) per il retraining (Sezione 6).

**Fine fase**: al 30° giorno (configurabile), task batch calcola la ROI (Sezione 5) e transiziona a ACTIVE.

### ACTIVE (produzione, blur immediato)

**Pipeline**: applica ROI geometrica. Se YOLO trova volto nella ROI → Classificatore Binario effettua secondo controllo sul crop. Input `[1, 3, 224, 224]` RGB normalizzato ImageNet. Output `[1, 2]` logits `[Falso_Positivo, Volto_Reale]`.

**Blur poligonale perfetto**: convex hull dei keypoints facciali (occhi, naso, bocca) se il modello YOLOv8-Face scelto li esporta — **PUNTO APERTO: verificare in fase di scelta modello**. Se non disponibili, fallback a ellisse inscritta nel bounding box. Blur applicato solo dentro il poligono/ellisse tramite maschera (`imageproc::drawing` per generare la mask, poi blur solo sui pixel mascherati). Sigma con la stessa formula di LEARNING.

Falsi positivi confermati dal classificatore → scartati istantaneamente, nessun blur.

### Intervento Operatore (regressione stati)
Endpoint dedicati per forzare reset a INITIAL (azzera ROI) o a LEARNING (mantiene blur di produzione, riapre raccolta dati). **PUNTO APERTO: autenticazione non specificata** — consigliato API key via header `X-Operator-Key`, validata contro valore in env var o tabella DB.

## 5. ALGORITMO DI ESTRAZIONE ROI (v2 — Robusto)

1. **Input**: tutte le coordinate (X, Y) delle detection della camera accumulate nei 30 giorni di LEARNING.
2. **Clustering**: DBSCAN (crate `linfa-clustering` o implementazione custom). Parametri iniziali: `eps=50px`, `min_samples=15` (configurabili per camera, da tarare su dati reali).
3. **Outlier rejection**: punti non assegnati a nessun cluster vengono scartati.
4. **Convex hull**: per l'unione dei cluster validi, calcolare l'inviluppo convesso (crate `geo` + `geo-types`, Andrew's monotone chain).
5. **Smoothing poligonale**: Ramer-Douglas-Peucker (`geo::algorithm::simplify` o implementazione custom), epsilon iniziale `5px`, per ridurre il rumore dei vertici.
6. **Validazione geometrica**: l'area del poligono finale deve essere tra il 10% e il 90% dell'area totale dell'immagine. Se fuori range → fallback a blur cautelativo sull'intera immagine, la camera resta in LEARNING più a lungo, loggare anomalia per revisione manuale.
7. **Margine di sicurezza**: espandere il poligono finale del 5% (buffer geometrico) per evitare tagli netti sui bordi dei volti a ridosso del confine ROI.

## 6. CLASSIFICATORE BINARIO E RETRAINING AUTOMATICO

**Strategia**: ibrida — modello base pre-addestrato + fine-tuning incrementale automatico.

**Dataset iniziale**:
- Classe `Volto_Reale`: dataset seed curato una tantum, non generato dal sistema (`/app/dataset_seed/real_faces/`). **PUNTO APERTO: dataset da fornire esternamente.**
- Classe `Falso_Positivo`: crop estratti automaticamente durante LEARNING (Sezione 4).

**Retraining automatico (fast path, PyO3)**:
- Ambiente Docker: Python3 + PyTorch + torchvision (stima aumento immagine: +2-4GB).
- Trigger: schedulazione notturna (`tokio::time::interval` o crate `tokio-cron-scheduler`), default `03:00`, configurabile via `CRON_RETRAIN_SCHEDULE`.
- Pipeline: Rust invoca via `Python::with_gil` uno script Python che fine-tuna un modello leggero (es. MobileNetV2 pre-addestrato, backbone freezato, retrain solo ultimo layer) su seed reali + falsi positivi accumulati, poi esporta in ONNX (`torch.onnx.export`) in una directory di staging.
- **Validazione pre-swap**: caricare il nuovo ONNX con `ort` su un mini validation set (holdout ~10%). Se accuracy < soglia (es. 0.85) o il file non è caricabile → scartare il nuovo modello, mantenere quello corrente, log warning.
- Se validazione OK: swap atomico in produzione (`ArcSwap::store`), backup del modello precedente in `/app/models/backup/{timestamp}.onnx` per rollback manuale.

## 7. STACK TECNOLOGICO E VERSIONING

```toml
tokio = { version = "1.35", features = ["full"] }
ort = "2.0"
axum = "0.7"
sqlx = { version = "0.7", features = ["sqlite", "runtime-tokio"] }
image = "0.24"
imageproc = "0.23"
zip = "2.0"
reqwest = { version = "0.12", features = ["stream"] }
ndarray = "0.15"
geo = "0.28"
pyo3 = { version = "0.21", features = ["auto-initialize"] }
arc-swap = "1.7"
tracing = "0.1"
tracing-subscriber = "0.3"
```

Non usare versioni diverse da quelle indicate o placeholder generici.

## 8. REQUISITI DI STRUTTURA DEL CODICE RUST

Codice pronto all'uso (senza placeholder logici), modulare, robusto, sfruttando il type system per prevenire panic:

- `main.rs`: init modelli ONNX (con download runtime), pool SQLite, scheduler retraining, avvio server Axum.
- `models.rs`: strutture dati, inferenza `ort`, manipolazione tensori `ndarray`, wrapping `ArcSwap<Session>`.
- `model_loader.rs`: download modelli, verifica checksum, cache, retry logic.
- `zip_worker.rs`: stream reader ZIP in memoria, isolamento file validi/corrotti, controllo concorrenza (Semaphore), scrittura output STORE.
- `pipeline.rs`: coordinamento condizionale basato su stato DB della telecamera, applicazione blur (rettangolare/poligonale).
- `roi.rs`: DBSCAN, convex hull, RDP smoothing, validazione geometrica.
- `training.rs`: bridge PyO3, invocazione script retraining, validazione pre-swap, backup modelli.
- `db.rs`: transazioni sqlx, persistenza FSM, coordinate detection.

## 9. REQUISITI HARDWARE E PERFORMANCE TARGET

**Carico atteso**: ZIP fino a 3GB, ~10.000 immagini, risoluzione 1920x1080, SLA di elaborazione entro 1 ora, frequenza 1 upload/ora.

**Budget di memoria** (picco stimato):
- Buffer ZIP input: fino a 3GB
- Buffer ZIP output (STORE, dimensione paragonabile all'input): fino a ~3GB
- Buffer di lavoro per task concorrente: immagine raw (~6MB @ 1080p) + tensore YOLO input (640×640×3×4 byte ≈ 4.9MB) + tensore classificatore (224×224×3×4 byte ≈ 0.6MB) ≈ ~15-20MB/task
- **Raccomandazione**: minimo 16GB RAM; 32GB se il retraining notturno (Sezione 6) può sovrapporsi temporalmente all'elaborazione ZIP (l'interprete Python/PyTorch embedded via PyO3 aggiunge overhead non trascurabile).

**Concorrenza**: `Semaphore` iniziale = numero core CPU fisici − 2 (margine per runtime tokio e Axum), default 8 se non altrimenti configurato, esposto via env var `MAX_CONCURRENT_IMAGES`. Da validare con load test sull'hardware target: il target di ~2.8 img/s (10.000 immagini/ora) è ampiamente raggiungibile anche su CPU con questa concorrenza, ma va confermato empiricamente.

## 10. PUNTI APERTI DA VALIDARE

1. **Dataset seed "Volto_Reale"**: non generato automaticamente dal sistema, va fornito esternamente prima del go-live del retraining.
2. **Keypoints facciali YOLOv8-Face**: da confermare se il modello scelto li esporta (determina geometria blur ACTIVE: convex hull vs ellisse fallback).
3. **Autenticazione endpoint operatore**: non specificata nella spec originale — proposta API key via header.
4. **Formati immagine supportati**: assunto solo JPEG/PNG salvo diversa indicazione (WebP/HEIC non gestiti).
5. **Comportamento upload concorrenti**: proposto lock globale + HTTP 429, da confermare che sia accettabile lato client/operatore.