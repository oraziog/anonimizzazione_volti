# Manuale di installazione e messa in servizio (Windows)

Questo documento distilla **l'intero percorso reale** di installazione, avvio
e messa a punto del servizio su Windows: ogni passaggio è stato eseguito e
verificato su una macchina Windows 10/11 Pro. Per la via Linux/Docker vedi
`src/README.md` (quick start) e `src/Dockerfile`.

> Regola pratica appresa sul campo: **l'ordine dei passi conta**. Prima
> l'ambiente, poi il binario, poi la configurazione, poi i dati di addestramento.
> Saltare un passaggio si paga sempre più avanti.

## Indice

1. [Prerequisiti](#1-prerequisiti)
2. [Ottenere il codice](#2-ottenere-il-codice)
3. [Compilare il servizio](#3-compilare-il-servizio)
4. [Configurazione (.env)](#4-configurazione-env)
5. [Primo avvio e verifica](#5-primo-avvio-e-verifica)
6. [Retraining notturno (classificatore chirurgico)](#6-retraining-notturno)
7. [Messa in servizio: FSM delle telecamere](#7-messa-in-servizio-fsm-delle-telecamere)
8. [Strumenti operativi](#8-strumenti-operativi)
9. [Problemi incontrati e soluzioni (lezioni apprese)](#9-problemi-incontrati-e-soluzioni)

---

## 1. Prerequisiti

| Componente | Versione verificata | Note |
|---|---|---|
| Rust (rustup + cargo) | 1.88+ | richiesto da `ort 2.0.0-rc.13` |
| Python | **3.11 o 3.12** | per PyO3 (`retraining`) e per gli script di supporto. **Il 3.13/3.14 NON è supportato da PyO3 0.21** |
| Git | qualsiasi | per clonare la repo |
| 4–6 GB liberi su disco | — | build + modelli ONNX + dati runtime |
| CPU con AVX2 | — | ONNX Runtime |

Consiglio pratico (verificato): con **uv** (`pip install uv` o winget) si
installa Python 3.12 gestito e per-utente in secondi, senza privilegi di
amministratore:

```
uv python install 3.12
uv venv --python 3.12 .venv312
uv pip install --python .venv312\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv312\Scripts\python.exe pillow onnx onnxscript onnxruntime
```

> Lezione appresa: **torch 2.5.1** è la versione allineata al Dockerfile; le
> versioni più nuove (2.14+) esportano ONNX con pesi in un file `.onnx.data`
> separato, formato che la logica di backup/prune del runtime non gestisce.

## 2. Ottenere il codice

```
git clone https://github.com/oraziog/anonimizzazione_volti.git C:\Users\Admin\Music\anonimizzazione_volti
cd C:\Users\Admin\Music\anonimizzazione_volti
```

La radice contiene gli strumenti operativi Windows (avvio/stop, log, report);
il crate Rust è in `src/`.

## 3. Compilare il servizio

### Build base (senza retraining)

```
cd src
cargo build --release
```

Produce `src\target\release\anonimizzazione_volti.exe`.

### Build con retraining notturno (consigliata in produzione)

```
cd src
set PYO3_PYTHON=C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe
cargo build --release --features retraining
```

> Lezione appresa (importante): con la feature `retraining` l'exe embedde
> Python e ha bisogno di **`python312.dll` E `python3.dll`** nella sua stessa
> cartella. Le distribuzioni gestite (uv) non le mettono nel PATH del sistema:
>
> ```
> copy "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python312.dll" target\release\
> copy "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python3.dll" target\release\
> ```
>
> Sintomo senza fix: l'exe esce immediatamente con codice `0xC0000135`
> (STATUS_DLL_NOT_FOUND) **senza stampare nulla**. E senza `python3.dll` il
> training funziona ma l'export fallisce con `DLL load failed while importing
> onnx_cpp2py_export`.

## 4. Configurazione (.env)

```
cd src
copy .env.example .env
```

Variabili da impostare almeno una volta:

| Variabile | Ruolo |
|---|---|
| `OPERATOR_API_KEY` | chiave delle API operatore. **Generarla robusta** (es. 48 caratteri da `secrets.token_urlsafe`); il server la confronta in tempo costante |
| `CAMERA_ID_SOURCE` | `exif` = identità camera dal numero di serie EXIF (fallback automatico sul nome file); `filename` = solo prefisso cartella/nome |
| `MODEL_CACHE_DIR` | cartella locale dove vengono scaricati i modelli ONNX |
| `DATASET_SEED_REAL_FACES_DIR` | seed di volti reali (vedi §6) |
| `DATASET_FP_DIR` | falsi positivi raccolti automaticamente |
| `CRON_RETRAIN_SCHEDULE` | orario del retraining notturno (default `03:00`) |
| `RUST_LOG` | rumore dei log: `warn,anonimizzazione_volti=info` = solo INFO del servizio; `RUST_LOG=warn,perf=info` per il minimo |
| `NO_COLOR=1` | log in testo puro, senza codici colore ANSI |
| `PYTHONHOME` / `PYTHONPATH` | (solo build con `retraining`) home della distribuzione Python 3.12 e cartella `site-packages` del venv |

> Lezione appresa: `start-server.ps1` carica il `.env` nella sessione prima di
> lanciare l'exe: ogni nuova variabile è quindi subito efficace. Il file
> `.env` contiene segreti ed è escluso da git — mai committarlo.

## 5. Primo avvio e verifica

**Doppio click** su `start-server.cmd` (o da console: `powershell -NoProfile
-ExecutionPolicy Bypass -File start-server.ps1`). Lo script:

1. carica il `.env` (con fallback automatico se si è in sottocartelle);
2. lancia l'exe **nascosto e indipendente** dalla finestra;
3. scrive i log in `logs\server.log` e `logs\server.err`
   (la run precedente resta in `server.previous.log`);
4. salva il PID in `logs\server.pid`;
5. attende che la porta 8080 risponda e stampa `Server ONLINE`.

Verifiche immediate:

```
curl http://localhost:8080/health
curl -X POST http://localhost:8080/anonymize -F "file=@C:\percorso\test.zip" -o C:\percorso\out.zip
```

L'header `X-Processing-Errors` della risposta indica gli errori per-immagine.
Stop: doppio click su `stop-server.cmd`.

> Lezione appresa: su PowerShell 5.1 `Start-Process -RedirectStandardOutput`
> **resta appeso** se lanciato da console interattiva; per questo gli script
> generano un launcher `.cmd` intermedio per la redirezione dei log. Anche il
> probe di attesa usa un test TCP diretto, non `Invoke-WebRequest`.

## 6. Retraining notturno

Il classificatore "chirurgico" (che dentro la ROI sfoca solo i veri volti)
richiede tre ingredienti:

1. **Seed di volti reali** — si costruisce con `src/python/prepare_seed.py`
   da un dataset YOLO o da WIDER FACE (100 immagini val → ~646 crop). Percorsi
   attesi: `dataset_seed/real_faces/` (volto) e opzionale
   `dataset_seed/real_faces_label/faces/` per la revisione manuale.
2. **Falsi positivi** — raccolti **automaticamente** dalle cam in LEARNING:
   ogni detection YOLO con confidenza in `[YOLO_CONF_THRESHOLD, FP_CROP_CONF_MAX)`
   viene salvata come crop in `dataset_falsi_positivi/{camera}/`.
3. **La build con `retraining`** (§3) e, nel `.env`, `PYTHONHOME`/`PYTHONPATH`.

Alle `03:00` il runtime: addestra MobileNetV2 (testa nuova, backbone congelato)
→ esporta ONNX → **doppia validazione** (accuracy Python ≥
`RETRAIN_MIN_ACCURACY=0.85` e test Rust-side con ort, non peggiore del modello
attivo) → swap a caldo con backup → scrive `data/retrain_audit.json`
(consultabile su `GET /operator/retrain-audit`). Il classificatore swappato
**sopravvive ai riavvii**: all'avvio viene ripristinato da
`data/classifier_state.json` (log: `classifier restored from persisted state`).

Stato in ogni momento: `GET /operator/classifier` (`loaded`,
`active_check_enabled`, accuracy, contatori inference).

> Lezione appresa: al primo swap tutti i falsi positivi consumati vengono
> **cancellati** (il modello ha già imparato su di loro): il dataset ricresce
> da solo nei giorni successivi. Attenzione inoltre al **contatore di
> sequenza**: prima del fix ora incluso, più immagini dello stesso job
> sovrascrivevano i crop delle altre (log "saved 1670" vs 178 reali).

## 7. Messa in servizio: FSM delle telecamere

Ciclo di vita di ogni camera (identità dal seriale EXIF o dal nome file):

```
INITIAL → LEARNING (blur box + 15%, raccolta detection e FP)
        → (dopo LEARNING_DAYS=30, job notturno) → estrazione ROI (DBSCAN → hull)
        → ACTIVE (blur solo dentro la ROI, confermato dal classificatore)
```

- La notte stessa l'FSM fa **catch-up**: una camera con finestra scaduta viene
  promossa al primo avvio utile (log: `ROI extracted (…% of frame) → ACTIVE`).
- La promozione richiede dati sufficienti: cluster denso (DBSCAN
  `ROI_EPS_PX=50`, min 15 punti) e area tra `ROI_AREA_MIN` (10%) e max.
  Con pochi punti resta in LEARNING (`not enough detection data`).
- Le cam ACTIVE rivalutano la ROI ogni notte (finestra
  `ROI_REEXTRACT_WINDOW_DAYS`, adozione solo se stabile IoU ≥ 0.6).
- Un reset operatore (`POST /operator/cameras/{id}/reset`) riporta la camera
  a LEARNING; `POST /operator/cameras/{id}/roi` con `{"type":"full"}` forza
  ACTIVE a full-frame (utile per test).

Verifica soglie a caldo: pannello `http://localhost:8080/operator/settings`
(con la `OPERATOR_API_KEY`) — ogni modifica è immediata e persistita in
`data/runtime_config.json`.

## 8. Strumenti operativi

(robusti su Windows, tutti testati; eseguire con
`powershell -NoProfile -ExecutionPolicy Bypass -File <script>`)

| Script | Funzione |
|---|---|
| `start-server.cmd` / `stop-server.cmd` | avvio/stop con doppio click, log e PID |
| `anonimizza-cartella.ps1` | cartella di foto → ZIP → `/anonymize` → archivio anonimizzato (anche drag & drop) |
| `load-env.ps1` | carica il `.env` nella sessione (fallback automatico sulla cartella padre) |
| `rotate-logs.ps1` | log vecchi → ZIP settimanale, tiene gli ultimi 4 (task `AnonVolt_RotateLogs`) |
| `riassunto-log.ps1` | ERROR + transizioni FSM per giorno (`-Giorni`, `-Dettaglio`) |
| `report-camere.ps1` | report giornaliero: cam, job, audit retraining, salute classificatore, **storico accuracy con grafico testuale** (task `AnonVolt_ReportCamere` 08:00) |
| `report-settimanale.ps1` | trend settimanale accuracy in Markdown (`reports\trend-accuracy-AAAA-Www.md`, task `AnonVolt_TrendAccuracy`) |
| `alert-retraining.ps1` | se l'ultimo retraining è `failed`/`rejected` scrive `alerts\RETRAINING-ATTENTION.txt` con causa e checklist (task `AnonVolt_RetrainAlert` 04:00); si auto-rimuove al successo |
| `valida-classificatore.py` | validazione con verità certa: recall volti chiari/piccoli, FP su sfondi, tabella per soglia |
| `test_zip/make_exif_test.py` | genera immagini con seriale EXIF per testare `CAMERA_ID_SOURCE=exif` |

## 9. Problemi incontrati e soluzioni

Diario delle difficoltà reali incontrate durante installazione e messa a
punto — se riscontri un sintomo simile, la cura è qui.

| # | Sintomo | Causa | Soluzione |
|---|---|---|---|
| 1 | `load-env.ps1`: "File .env non trovato" | il cercava il `.env` nella CWD | patch: fallback sul `.env` accanto alla cartella padre dello script (già incluso) |
| 2 | Avvio con doppio click bloccato / log non scritti | `Start-Process -RedirectStandardOutput` si appende da console | launcher `.cmd` intermedio generato da `start-server.ps1` |
| 3 | Caratteri `[2m [0m` nei file di log | codici ANSI di `tracing` | `NO_COLOR=1` nel `.env` (rispettato da tracing-subscriber ≥ 0.3.18) |
| 4 | L'exe esce subito, codice `-1073741515` (0xC0000135), nessun output | `python312.dll` non trovata | copiare `python312.dll` + `python3.dll` accanto all'exe (vedi §3) |
| 5 | Audit retraining: `Module onnx is not installed!` | pacchetti mancanti nel venv usato in-process | `uv pip install pillow onnx onnxscript` nel venv e `PYTHONPATH` corretto nel `.env` |
| 6 | Export ONNX genera file `.onnx.data` separato | torch troppo nuovo | allineare torch a 2.5.1 (indice CPU) |
| 7 | `cargo test --features retraining` non compila (`TempDir` senza `Clone`) | test mai compilati con la feature | fix incluso: `dir.to_path_buf()` in `training.rs` |
| 8 | FP crop persi: log ne dichiara 1670, su disco 178 | indice per-immagine nel nome file → sovrascritture concorrenti | fix incluso: contatore atomico globale `FP_CROP_SEQ` |
| 9 | Camera resta in LEARNING: "not enough detection data" | DBSCAN senza cluster denso | serve una banda di detection densa e realistica; verificare `ROI_EPS_PX`/`ROI_MIN_SAMPLES` |
| 10 | Camera resta in LEARNING: "ROI validation failed (area 0.1%)" | ROI più piccola del minimo (10%) | è protezione, non errore: aumentare i dati o rivedere la scena |
| 11 | Classe FP sbilanciata → classificatore "sempre negativo" | FP raccolti in settimane superano il seed | già gestito: cap `MAX_CLASS_RATIO` 4:1 + clear FP dopo lo swap |
| 12 | WSL2/Docker non installabili (`Feature name is unknown`) | immagine Windows senza pacchetti di virtualizzazione (installazione modificata) | `DISM /RestoreHealth` NON li ripristina se rimossi: serve repair-install con ISO originale. Alternativa: build nativa (questo manuale) |

## Verifica finale dell'installazione

Checklist rapida dopo il primo avvio:

- [ ] `curl http://localhost:8080/health` → 200
- [ ] Upload di prova → HTTP 200, `x-processing-errors: 0`
- [ ] `GET /operator/cameras` con `OPERATOR_API_KEY` → elenco cam in LEARNING
- [ ] (con `retraining`) log contiene `classifier restored from persisted state` dopo un riavvio
- [ ] `report-camere.ps1 -Stampa` → report completo senza errori
- [ ] Primo retraining notturno: `GET /operator/retrain-audit` → `status: swapped`
