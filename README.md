# Anonimizzazione Volti

Servizio batch di anonimizzazione facciale (Rust + ONNX) per telecamere fisse
ZTL/traffico, orientato alla conformità GDPR: **zero volti reali visibilmente
non offuscati in uscita**, precisione chirurgica tramite classificatore binario
e apprendimento automatico del ROI per telecamera.

Implementazione della specifica tecnica (italiano) in `MD/anonimizzazione_volti.md`;
le sezioni del documento sono referenziate nei sorgenti. Il piè di pagina
di `src/README.md` è il riferimento operativo completo.

## Cosa fa

- **Ingest batch** — `POST /anonymize` (zip/7z/rar, streaming su disco, mai in
  RAM), `/anonymize/batch` per volumi oltre `BODY_LIMIT_BYTES`.
- **Pipeline a stati per telecamera (FSM)** — `INITIAL` → `LEARNING` → `ACTIVE`:
  sfocatura full-frame cautelativa, raccolta dati, estrazione **ROI** (DBSCAN →
  hull convesso), gate ROI + **classificatore binario** + maschera hull/ellisse
  dei 5 landmark facciali, fallback "head" per profili puri.
- **Autoapprendimento notturno** — retraining PyO3 (MobileNetV2: seed + falsi
  positivi), export ONNX, validazione Rust-side, gate A/B, swap atomico con
  backup, audit JSON consultabile dall'operatore.
- **Store S3** — backend di ingest asincrono da bucket S3-compatibili
  (MinIO / AWS / Spaces): job intake, output su bucket, audit log JSON,
  **webhook di completamento con URL presigned (1 h)**, sweep operator batch,
  soglia di concorrenza `S3_MAX_CONCURRENT_JOBS`. Feature cargo `s3`
  (vedi `src/docker-compose.minio.yml`).
- **Code asincrone** — consumer **SQS** e **RabbitMQ** (feature `queue` /
  `rabbitmq`) sullo stesso worker S3: ack su successo, retry esponenziale e DLQ.
- **Header X-Processing-Errors** + `<input>_error.txt` per report per-file.
- **GPU** — provider ONNX Runtime `cpu | cuda | tensorrt | directml`
  (feature cargo + `Dockerfile.gpu`), fail-fast se la GPU configurata non è
  utilizzabile.

## Struttura

| Percorso | Contenuto |
| --- | --- |
| `src/` | Crate Rust (build `cargo build --release` dentro `src/`) |
| `src/README.md` | Documentazione completa: architettura, API, configurazione, S3, code, tuning misurato |
| `src/docker-compose.yml` | Stack completo con retraining notturno |
| `src/docker-compose.minio.yml` | Stack + MinIO + backend S3 (feature `s3`) |
| `src/docker-compose.gpu.yml` | Stack con ONNX Runtime CUDA (feature `cuda`) |
| `src/python/` | `retrain.py` (retraining notturno), `prepare_seed.py` (seed del classificatore) |
| `src/scripts/` | Harness di test/esplorazione (`test-wider.ps1`, `eval_wider_output.py`, `blur_compare.py`, …) |
| `MD/anonimizzazione_volti.md` | Specifica tecnica |
| `models_cache/` | Modelli ONNX scaricati a runtime (ignorati da git, vedi `.gitignore`) |

## Avvio rapido (backend S3 con MinIO)

```bash
cd src
cp .env.example .env    # impostare S3_ENABLED=true + credenziali
docker compose -f docker-compose.minio.yml up -d --build

./scripts/s3_tools.sh upload frame.zip camera_001.zip   # -> bucket input
curl -X POST localhost:8080/anonymize/s3 \
     -H 'Content-Type: application/json' \
     -d '{"input_key":"camera_001.zip"}'
curl localhost:8080/status/<JOB_ID>
```

Flow S3: scarica l'archivio dal bucket input → stessa `ZipProcessor` del path
HTTP → upload `elaborati/<stem>_elaborato.zip` sul bucket output → audit log
JSON sul bucket logs → webhook facoltativo con URL presigned. In caso di
errore l'oggetto in ingresso viene spostato sotto `errori/` nel bucket input
(nessun re-pick da parte dei sweep). Dettaglio completo nel README di `src/`.

## Licenza dei dati

Il sistema **non genera** il seed del classificatore: è esterno/mountato
(`dataset_seed/real_faces`); `src/python/prepare_seed.py` lo costruisce da un
dataset YOLO o da WIDER FACE (verificare la licenza delle immagini prima
dell'uso in produzione). `dataset_falsi_positivi/`, `dataset_seed/`,
`.test-assets/` e i modelli ridimensionati sono esclusi dal repository.