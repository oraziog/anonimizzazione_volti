# Avvio completo con Docker Desktop (incluso riaddestramento notturno)

Guida passo-passo per eseguire il servizio completo con **Docker Desktop** su
Windows. Il container compila il binario **con la feature `retraining`**
(PyO3 + Python 3.11 + PyTorch CPU) ed esegue ogni notte il riaddestramento del
classificatore, cosa che la build locale nativa non fa (pyo3 0.21 non supporta
Python 3.14).

> Nota ambiente: questa guida è verificata contro la configurazione
> (`Dockerfile`, `docker-compose.yml`, `.env.example`), ma chi la scrive non ha
> Docker installato nella propria shell: i comandi vanno eseguiti dal tuo
> terminale.

## 0. Prerequisiti

- **Docker Desktop** installato e avviato, con backend **WSL2** (impostazione
  di default nelle versioni recenti).
- **RAM**: minimo 16 GB (spec §9), consigliati 32 GB perché il PyTorch
  incorporato può sovrapporsi a un job ZIP notturno. `docker-compose.yml` fissa
  `mem_limit: 32g`.
- **Disco**: ~5 GB per le immagini Docker (torch CPU ~2 GB) + i volumi dati.
- La cartella del crate (`src/`) deve contenere `Dockerfile`,
  `docker-compose.yml` e `.env.example` — è già così.

## 1. Configurazione

Nella cartella del crate:

```bash
cd C:/Users/Admin/Music/anonimizzazione_volti/src
cp .env.example .env
```

Modifica almeno `OPERATOR_API_KEY` in `.env` (se vuoto, gli endpoint operatore
rispondono 403). I percorsi in `.env.example` (`/app/...`) sono già quelli
giusti per il container: **non cambiarli** in locale.

Per provare più in fretta il ciclo notturno puoi anche impostare:

```env
LEARNING_DAYS=1            # default 30; minimo accettato: 1
CRON_RETRAIN_SCHEDULE=16:45  # HH:MM vicino all'ora corrente per test immediato
```

## 2. Build e avvio

```bash
docker compose up -d --build
```

La prima build scarica Rust 1.88, Python 3.11 e PyTorch CPU: **15–40 minuti**
la prima volta (le volte successive è quasi istantanea). Al termine:

```bash
docker compose logs -f anonimizzazione-volti
```

Nel log devi vedere, in ordine:

1. `effective image concurrency: N` — il limite di concorrenza §9;
2. `model cache hit / downloading model from ...` — il modello YOLO
   (`yolov8n-face.onnx`, ~12 MB) scaricato e verificato allo startup;
3. `listening on http://0.0.0.0:8080` — server pronto.

Health check (anche dal browser):

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

Il container ha un `HEALTHCHECK` ogni 30 s: `docker compose ps` mostra
`healthy`.

## 3. Prova il ciclo /anonymize

```bash
# crea lo ZIP di test: CAM_001/foto1.jpg (JPEG o PNG)
curl -fsS -F "file=@test-volti.zip" http://localhost:8080/anonymize \
  -o elaborato.zip -D -
```

Atteso: `HTTP/1.1 200 OK`, header `x-processing-errors: 0` (o un conteggio se
ci sono file non immagine), e `elaborato.zip` = `<nome>_elaborato.zip` con le
immagini anonimizzate. Una copia dell'archivio è salvata in `/app/data` (volume
`data`).

La telecamera viene creata automaticamente in **LEARNING**:

```bash
curl -H "X-Operator-Key: <la-tua-chiave>" \
     http://localhost:8080/operator/cameras
```

## 4. Riaddestramento notturno del classificatore

Il riaddestramento **richiede dati** e viene saltato (con warning nel log) finché
non ci sono **almeno 2 immagini reali e 2 falsi positivi**:

- **Classe 1 (volti reali)** — seed curato in `/app/dataset_seed/real_faces`
  (volume `seed`). Allo stato attuale è **vuoto**: devi popolarlo tu, ad
  esempio con lo script `python/prepare_seed.py` (ritaglia i volti da un
  dataset YOLO, es. il "Face Detection Dataset" CC0 di Kaggle).
- **Classe 0 (falsi positivi)** — raccolti automaticamente dal servizio durante
  LEARNING in `/app/dataset_falsi_positivi/{camera_id}/` (volume `fp_crops`):
  dopo qualche job avrai già i tuoi negativi.

### Come iniettare il seed nel volume

Prepara i crop in locale, poi copiali nel volume `seed` (il nome esatto del
volume lo trovi con `docker compose config --volumes`; di solito
`src_seed` perché la cartella del compose è `src`):

```bash
python python/prepare_seed.py --source C:/percorso/dataset \
    --out dataset_seed/real_faces

docker run --rm -v src_seed:/seed -v "C:/Users/Admin/Music/anonimizzazione_volti/src/dataset_seed":/src \
    alpine sh -c 'cp -r /src/. /seed/'
```

In alternativa, per uno sviluppo comodo, monta una bind mount nel compose
(riga sotto `volumes:` del servizio):

```yaml
- ./dataset_seed:/app/dataset_seed
```

### Quando parte

All'avvio il servizio fa subito il passaggio ROI per le camere in LEARNING con
finestra scaduta (catch-up), poi il **retrain** parte all'orario di
`CRON_RETRAIN_SCHEDULE` (default `03:00`). Nel log vedrai:

```
nightly retraining: N real samples, M false-positive samples
classifier swapped → /app/models/cache/classifier_20260907_030000.onnx (accuracy 0.9123)
```

Il candidato viene **validato due volte** prima dello swap: accuracy di holdout
(≥ `RETRAIN_MIN_ACCURACY`, default 0.85) e re-inferenza Rust con `ort` su un
mini validation set; il classificatore precedente viene copiato in
`/app/models/backup/` (volume `models`). Se la validazione fallisce, il
candidato viene scartato e il classificatore attuale resta in uso.

Dopo il primo retrain riuscito, le camere **ACTIVE** usano il classificatore
per filtrare i falsi positivi dentro la ROI (finché non c'è un classificatore,
ACTIVE sfoca comunque tutto ciò che cade nella ROI: comportamento fail-safe).

## 5. Comandi utili

```bash
docker compose ps                          # stato / healthcheck
docker compose logs -f anonimizzazione-volti   # log in tempo reale
docker compose down                        # ferma (i volumi restano)
docker compose down -v                     # ferma E cancella i volumi (dati inclusi!)
docker compose exec anonimizzazione-volti sh -c 'ls /app/data /app/dataset_falsi_positivi'
```

## 6. Troubleshooting

| Problema | Causa probabile / soluzione |
|---|---|
| `docker compose build` lento | Prima build: scarica toolchain + torch. È normale; le successive usano la cache. |
| Porta 8080 occupata | Cambia `ports` in `docker-compose.yml` (es. `8081:8080`) e `BIND_ADDR`. |
| `retraining skipped: need ≥2 samples per class` | Seed e/o FP insufficienti: popola il volume `seed` (passo 4) e invia qualche job per accumulare FP. |
| Nessun swap classificatore dopo il cron | La run notturna parte all'ora configurata: per test immediato metti `CRON_RETRAIN_SCHEDULE` a un minuto dopo l'ora corrente e riavvia (`docker compose restart`). |
| La camera resta in LEARNING | `LEARNING_DAYS` (default 30) non ancora trascorso, o ROI non validabile (§5.6): il log spiega il motivo; `LEARNING_DAYS=1` accelera i test. |
| `mem_limit: 32g` rifiutato | Docker Desktop su WSL2 richiede abbastanza RAM allocata (Settings → Resources → Memory); riduci `mem_limit` se serve. |
| Niente rete per il download del modello | Il modello viene scaricato all'avvio: serve internet la prima volta; dopo è in cache nel volume `models`. |

## 7. Posta in sicurezza

- Cambia `OPERATOR_API_KEY` con una chiave reale prima di esporre il servizio.
- Il servizio accetta ZIP fino a `BODY_LIMIT_BYTES` (default 3.5 GB) e processa
  un job alla volta (gli upload concorrenti ricevono HTTP 429).
- I volumi Docker contengono **dati personali** (immagini originali nel job,
  detection nel DB): valuta cifratura del volume e policy di retention.