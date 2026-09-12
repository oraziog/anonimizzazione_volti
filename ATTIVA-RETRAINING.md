# Retraining notturno — ATTIVO ✅

Aggiornato al 12/09/2026: il retraining notturno è **compilato, attivo e già
eseguito con successo**. Il classificatore "chirurgico" è in produzione.

## Stato attuale

| Componente | Stato |
|---|---|
| Feature `retraining` (PyO3) | **Compilata** nel binario `target\release\anonimizzazione_volti.exe` |
| Python per PyO3 | **3.12.14** gestito da uv (`C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none`) |
| Pacchetti Python (torch 2.5.1 CPU, torchvision, pillow, onnx, onnxscript) | venv `\.venv312\` — caricato in-process via `PYTHONPATH` nel `.env` |
| Seed | 646 volti reali in `dataset_seed\real_faces\` |
| Falsi positivi | raccolta automatica continua in `dataset_falsi_positivi\` |
| Schedulazione | ogni notte **03:00** (`CRON_RETRAIN_SCHEDULE` nel `.env`) |
| **Primo ciclo eseguito** | 12/09/2026 16:55: **status=swapped**, val_acc Python 0.9756, validazione Rust **0.9583**, 178 FP consumati e azzerati |
| Persistenza | `data\classifier_state.json` — il classificatore **sopravvive ai riavvii** (ripristino automatico all'avvio, patch nel `main.rs`) |
| Audit | `GET /operator/retrain-audit` e `GET /operator/classifier` (entrambe nel report giornaliero `report-camere.ps1`) |

## Requisiti Windows per questa installazione (già sistemati)

1. **DLL Python accanto all'exe** (la distribuzione uv non le registra nel PATH):
   `python312.dll` **e** `python3.dll` in `target\release\`.
   Se ricompili da zero, ricopiale:
   ```
   copy "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python312.dll" target\release\
   copy "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python3.dll" target\release\
   ```
   (senza `python3.dll` l'export ONNX fallisce: `onnx_cpp2py_export` la richiede).
2. **Nel `.env`** (caricato da `start-server.ps1` a ogni avvio):
   - `PYTHONHOME=C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none`
   - `PYTHONPATH=C:\Users\Admin\Music\anonimizzazione_volti\.venv312\Lib\site-packages`
3. Per ricompilare: `PYO3_PYTHON` deve puntare al Python 3.12:
   ```
   cd C:\Users\Admin\Music\anonimizzazione_volti\src
   set PYO3_PYTHON=C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe
   cargo build --release --features retraining
   ```

## Come funziona (invariato)

Ogni notte alle 03:00 il servizio:
1. addestra MobileNetV2 (backbone congelato + testa 2 classi) su
   `dataset_falsi_positivi\**` (classe 0) + `dataset_seed\real_faces\` (classe 1);
2. esporta il modello in ONNX (input `[1,3,224,224]`) in `models\cache\`;
3. valida in Rust: accuracy >= `RETRAIN_MIN_ACCURACY` (0.85) e non peggiore del
   modello attivo (`RETRAIN_REGRESSION_EPS=0.0`) — un candidato scarso non
   peggiora mai il servizio;
4. se passa, **swap a caldo** + audit + pulizia dei FP consumati.

Con classificatore caricato e `CLASSIFIER_ENFORCE=true`: dentro la ROI delle
cam ACTIVE vengono sfocati solo i crop confermati come volti (p_volt >=
`CLASSIFIER_CONFIRM_THRESHOLD=0.5`) — maschera chirurgica, meno area inutile.

## Diagnostica rapida

```
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Music\anonimizzazione_volti\report-camere.ps1 -Stampa
curl -H "X-Operator-Key: TUACHIAVE" http://localhost:8080/operator/classifier
```

Log da cercare: `nightly retraining audit: status=swapped` (ok),
`restored from persisted state` (il modello è tornato dopo un riavvio).
Per testare gli import Python in-process:
`cargo run --release --features retraining --example pydiag` (da `src\`).

## Stack Docker (alternativo, non ancora attivo)

`docker-compose.yml` è pronto nella root (porta host **8081** -> container 8080,
monta `dataset_seed\` e `dataset_falsi_positivi\` locali, neutralizza
PYTHONHOME/PYTHONPATH Windows). Stato dell'installazione Docker:

- Client Docker **29.7.2** già installato (`C:\Program Files\Docker\Docker\`).
- Pacchetto **WSL 2.7.14** installato (12/09/2026).
- Manca ancora il componente Windows `VirtualMachinePlatform` (l'attivazione
  è fallita con errore DISM e richiede comunque un riavvio).

Verdetto finale (12/09/2026):

Su questa immagine Windows le feature `VirtualMachinePlatform`,
`Microsoft-Windows-Subsystem-Linux` e `Hyper-V` **non esistono** nell'elenco
DISM (verificato con `Get-WindowsOptionalFeature -Online`). Ho anche eseguito
la riparazione completa dell'immagine (`sfc /scannow` +
`DISM /RestoreHealth`, completata con successo) e le feature **restano
assenti**: non e' corruzione riparabile, i pacchetti di virtualizzazione sono
stati rimossi dall'immagine (installazione Windows modificata/"strippata":
kernel build 26100 con ProductName "Windows 10 Pro").

Docker/WSL2 **non e' installabile su questa installazione** senza un
repair-install con ISO originale di Windows (o reinstallazione). Alternativa:
**restare sulla build nativa** (consigliata per questo PC) — il retraining
notturno funziona gia' in locale (vedi sopra) e il `docker-compose.yml` resta
pronto per un futuro desktop/server con Windows originale.
