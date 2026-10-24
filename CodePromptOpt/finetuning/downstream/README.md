# Code generation dai prompt locali, con vLLM remoto

Questa cartella invia i prompt prodotti dagli optimizer a un **code generator fisso**, poi salva il codice in un formato compatibile con la sandbox di valutazione esistente. Il client usa solo Python 3.10+ standard: non scarica modelli e non richiede GPU. I modelli risiedono nella DGX Spark, serviti da vLLM. Il PC mantiene prompt, risposte e risultati; non occorre copiare gli adapter sulla Spark.

L'API è [Chat Completions compatibile OpenAI di vLLM](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/). Il server viene avviato tramite Podman; il client può girare sul PC Windows oppure in un container CPU Linux. L'immagine di training H200 non viene riutilizzata sulla Spark: serve una build vLLM adatta a ARM64/GB10, secondo la [procedura NVIDIA DGX Spark](https://build.nvidia.com/spark/vllm/instructions).

## Risultati locali trovati

Percorso rilevato: `CodePromptOpt/finetuning/outputs/runs/`, diverso dal precedente `outputs/h200/runs/`. Il client usa questo percorso come default.

L'audit locale iniziale trova **36 cartelle, 83 file di prompt, 822 righe**; alcuni run non hanno ancora tutti i prompt e un file è parziale. Sono presenti **55 prompt marcati come troncati** e **392 righe che non includono il problema originale verbatim**. Questo è un controllo strutturale: non dimostra da solo un cambiamento semantico, ma nei file si osservano anche risposte ripetitive prive del task. Il client non riscrive, non completa e non scarta quei prompt: la loro qualità fa parte del risultato dell'optimizer. Un confronto funzionale deve conservarne il denominatore.

Prima di usare GPU eseguire un audit offline. Non si effettuano chiamate durante `plan`:

```powershell
& .\CodePromptOpt\finetuning\downstream\client.ps1 plan
```

Il dettaglio è salvato in `outputs/codegen/qwen25-coder-7b-v1/validation/audit.json`. Per default l'esecuzione vera rifiuta copertura incompleta: scegliere run completi con `--runs` o `--run-glob`. `--allow-partial` è disponibile solo come scelta esplicita, da dichiarare nei risultati; non rende automaticamente confrontabili task set differenti.

## 1. Avviare vLLM sulla Spark

Sul dispositivo Linux servono Podman, NVIDIA driver/toolkit e CDI. Verificare `uname -m` (Spark: `aarch64`), `nvidia-smi -L` e `nvidia-ctk cdi list`. Sulla Spark il default dello script è **GPU 0**, mentre il training H200 usava GPU 3. Si può cambiare con `CODEGEN_GPU`.

Trasferire alla Spark il solo `serve_vllm.sh` oppure la cartella downstream. Impostare `VLLM_IMAGE` al tag/digest ARM64 supportato dalla ricetta NVIDIA/vLLM scelta. Non fissiamo un'immagine x86 o una release non verificata sul dispositivo.

```bash
# Sulla Spark: sostituire il placeholder con il tag/digest della ricetta ufficiale.
export VLLM_IMAGE='IMMAGINE_VLLM_COMPATIBILE_SPARK'
export CODEGEN_MODEL='Qwen/Qwen2.5-Coder-7B-Instruct'
# Facoltativo per autenticazione; la stessa chiave va nel client.
# export CODEGEN_API_KEY='...'
# Per un esperimento finale: commit HF del modello.
# export CODEGEN_REVISION='COMMIT_HUGGINGFACE'
bash CodePromptOpt/finetuning/downstream/serve_vllm.sh
podman logs -f cpo-vllm
```

Lo script controlla l'architettura dell'immagine, avvia in detached mode, monta la cache dei soli pesi del code generator e limita il log a 50 MB per container. Usa BF16, contesto 16384 token totali, massimo 4 sequenze e utilizzo GPU 0,70 come valori iniziali da verificare, non come garanzia di memoria sufficiente. Variabili: `CODEGEN_CONTEXT`, `CODEGEN_GPU_UTIL`, `CODEGEN_MAX_SEQS`, `CODEGEN_PORT`, `CODEGEN_CACHE`, `CODEGEN_CONTAINER`. Si possono aggiungere argomenti vLLM dopo il nome dello script. Non viene usato `trust-remote-code`.

L'API è pubblicata **solo su loopback**, porta 8000. Attendere che vLLM sia pronto. Dal dispositivo:

```bash
curl http://127.0.0.1:8000/v1/models
```

Se è impostata una chiave, aggiungere l'header `Authorization: Bearer ...`. Per cambiare modello, terminare il servizio con `podman stop cpo-vllm`, conservare i log, poi `podman rm cpo-vllm` e rilanciare con il nuovo `CODEGEN_MODEL`. La rimozione del container non elimina il volume dei pesi. Servire un modello alla volta limita la memoria occupata.

Configurazioni client incluse: Qwen2.5-Coder-7B-Instruct e 14B-Instruct. Sono punti di partenza per il confronto, non una replica del generatore storico chiamato `qwen3.5-coder` nei dati. Il [modello 7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) documenta l'uso con vLLM. Per altri generatori copiare un JSON, cambiare `model`, `experiment_id` e gli eventuali parametri richiesti dalla ricetta.

## 2. Collegare il PC alla Spark

Aprire un terminale PowerShell separato e mantenere il tunnel attivo:

```powershell
ssh -N -L 8000:127.0.0.1:8000 UTENTE@INDIRIZZO_SPARK
```

Se la porta locale è occupata usare `8001:127.0.0.1:8000` e passare `--base-url http://127.0.0.1:8001/v1` al client. L'indirizzo del client deve terminare con `/v1`. Non occorre esporre il servizio alla rete pubblica. Il tunnel usa l'autenticazione SSH già configurata.

## 3. Pilot dal PC Windows

Eseguire dalla radice del repository. `client.ps1` cerca prima l'interprete portabile esistente in `CodePromptOpt/.runtime/python/python.exe`, altrimenti `python` nel PATH. Non installa librerie. Per autenticazione impostare `$env:CODEGEN_API_KEY = '...'` senza salvarla nel JSON.

Prima delle misure finali compilare `server_metadata` nel config JSON con commit del modello, digest dell'immagine e dettagli di quantizzazione/serving. Lo script registra tali informazioni come dichiarate; non può verificarle automaticamente tramite l'API. Cambiare `experiment_id` se cambia modello, revisione, ambiente o protocollo. Gli esempi con placeholder sono adatti a verificare il collegamento, non a documentare una replica scientifica.

```powershell
# Audit di un run completo
& .\CodePromptOpt\finetuning\downstream\client.ps1 plan --runs qwen3_8b-best-s42

# Originale, statico, optimizer base e optimizer SFT sullo stesso code generator
& .\CodePromptOpt\finetuning\downstream\client.ps1 run --runs qwen3_8b-best-s42
```

Il client interroga `/v1/models` prima di inviare i prompt. Non esegue il codice ricevuto sul PC. I parametri del generatore sono identici per tutti gli input: temperatura 0, top-p 1, seed 42, massimo 2048 token di output. vLLM viene avviato con `--generation-config vllm` per evitare default ereditati dal repository del modello. Il seed non garantisce identità bit per bit fra hardware/versioni/concorrenza diversi: tenere fisso l'ambiente del server.

## 4. Tutti i run disponibili e più code generator

Nel materiale locale, i run 4B e 8B sono un primo gruppo da sottoporre all'audit; il controllo resta automatico prima delle richieste:

```powershell
& .\CodePromptOpt\finetuning\downstream\client.ps1 plan --run-glob 'qwen3_4b-*' 'qwen3_8b-*'
& .\CodePromptOpt\finetuning\downstream\client.ps1 run --run-glob 'qwen3_4b-*' 'qwen3_8b-*'
```

Per tutti i file effettivamente presenti, accettando esplicitamente le mancanze:

```powershell
& .\CodePromptOpt\finetuning\downstream\client.ps1 run --allow-partial
```

Meglio completare la generazione mancante e valutare suite confrontabili: `--allow-partial` non sostituisce i task mancanti. Per usare il test finale aggiungere `--split test` solo quando il protocollo è fissato e i rispettivi file sono disponibili.

Quando il server serve il 14B, usare il config corrispondente:

```powershell
& .\CodePromptOpt\finetuning\downstream\client.ps1 run --runs qwen3_8b-best-s42 --config CodePromptOpt/finetuning/downstream/configs/qwen25_coder_14b.json
```

Non ricaricare gli optimizer: servono solo i file prompt. Non è necessario conservare in cache tutti i pesi del training.

### Client interamente containerizzato su Linux

Se copi i prompt sulla Spark o su un altro server Linux, basta un'immagine Python CPU multiarch, indipendente da CUDA:

```bash
bash CodePromptOpt/finetuning/downstream/client.sh build
bash CodePromptOpt/finetuning/downstream/client.sh plan --runs qwen3_8b-best-s42
bash CodePromptOpt/finetuning/downstream/client.sh run --runs qwen3_8b-best-s42
```

Il client usa networking host e raggiunge `127.0.0.1:8000`; non occupa GPU. È in foreground: per lavori lunghi usare una sessione persistente come `tmux`. La risposta cache permette comunque di riprendere un'esecuzione interrotta. Questa variante non richiede Python installato sull'host.

## Ripresa, cache e output

Il client usa retry limitati su errori di rete, rate limit e HTTP 5xx. Errori HTTP 400 (per esempio contesto insufficiente) vengono registrati, senza troncare automaticamente il prompt e senza diventare falsi fallimenti funzionali. Se il modello produce contenuto vuoto con finish `length`, il record rimane un output del generatore; se manca del tutto il contenuto testuale la richiesta fallisce.

Ogni risposta riuscita viene salvata con transazione SQLite. Per riprendere basta rilanciare **lo stesso comando e config**. Richieste identiche per stesso task, parametri e metadata del server vengono riutilizzate: original/static/base ripetuti fra strategie non producono chiamate inutili. Con i dati locali iniziali si osservano **184 richieste uniche su 822 righe**, prima di scegliere i run. Le risposte condivise non sono repliche statistiche indipendenti. Per temperature non zero serve un protocollo diverso: questo client le rifiuta per mantenere esplicita la semantica di cache greedy.

```text
outputs/codegen/qwen25-coder-7b-v1/validation/
  audit.json                    copertura e problemi strutturali dei prompt
  configuration.json            protocollo del code generator
  responses.sqlite              risposte API integrali, una per richiesta unica
  status.json                   richieste fallite e file incompleti
  qwen3_8b-best-s42/
    validation-original-code.jsonl
    validation-static-code.jsonl
    validation-base-code.jsonl
    validation-sft-code.jsonl
```

I file code vengono scritti atomicamente solo quando tutte le righe del rispettivo file input sono state generate. Se il processo termina prima della materializzazione, la cache delle risposte riuscite permette di ricostruirli al rilancio. `--workers 2` è il default: ridurlo a 1 se il server è sotto pressione. I record conservano il prompt originale, `task_id`, variante, codice estratto, motivo di fine, token usage, latenza della richiesta originale, metadata e ID della risposta condivisa. La latenza di un hit di cache non va trattata come una nuova misura di inferenza. Il testo API integrale è nel database. Non si cancellano o alterano i file prompt.

Non eseguire contemporaneamente due client che scrivono nello stesso output esperimento. Per un nuovo esperimento cambiare `experiment_id`; conservare config e database. I risultati JSONL occupano solo testo, non copie dei pesi.

## 5. Test funzionali in sandbox

Code generation non equivale ancora a pass@1: servono harness verificati per i task. I log locali non contengono le suite complete; non vengono inventate. Preparare nel repository un `verified_tests.jsonl` con schema:

```json
{"task_id":"task_ID_REALE", "harness":"from solution import funzione\nassert funzione(...) == ...\n"}
```

Usare le suite ufficiali e un adattamento verificato di interfacce/import; per APPS verificare la mappatura al problema originale e gestire stdin/stdout. Il placeholder sopra descrive il formato, non è un test eseguibile. `container_evaluate` richiede un harness per ogni task del file.

La valutazione va eseguita su Linux con Podman, in VM/macchina dedicata al codice non attendibile; non lanciare le risposte direttamente nel Python del PC. Copiare solo code JSONL e test, oppure tutta la struttura di output. Dalla radice del repository:

```bash
bash CodePromptOpt/finetuning/downstream/client.sh build
bash CodePromptOpt/finetuning/downstream/evaluate_suite.sh \
  CodePromptOpt/finetuning/outputs/codegen/qwen25-coder-7b-v1/validation verified_tests.jsonl
```

Il wrapper riusa la sandbox esistente: niente rete/GPU, nessun socket Podman montato, utente non root, filesystem read-only e limiti di tempo/RAM/CPU. I risultati vengono scritti accanto ai code JSONL con suffisso `-results.jsonl`; ogni file ha una directory `-jobs` distinta. Per una sola valutazione usare `CPO_IMAGE=localhost/cpo-codegen-client:1 bash CodePromptOpt/finetuning/podman.sh evaluate ...` con i quattro argomenti descritti nella guida H200. Il wrapper batch si ferma davanti a jobs/risultati esistenti per evitare sovrascritture: per proseguire passare la singola cartella run ancora da valutare.

Confrontare base contro SFT, e poi original/static contro SFT, sullo stesso insieme di task e server:

```bash
CPO_IMAGE=localhost/cpo-codegen-client:1 bash CodePromptOpt/finetuning/podman.sh cpu \
  python -m CodePromptOpt.finetuning.compare \
  --baseline CodePromptOpt/finetuning/outputs/codegen/qwen25-coder-7b-v1/validation/qwen3_8b-best-s42/validation-base-results.jsonl \
  --candidate CodePromptOpt/finetuning/outputs/codegen/qwen25-coder-7b-v1/validation/qwen3_8b-best-s42/validation-sft-results.jsonl
```

La metrica è pass@1 empirica e il delta ha bootstrap appaiato per task. Mancanza di problema nel prompt e troncamento non devono essere esclusi retroattivamente dal denominatore. I seed del training non aumentano i dieci problemi indipendenti della validation.

## Verifica del client

```powershell
& .\CodePromptOpt\.runtime\python\python.exe -m unittest CodePromptOpt.finetuning.downstream.test_run -v
```

Il test usa un server HTTP locale fittizio: verifica invio, estrazione del codice, riuso delle risposte, ripresa e protezione dal cambio di revisione. Non contatta modelli reali. Il deployment Spark e i test benchmark richiedono l'hardware e le suite effettive.
