# Esperimenti H200 con Podman

Per usare anche il code generator sulla H200 via Podman, dopo la generazione dei prompt, vedere [downstream/H200.md](downstream/H200.md): `h200.sh` gestisce vLLM sulla GPU 3 e client locale, senza tunnel SSH.

Configurazione persistente: `podman.env` viene letto a ogni invocazione del wrapper (anche da una nuova sessione SSH). Contiene GPU, immagine, container, cache e `CPO_OUTPUT`, inizialmente `CodePromptOpt/finetuning/outputs/h200-batteria-v1`. `podman.sh config` mostra i valori attivi. I comandi suite senza `--output` usano ora quel percorso; per continuare i vecchi esperimenti `outputs/h200` passare `--output CodePromptOpt/finetuning/outputs/h200` o modificare `CPO_OUTPUT` nel file. Le variabili gia esportate nel terminale hanno precedenza sul file.

La funzione di attesa non va piu definita nella shell: `bash CodePromptOpt/finetuning/podman.sh await-job` aspetta, salva automaticamente un log univoco nella directory del job e rimuove solo il container terminato con successo. In caso di fallimento conserva il container e restituisce l'exit code. Funziona dopo una riconnessione SSH; `wait` resta disponibile come sola attesa senza pulizia. Non cambiare `CPO_CONTAINER_NAME` mentre un lavoro e in esecuzione. Non serve usare `source podman.sh`.

```bash
bash CodePromptOpt/finetuning/podman.sh train --models qwen3_4b
bash CodePromptOpt/finetuning/podman.sh await-job
```

Questa guida assume un server Linux con una H200 accessibile come **GPU host 3**. Tutti gli script Python, i download, il training, l'inferenza e l'analisi girano in container. Sull'host servono Bash, Podman, driver NVIDIA, NVIDIA Container Toolkit con CDI e GNU `timeout` per la valutazione. Non occorrono Python, pip, CUDA toolkit o ambienti virtuali sull'host. Il runtime CUDA è nell'immagine; il driver resta sull'host.

I comandi vanno eseguiti dalla radice della copia del repository sul server. Trasferire almeno `CodePromptOpt/` e `.containerignore`, includendo `CodePromptOpt/data/` ed escludendo `.runtime/`, cache e vecchi `outputs/`. Non è necessario trasferire i log grezzi `dataset_code_prompt/` per addestrare. Le directory di output e la cache Hugging Face sono persistenti.

## Cosa viene addestrato

| Modello | Modalità | Uso |
| --- | --- | --- |
| Qwen3-8B | LoRA BF16 | Primo esperimento |
| Qwen3-4B | LoRA BF16 | Controllo di scala |
| Qwen3-14B | LoRA BF16 | Controllo di scala |
| Phi-4-mini-instruct | LoRA BF16 | Confronto fra famiglie |
| Qwen3-32B | QLoRA NF4 | Opzionale, escluso dal default |

Non è full fine-tuning: i pesi base sono congelati e vengono addestrati adapter LoRA. Per ogni modello sono previsti Best, Random e Top-k, con seed 42, 43 e 44: **36 training** nella matrice completa. Il primo pilot con un solo seed comprende 3 training; il pilot esteso su 8B con tre seed ne comprende 9. Ogni esecuzione usa un solo processo e una sola GPU; nessun parallelismo fra training. Le configurazioni da 24 GB restano utilizzabili separatamente.

La matrice è in `configs/h200_matrix.json`: 4096 token totali input+output, rank 16, alpha 32, dropout 0,1, LR 5e-5, massimo 5 epoche, batch 1 e accumulazione 8. Si adattano le proiezioni attention: `q_proj/k_proj/v_proj/o_proj` per Qwen, `qkv_proj/o_proj` per Phi. Il carico effettivo va verificato sul server; non viene stimato un tempo di training prima del pilot.

## 1. Preparare il server e costruire l'immagine

Se appare `Trying to pull localhost/codepromptopt-ft:h200` seguito da `connect: connection refused`, l'immagine non è disponibile nello storage Podman dell'utente corrente. Ricostruirla con `bash CodePromptOpt/finetuning/podman.sh build`, poi rilanciare `doctor` e il comando desiderato. Usare sempre lo stesso utente e non alternare Podman rootless con `sudo podman`. Per elencare le immagini disponibili: `podman images`. Il wrapper usa `--pull=never` per l'immagine di lavoro, così non tenta di contattare un registry localhost quando manca. Dataset e adapter montati non vengono eliminati dal build.

```bash
cd /percorso/meta_prompting
nvidia-smi -L
nvidia-ctk cdi list
bash CodePromptOpt/finetuning/podman.sh build
bash CodePromptOpt/finetuning/podman.sh doctor
bash CodePromptOpt/finetuning/podman.sh cpu python -m unittest CodePromptOpt.finetuning.test_pipeline -v
```

`doctor` deve mostrare una sola GPU visibile, il nome atteso e la memoria. Lo script imposta ed esporta automaticamente `CPO_GPU=3` se la variabile non è già definita: non serve eseguire un export manuale. Il wrapper espone `nvidia.com/gpu=3` e imposta nel container `CUDA_VISIBLE_DEVICES=0`, perché la singola GPU host esposta diventa il device logico 0. Non impostare 3 dentro questo container. `CPO_GPU` può anche contenere un UUID CDI verificato. Per usare un'altra GPU in una singola esecuzione: `CPO_GPU=0 bash CodePromptOpt/finetuning/podman.sh doctor`.

Se CDI non elenca il device richiesto, far configurare driver/toolkit all'amministratore. Con toolkit recenti la specifica è gestita dal servizio `nvidia-cdi-refresh`; le procedure e le differenze fra versioni sono nella [documentazione NVIDIA CDI](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html). Non usare `--privileged` per aggirare un errore GPU.

Il wrapper usa mount condivisi con label `:z` e un volume `cpo-hf-cache`. `CPO_IMAGE` e `CPO_CACHE` permettono di cambiare immagine e cache. Usare Podman rootless con lo stesso utente per tutti i comandi: gli output appartengono all'utente host. Se il download richiede autenticazione, esportare `HF_TOKEN`; il wrapper inoltra la variabile senza scriverla negli artefatti.

## 2. Pianificare il dataset e fissare i modelli

Se lo smoke fallisce con `Failed to find C compiler`, aggiornare i file sul server e ricostruire l'immagine: il Containerfile installa `build-essential` e imposta `CC`/`CXX` dentro il container. `doctor` verifica anche il compilatore e l'import PEFT/bitsandbytes che inizializza Triton. Non serve installare GCC sull'host.

```bash
bash CodePromptOpt/finetuning/podman.sh build
bash CodePromptOpt/finetuning/podman.sh doctor
bash CodePromptOpt/finetuning/podman.sh smoke --models qwen3_8b --seeds 42 --strategies best
bash CodePromptOpt/finetuning/podman.sh logs
bash CodePromptOpt/finetuning/podman.sh wait
```

Lo smoke fallito prima del training non ha salvato adapter: si può rilanciare sulla stessa directory. Il wrapper conserva il vecchio container rinominandolo e usa la nuova immagine per il nuovo lancio; non usare `podman start` sul container fallito. Dataset preparati e cache dei pesi vengono riutilizzati.

Il comando seguente prepara tutti i dataset e i 36 file config, senza scaricare i pesi. Le revisioni Hugging Face sono risolte a commit e conservate in `revisions.json`.

```bash
bash CodePromptOpt/finetuning/podman.sh plan --pin-revisions
```

Default output: `CodePromptOpt/finetuning/outputs/h200/`. Il piano completo è in `plan.json`, le configurazioni in `configs/`, gli input derivati in `datasets/`. Per un esperimento distinto passare lo stesso `--output CodePromptOpt/finetuning/outputs/NOME` a ogni comando successivo. Non usare l'output locale di verifica `h200-check`: è un controllo offline non fissato a revisioni.

Protocollo dei dataset:

- Si mantengono i task train/validation/test ufficiali. Per ogni task train si sceglie una sola condizione osservata, ordinata per ID e indipendentemente dallo score; non si confrontano dev suite sconosciute fra file diversi.
- La selezione riusa esclusioni di qualità e conflitti del dataset originale, conserva il problema verbatim e usa soltanto osservazioni dev positive, mai i test finali. Random significa casuale nel pool **valido e positivo**, non in tutti i candidati grezzi.
- Best sceglie lo score dev massimo. Random sceglie un candidato con lunghezza in caratteri fra 0,5 e 2 volte quella del best. Il seed di selezione dataset è 42 ed è fisso per tutti i modelli e seed di training: questa replica misura la variabilità del training, non la variabilità delle estrazioni Random.
- Top-k conserva al massimo 3 candidati entro 0,1 dallo score best; elimina formulazioni con Jaccard dei token separati da spazi almeno 0,9, dopo aver rimosso il problema. È una diversità lessicale euristica, non semantica.
- La validation ha sempre le stesse etichette Best ufficiali. Con il target predefinito `qwen3.5-coder`, il dataset attuale produce Best **66 esempi**, Random **66**, Top-k **182**, sugli stessi **66 task train**; validation **17 esempi su 10 task**. Il test rimane di 10 task.

Per controllare il maggior numero di esempi Top-k si usa un sampler con probabilità uniforme per task: ogni epoca estrae 66 esempi con rimpiazzo, anche per Top-k, prima di eventuali esclusioni per lunghezza. Ogni task ha quindi lo stesso peso atteso; non è garantita la copertura di tutti i task in ogni epoca. Il numero di step per epoca rimane confrontabile quando la copertura dopo tokenizzazione è identica. La loss resta token-level sulle sole risposte, non una media pesata per lunghezza. I report `audit.json` vanno confrontati per verificare le esclusioni per modello/strategia. Non equiparare la lunghezza in caratteri a un budget esatto di token.

## 3. Pilot: tokenizzazione e prova di due step

### Modalità detached e monitoraggio

`smoke`, `train`, `generate` e `gpu` avviano ora un container in **detached mode** (`podman run -d`), chiamato di default `cpo-gpu-3`. Il comando ritorna subito e il lavoro continua dopo la chiusura di SSH. Il container viene conservato dopo la fine, insieme ai log. `build`, `doctor`, `plan`, `prepare`, `cpu` ed `evaluate` restano in foreground.

Prima di avviare la fase successiva, attendere il completamento e controllare l'exit code:

```bash
bash CodePromptOpt/finetuning/podman.sh logs
bash CodePromptOpt/finetuning/podman.sh status
bash CodePromptOpt/finetuning/podman.sh wait
```

`logs` segue l'output: Ctrl+C interrompe soltanto la visualizzazione. `wait` restituisce l'exit code del lavoro (0 indica successo). Per concatenare fasi usare `&&` dopo `wait`; i comandi smoke/train/generate elencati sotto non attendono automaticamente. Non lanciare `prepare` o altri processi GPU mentre un lavoro è attivo.

```bash
bash CodePromptOpt/finetuning/podman.sh smoke --models qwen3_8b --seeds 42 --strategies best
bash CodePromptOpt/finetuning/podman.sh wait && bash CodePromptOpt/finetuning/podman.sh train --models qwen3_8b --seeds 42
```

Un secondo lavoro detached viene rifiutato mentre il container è attivo. Al lancio successivo, un container terminato viene rinominato con timestamp e conservato: il nome archiviato viene stampato e i log rimangono leggibili con `podman logs NOME_ARCHIVIATO`. `CPO_CONTAINER_NAME` consente un nome personalizzato, da usare anche nei comandi di monitoraggio. Mantenere lo stesso nome per evitare training concorrenti sulla stessa GPU.

Per fermare esplicitamente il lavoro usare `bash CodePromptOpt/finetuning/podman.sh stop`; la ripresa avviene dal checkpoint tramite il comando documentato sotto, non con `podman start`. Gli output sul repository e la cache restano persistenti. Detached mode non implica ripresa automatica dopo reboot o crash del server. I container archiviati si possono rimuovere manualmente con `podman rm NOME_ARCHIVIATO` dopo aver conservato i log necessari.

Iniziare da Qwen3-8B con seed 42. `prepare` scarica il tokenizer, esclude senza troncamento le sequenze oltre 4096 e scrive il report; non carica i pesi.

```bash
bash CodePromptOpt/finetuning/podman.sh prepare --models qwen3_8b --seeds 42
bash CodePromptOpt/finetuning/podman.sh smoke --models qwen3_8b --seeds 42 --strategies best
```

Controllare `outputs/h200/runs/qwen3_8b-best-s42/audit.json` ed `excluded.jsonl`, poi la memoria GPU durante lo smoke con `nvidia-smi -i 3`. Lo smoke salva in `outputs/h200/smoke/`, quindi non viene confuso con un risultato sperimentale. Non viene valutato dalla suite. Per ripeterlo usare un nuovo `--output` con un piano corrispondente, oppure una nuova directory tramite il comando `train --max-steps 2` diretto.

## 4. Eseguire i training

Per **riaddestrare da zero** run già presenti aggiungere `--retrain`. Funziona con `train` e `smoke`: prima di ogni run selezionato sposta la vecchia cartella in `outputs/h200/archive/TIMESTAMP/train/` (o `smoke/`), poi riparte dal modello base. Adapter precedenti, checkpoint e vecchi prompt vengono conservati nell'archivio e non si mescolano con quelli nuovi. Il piano, gli split e le revisioni rimangono gli stessi. Senza questo flag resta il comportamento protetto: i training completati vengono saltati.

```bash
# Rifare lo smoke di Qwen3-14B
bash CodePromptOpt/finetuning/podman.sh smoke --models qwen3_14b --seeds 42 --strategies best --retrain
bash CodePromptOpt/finetuning/podman.sh wait

# Rifare tutti i nove training Qwen3-14B
bash CodePromptOpt/finetuning/podman.sh train --models qwen3_14b --retrain
```

Per rifare soltanto un run aggiungere `--seeds 42 --strategies best`. Ogni lancio è detached: aspettare che termini la fase precedente. L'archiviazione è una rinomina nello stesso filesystem, non una copia dei pesi; i vecchi risultati continuano comunque a occupare disco. Il nuovo training scriverà nuovi file. Dopo il retraining occorre rigenerare i prompt; per nuovi test downstream scegliere un nuovo `experiment_id`.

Primo confronto, 3 run:

```bash
bash CodePromptOpt/finetuning/podman.sh train --models qwen3_8b --seeds 42
```

Repliche, 9 run totali su 8B (salta quelli già completati):

```bash
bash CodePromptOpt/finetuning/podman.sh train --models qwen3_8b
```

Controlli di scala e famiglia, altri 27 run:

```bash
bash CodePromptOpt/finetuning/podman.sh prepare --models qwen3_4b qwen3_14b phi4_mini
bash CodePromptOpt/finetuning/podman.sh train --models qwen3_4b qwen3_14b phi4_mini
```

Oppure, dopo il pilot, `bash CodePromptOpt/finetuning/podman.sh train` completa tutta la matrice sequenzialmente in detached mode. Non serve `tmux`. Per seguire e conservare il log:

```bash
bash CodePromptOpt/finetuning/podman.sh train --models qwen3_8b
podman logs -f cpo-gpu-3
podman logs cpo-gpu-3 > h200-8b-training.log 2>&1
```

Ogni cartella `runs/MODELLO-STRATEGIA-sSEED/` contiene `adapter/`, `checkpoints/`, `history.json`, `audit.json`, `excluded.jsonl`, `environment.txt` ed `experiment.json`. L'adapter è il migliore per validation loss. I checkpoint contengono anche gli stati necessari alla ripresa; non sono modelli base completi. Conservare anche l'immagine (`podman image inspect localhost/codepromptopt-ft:h200`) e la matrice utilizzata. La risoluzione del commit è comune alle repliche, e base/SFT usano la stessa precisione.

Se un run si interrompe la suite si ferma e segnala l'output incompleto. Riprenderlo esplicitamente, sostituendo `checkpoint-N` con quello reale:

```bash
bash CodePromptOpt/finetuning/podman.sh gpu python -m CodePromptOpt.finetuning.train \
  --config CodePromptOpt/finetuning/outputs/h200/configs/qwen3_8b-best-s42.json \
  --data CodePromptOpt/finetuning/outputs/h200/datasets/best \
  --output CodePromptOpt/finetuning/outputs/h200/runs/qwen3_8b-best-s42 \
  --resume CodePromptOpt/finetuning/outputs/h200/runs/qwen3_8b-best-s42/checkpoints/checkpoint-N
```

Un adapter già presente viene protetto dalla sovrascrittura. Se il crash è avvenuto prima del primo checkpoint, usare una nuova directory di output per il run diretto. I dettagli dell'ordine di campionamento dopo una ripresa possono dipendere dallo stato ripristinato dal trainer: per repliche scientifiche preferire run ininterrotti.

Per il 32B creare un piano separato, poi ripetere `prepare`, `smoke` e `train` sullo stesso output:

```bash
bash CodePromptOpt/finetuning/podman.sh plan --pin-revisions --models qwen3_32b --seeds 42 --output CodePromptOpt/finetuning/outputs/h200-32b
```

## 5. Generare i prompt e misurare il beneficio

### Liberare la cache fra modelli

Lo script `clear_cache.sh` elimina soltanto le directory dei modelli base nel volume Podman `cpo-hf-cache` (o `CPO_CACHE`). Non modifica adapter, checkpoint, report o modelli Ollama. Viene eseguito sull'host senza avviare container, anche quando il disco è pieno. Non eseguirlo contemporaneamente al lancio di nuovi job: rifiuta la cancellazione se un container attivo usa il volume.

```bash
# Anteprima della cache di un modello
bash CodePromptOpt/finetuning/clear_cache.sh --model qwen3_4b

# Eliminazione effettiva dopo aver completato training/generazione
bash CodePromptOpt/finetuning/clear_cache.sh --model qwen3_4b --delete

# Anteprima di tutti i modelli, poi eliminazione
bash CodePromptOpt/finetuning/clear_cache.sh --all
bash CodePromptOpt/finetuning/clear_cache.sh --all --delete
```

Alias supportati: `qwen3_4b`, `qwen3_8b`, `qwen3_14b`, `qwen3_32b`, `phi4_mini`; si può anche passare `--model Qwen/Qwen3-8B`. Per usare di nuovo un adapter servirà riscaricare il modello base. I container terminati e le immagini Podman non vengono rimossi.

La validation serve alle scelte sperimentali; usare il test finale soltanto quando il protocollo è fissato. Esempio sul pilot completo:

```bash
bash CodePromptOpt/finetuning/podman.sh generate --models qwen3_8b --seeds 42 --split validation
```

Per ogni run produce `validation-original-prompts.jsonl`, `validation-static-prompts.jsonl`, `validation-base-prompts.jsonl` e `validation-sft-prompts.jsonl`. I controlli original/static e base greedy sono ripetuti fra strategie per comodità: non considerarli repliche indipendenti. La suite rifiuta di sovrascrivere file di prompt esistenti. Dopo le scelte finali ripetere con `--split test`, che usa nomi diversi. Non vengono eseguiti test Python sulle risposte dell'optimizer: bisogna prima generare il codice con lo stesso code generator.

Il code generator resta un servizio separato in container. Se ne hai già uno compatibile OpenAI, riutilizzalo; il wrapper lo raggiunge tramite `host.containers.internal` e la porta pubblicata sul server. In alternativa, ecco un server Ollama containerizzato sulla stessa GPU 3, da avviare **dopo** aver completato training e generazione dei prompt per liberare VRAM:

```bash
podman volume create cpo-ollama
podman network create cpo-codegen-net
export CPO_NETWORK=cpo-codegen-net
podman run -d --name cpo-codegen --network cpo-codegen-net --device nvidia.com/gpu=3 \
  -p 127.0.0.1:11434:11434 -v cpo-ollama:/root/.ollama docker.io/ollama/ollama:latest
podman exec cpo-codegen ollama pull MODELLO_RUNTIME_ESATTO
```

Sostituire il placeholder con un modello disponibile. Fissare un tag/digest dell'immagine e il modello scelto prima di raccogliere le misure; `latest` è solo un esempio di avvio. Verificare che il backend produca solo codice, rispetti il budget e non emetta reasoning. Il network dedicato permette al client di raggiungere `cpo-codegen:11434`; mantenere `CPO_NETWORK` esportata nella sessione. Il wrapper `cpu` non usa GPU: tutte le risorse GPU del code generator sono assegnate dal suo container.

Esempio per un file, ripetuto con le stesse opzioni su **original/static/base/SFT e tutte le strategie**:

```bash
bash CodePromptOpt/finetuning/podman.sh cpu python -m CodePromptOpt.finetuning.codegen \
  --prompts CodePromptOpt/finetuning/outputs/h200/runs/qwen3_8b-best-s42/validation-base-prompts.jsonl \
  --output CodePromptOpt/finetuning/outputs/h200/runs/qwen3_8b-best-s42/validation-base-code.jsonl \
  --base-url http://cpo-codegen:11434/v1 --model MODELLO_RUNTIME_ESATTO
```

Per endpoint autenticati esportare `CODEGEN_API_KEY`. Tutte le varianti devono usare stesso modello runtime, versione, quantizzazione, test suite e parametri di decoding. L'etichetta storica `qwen3.5-coder` del dataset non identifica automaticamente un modello scaricabile.

Preparare poi un file `verified_tests.jsonl` **dentro il repository** con una riga per task, `task_id` e `harness` attendibile. Il formato e le limitazioni della mappatura APPS sono descritti nel [README](README.md#test-funzionali-verificati). Le suite complete non sono presenti nei log del dataset: non si possono produrre misure funzionali attendibili senza questi test.

Il wrapper valuta senza Python sull'host e senza socket Podman nel container. Prepara i file in un container CPU, esegue ciascuna soluzione in un nuovo container senza rete/GPU, poi raccoglie i risultati in un altro container CPU:

```bash
bash CodePromptOpt/finetuning/podman.sh evaluate \
  CodePromptOpt/finetuning/outputs/h200/runs/qwen3_8b-best-s42/validation-base-code.jsonl \
  verified_tests.jsonl \
  CodePromptOpt/finetuning/outputs/h200/runs/qwen3_8b-best-s42/validation-base-results.jsonl \
  CodePromptOpt/finetuning/outputs/h200/jobs/qwen3_8b-best-s42-validation-base
```

Usare una cartella jobs nuova per ogni file, ripetere per SFT e confrontare:

```bash
bash CodePromptOpt/finetuning/podman.sh cpu python -m CodePromptOpt.finetuning.compare \
  --baseline CodePromptOpt/finetuning/outputs/h200/runs/qwen3_8b-best-s42/validation-base-results.jsonl \
  --candidate CodePromptOpt/finetuning/outputs/h200/runs/qwen3_8b-best-s42/validation-sft-results.jsonl
```

La sandbox limita RAM a 512 MiB, CPU a 1, processi a 64, tempo a 20 secondi, usa filesystem read-only e utente non root. Usa una macchina dedicata per codice non attendibile: resta isolamento a container, non a kernel separato. L'immagine di test viene risolta a ID e registrata; impostare `CPO_TEST_IMAGE` a un digest per mantenerla identica fra valutazioni. Il confronto controlla anche suite, immagine e timeout. Si misura pass@1 empirica con bootstrap appaiato per task; la frazione dei singoli test passati non è implementata. I tre seed di training non aumentano il numero di problemi test indipendenti: riportare ogni replica e la variabilità, senza aggregare i 10 task come se fossero 30 task diversi.

La diminuzione della loss non dimostra un miglioramento del code generator. La conclusione richiede il confronto funzionale su task held-out e l'ispezione dei prompt per vincoli aggiunti o alterati. Espansione del dataset, nuovi split, DPO e rietichettatura mediante nuove esecuzioni non fanno parte di questa suite.

Riferimenti: [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct), [PEFT LoRA](https://huggingface.co/docs/peft/v0.15.0/package_reference/lora), [quantizzazione PEFT](https://huggingface.co/docs/peft/developer_guides/quantization).


# ELENCO COMANDI
```bash
bash CodePromptOpt/finetuning/podman.sh logs
bash CodePromptOpt/finetuning/podman.sh status
bash CodePromptOpt/finetuning/podman.sh wait
bash CodePromptOpt/finetuning/podman.sh stop
bash CodePromptOpt/finetuning/clear_cache.sh --model qwen3_4b
```

Esegui i comandi dalla radice del repository sul server. Lo script seleziona già la GPU 3 e avvia tutto tramite Podman.
1. Setup iniziale, una sola volta
```
   bash CodePromptOpt/finetuning/podman.sh build
   bash CodePromptOpt/finetuning/podman.sh doctor
   bash CodePromptOpt/finetuning/podman.sh plan --pin-revisions
```
2. Qwen3-8B — partire da questo
```
   bash CodePromptOpt/finetuning/podman.sh prepare --models qwen3_8b
   bash CodePromptOpt/finetuning/podman.sh smoke --models qwen3_8b --seeds 42 --strategies best
   bash CodePromptOpt/finetuning/podman.sh train --models qwen3_8b
```
3. Qwen3-4B
```
   bash CodePromptOpt/finetuning/podman.sh prepare --models qwen3_4b
   bash CodePromptOpt/finetuning/podman.sh smoke --models qwen3_4b --seeds 42 --strategies best
   bash CodePromptOpt/finetuning/podman.sh train --models qwen3_4b
```
4. Qwen3-14B
```
   bash CodePromptOpt/finetuning/podman.sh prepare --models qwen3_14b
   bash CodePromptOpt/finetuning/podman.sh smoke --models qwen3_14b --seeds 42 --strategies best
   bash CodePromptOpt/finetuning/podman.sh train --models qwen3_14b
```
5. Phi-4-mini-instruct
```
   bash CodePromptOpt/finetuning/podman.sh prepare --models phi4_mini
   bash CodePromptOpt/finetuning/podman.sh smoke --models phi4_mini --seeds 42 --strategies best
   bash CodePromptOpt/finetuning/podman.sh train --models phi4_mini
```
Ogni comando train esegue 9 training: Best, Random e Top-k, ciascuno con seed 42, 43 e 44. Per un primo esperimento più breve aggiungi --seeds 42: saranno 3 training per modello.
6. Generazione dei prompt sulla validation, dopo i training
```
   bash CodePromptOpt/finetuning/podman.sh generate --models qwen3_8b --split validation
   bash CodePromptOpt/finetuning/podman.sh generate --models qwen3_4b --split validation
   bash CodePromptOpt/finetuning/podman.sh generate --models qwen3_14b --split validation
   bash CodePromptOpt/finetuning/podman.sh generate --models phi4_mini --split validation
```
   Se hai addestrato soltanto il seed 42, aggiungi --seeds 42 anche qui. Usa --split test solo dopo aver fissato le scelte sperimentali.
7. Qwen3-32B — opzionale, con output separato
```
   bash CodePromptOpt/finetuning/podman.sh plan --pin-revisions --models qwen3_32b --output CodePromptOpt/finetuning/outputs/h200-32b
   bash CodePromptOpt/finetuning/podman.sh prepare --models qwen3_32b --output CodePromptOpt/finetuning/outputs/h200-32b
   bash CodePromptOpt/finetuning/podman.sh smoke --models qwen3_32b --seeds 42 --strategies best --output CodePromptOpt/finetuning/outputs/h200-32b
   bash CodePromptOpt/finetuning/podman.sh train --models qwen3_32b --output CodePromptOpt/finetuning/outputs/h200-32b
```
I risultati principali finiscono in CodePromptOpt/finetuning/outputs/h200/runs/. La generazione dei prompt va poi seguita da code generation e test funzionali, descritti nella [guida server](C:/Users/flavi/Desktop/meta_prompting/CodePromptOpt/finetuning/SERVER_H200.md).
