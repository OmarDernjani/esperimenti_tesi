# Fine-tuning di prompt optimizer

**Nuovo percorso per H200:** seguire [SERVER_H200.md](SERVER_H200.md). Include LoRA BF16, Qwen3-4B/8B/14B e Phi-4-mini, dataset Best/Random/Top-k, tre seed e un wrapper Podman per ogni fase. Le istruzioni seguenti descrivono il percorso originale QLoRA da 24 GB; la guida H200 lo estende e include anche la valutazione senza Python installato sull'host.

Questa cartella addestra **ottimizzatori di prompt**, da mettere prima di un code generator mantenuto fisso. Il percorso originale include Qwen3-4B e Phi-4-mini-instruct (3,8B, non Phi-4 da 14B), SFT QLoRA, generazione, chiamate a un code generator compatibile OpenAI e valutazione funzionale. Le configurazioni originali sono un punto di partenza per una GPU NVIDIA da 24 GB; quelle H200 aggiungono LoRA BF16 e ablation Random/Top-k. La memoria effettiva va misurata con il proprio dataset. DPO non è implementato.

Eseguire tutti i comandi **dalla radice del repository**. Output e adapter vanno in directory distinte per ogni modello/seed. Il dataset originale non viene modificato.

## Dati e protocollo

Gli split ufficiali per task sono conservati; i test non entrano nel training. Il default seleziona esclusivamente l'etichetta storica `qwen3.5-coder`. Per studiare `qwen3.5`, cambiare `target_label` in entrambe le configurazioni e usare nuove directory di output. Queste etichette non identificano un modello runtime esatto: `codegen --model` deve specificare quello effettivamente disponibile.

Si conservano tutte le osservazioni SFT esportate per quel target, anche quando un task ha più condizioni indipendenti. Non si effettua un ranking nuovo fra condizioni. Il report registra esempi e task, checksum degli input e commit risolti dei modelli. Per una replica esatta impostare `revision` a un commit Hugging Face nei JSON **prima** di training e inferenza, archiviare configurazioni, ambiente (`pip freeze`) e immagine container. `main` è il default pratico per il download iniziale.

La loss è calcolata solo sui token della risposta, incluso EOS. Il prompt di ingresso usa lo stesso template del dataset. Qwen usa `enable_thinking=False`; la risposta appresa contiene direttamente il prompt ottimizzato. Non si tronca il problema: gli esempi oltre il budget totale input+output vengono esclusi e registrati in `excluded.jsonl`. `--prepare-only` scarica solo il tokenizer e mostra l'effettiva copertura prima di impegnare la GPU. Early stopping sulla validation loss e salvataggio del miglior adapter; test riservato al confronto finale.

## GPU dedicata: numero 3

Training e inferenza degli optimizer selezionano di default la **GPU fisica 3** (indice zero-based: la quarta GPU). Prima di importare PyTorch impostano `CUDA_VISIBLE_DEVICES=3` e, se assente, `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Nel processo la GPU selezionata diventa `cuda:0`, per questo `device_map` continua a usare 0. Se la GPU richiesta non esiste, il programma si interrompe senza ripiegare su altre GPU.

Un `CUDA_VISIBLE_DEVICES` preesistente viene rispettato per supportare container e scheduler. Per garantire la selezione locale della GPU 3 impostare esplicitamente:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
```

In PowerShell:

```powershell
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "3"
```

Il programma verifica che sia visibile una sola GPU e ne stampa il nome. Avviare con `python` senza launcher multiprocesso. Con Podman, CDI espone solo la GPU host 3; nel container diventa 0, quindi i comandi sotto passano `CUDA_VISIBLE_DEVICES=0`. Verificare gli indici sul server con `nvidia-smi -L` e `nvidia-ctk cdi list`. Il code generator esterno mantiene una propria configurazione GPU.

## Uso locale (Linux / WSL2 con CUDA)

```bash
python -m venv .venv-ft
source .venv-ft/bin/activate
pip install -r CodePromptOpt/finetuning/requirements.txt
python -m unittest CodePromptOpt.finetuning.test_pipeline -v
python -m CodePromptOpt.finetuning.train --config CodePromptOpt/finetuning/configs/qwen3_4b.json --output CodePromptOpt/finetuning/outputs/qwen --audit-only
python -m CodePromptOpt.finetuning.train --config CodePromptOpt/finetuning/configs/qwen3_4b.json --output CodePromptOpt/finetuning/outputs/qwen --prepare-only
python -m CodePromptOpt.finetuning.train --config CodePromptOpt/finetuning/configs/qwen3_4b.json --output CodePromptOpt/finetuning/outputs/qwen
python -m CodePromptOpt.finetuning.train --config CodePromptOpt/finetuning/configs/phi4_mini.json --output CodePromptOpt/finetuning/outputs/phi
```

`--audit-only` usa solo la libreria standard. `--resume percorso/checkpoint-N` riprende il training. Per problemi di memoria ridurre `max_length` dopo aver esaminato le esclusioni; ridurre il rank è un'altra possibilità. Batch 1, accumulazione 8, NF4, LoRA 16 e checkpointing sono già configurati. BF16 viene selezionato se supportato, altrimenti FP16.

## Podman

Build dalla radice, con contesto limitato a `CodePromptOpt` tramite `.containerignore` fornito. Occorrono Podman e GPU NVIDIA esposta al runtime tramite NVIDIA Container Toolkit/CDI. Su Windows la VM/WSL deve vedere CUDA: un container non abilita da solo il passthrough della GPU.

```bash
podman build -t codepromptopt-ft -f CodePromptOpt/finetuning/Containerfile .
podman volume create cpo-hf-cache
podman run --rm --device nvidia.com/gpu=3 -e CUDA_VISIBLE_DEVICES=0 codepromptopt-ft python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

Da Bash/WSL, montare il repository per rendere persistenti i risultati:

```bash
podman run --rm --device nvidia.com/gpu=3 -e CUDA_VISIBLE_DEVICES=0 --shm-size=8g -v "$PWD:/workspace:Z" -v cpo-hf-cache:/cache codepromptopt-ft python -m CodePromptOpt.finetuning.train --config CodePromptOpt/finetuning/configs/qwen3_4b.json --output CodePromptOpt/finetuning/outputs/qwen
```

Da PowerShell lo stesso comando usa `-v "${PWD}:/workspace"` (senza `:Z` su filesystem Windows). Sostituire il file config per Phi. Le esecuzioni sono sequenziali su una GPU. Riutilizzare la stessa immagine e cache per generare i prompt. La valutazione con Podman si esegue **sull'host** usando Python standard, senza montare il socket Podman nel container di training.

## Esperimento base contro SFT

Prima usare `--split validation` per le decisioni sperimentali. Per la misura finale, generare quattro varianti sugli stessi task test. Esempio Qwen:

```bash
python -m CodePromptOpt.finetuning.generate --config CodePromptOpt/finetuning/configs/qwen3_4b.json --variant original --output CodePromptOpt/finetuning/outputs/original.jsonl
python -m CodePromptOpt.finetuning.generate --config CodePromptOpt/finetuning/configs/qwen3_4b.json --variant static --output CodePromptOpt/finetuning/outputs/static.jsonl
python -m CodePromptOpt.finetuning.generate --config CodePromptOpt/finetuning/configs/qwen3_4b.json --variant base --output CodePromptOpt/finetuning/outputs/qwen-base.jsonl
python -m CodePromptOpt.finetuning.generate --config CodePromptOpt/finetuning/configs/qwen3_4b.json --variant sft --adapter CodePromptOpt/finetuning/outputs/qwen/adapter --output CodePromptOpt/finetuning/outputs/qwen-sft.jsonl
```

Ripetere base/SFT con Phi e il relativo adapter. I controlli original/static sono condivisi. I record conservano preservazione verbatim del problema, lunghezza, latenza e troncamento. La preservazione verbatim è un controllo strutturale, non certifica l'assenza di istruzioni contraddittorie: rivedere manualmente gli output. Non scartare i prompt invalidi dal denominatore finale. Il decoding dell'optimizer è greedy e il budget è identico per base/SFT.

Avviare separatamente il code generator (per esempio Ollama o vLLM), poi inviare ogni file di prompt allo stesso endpoint con gli stessi parametri:

```bash
python -m CodePromptOpt.finetuning.codegen --prompts CodePromptOpt/finetuning/outputs/qwen-base.jsonl --output CodePromptOpt/finetuning/outputs/qwen-base-code.jsonl --base-url http://localhost:11434/v1 --model MODELLO_RUNTIME_ESATTO
python -m CodePromptOpt.finetuning.codegen --prompts CodePromptOpt/finetuning/outputs/qwen-sft.jsonl --output CodePromptOpt/finetuning/outputs/qwen-sft-code.jsonl --base-url http://localhost:11434/v1 --model MODELLO_RUNTIME_ESATTO
```

Per autenticazione impostare `CODEGEN_API_KEY` nell'ambiente. Nel container l'host si raggiunge generalmente tramite `host.containers.internal`. Il backend deve supportare i parametri inviati; il seed non garantisce determinismo su ogni server. Si richiede una soluzione Python completa: alcuni generatori restituiscono solo il corpo funzione o testo di reasoning, da correggere nel protocollo prima dell'esperimento. Archiviare versione, quantizzazione e impostazioni del server target. La prima fence viene estratta come codice; l'assenza di fence conserva il testo intero.

## Test funzionali verificati

I log CodePromptOpt non forniscono suite eseguibili complete: preparare un JSONL esterno con una riga per task held-out e un `harness` Python attendibile. Esempio di formato (ID e funzione sono illustrativi):

```json
{"task_id":"task_ID_REALE", "harness":"from solution import add\nassert add(1, 2) == 3\nassert add(-1, 1) == 0\n"}
```

Per HumanEval+ è possibile associare la suite ufficiale al `source_id` verificato in `tasks.jsonl`: l'harness importa la funzione da `solution` e invoca i test. Per APPS serve una mappatura verificata al benchmark originale, poiché l'indice locale non basta. Non dedurre i test dalle risposte né usare soltanto gli esempi nella descrizione come misura del benchmark. L'harness deve terminare con exit code diverso da zero quando fallisce qualsiasi test. Gestire esplicitamente stdin/stdout per APPS e import dipendenti per HumanEval. Per copertura parziale creare prima lo stesso sottoinsieme di sample per tutte le varianti e dichiarare la copertura.

```bash
podman pull docker.io/library/python:3.11-slim
python -m CodePromptOpt.finetuning.evaluate --samples CodePromptOpt/finetuning/outputs/qwen-base-code.jsonl --tests /path/to/verified_tests.jsonl --output CodePromptOpt/finetuning/outputs/qwen-base-results.jsonl
python -m CodePromptOpt.finetuning.evaluate --samples CodePromptOpt/finetuning/outputs/qwen-sft-code.jsonl --tests /path/to/verified_tests.jsonl --output CodePromptOpt/finetuning/outputs/qwen-sft-results.jsonl
python -m CodePromptOpt.finetuning.compare --baseline CodePromptOpt/finetuning/outputs/qwen-base-results.jsonl --candidate CodePromptOpt/finetuning/outputs/qwen-sft-results.jsonl
```

La sandbox usa rete disabilitata, root filesystem in sola lettura, utente non root, limiti CPU/RAM/processi e timeout. Per codice non attendibile usare una macchina/VM dedicata; Podman condivide comunque il kernel. Non passare credenziali o socket alla sandbox. `--image` permette di fissare un digest immagine. Mancanza di harness e problemi del runtime interrompono la valutazione anziché diventare falsi fallimenti del modello.

La metrica è pass@1 empirica (una generazione per task); il report include il delta e un intervallo bootstrap appaiato per problema. Non calcola la frazione dei singoli test passati. Con soli dieci task test gli intervalli saranno ampi: evitare conclusioni forti e replicare eventualmente su ulteriori split/semi con protocollo dichiarato prima dell'esperimento. Per confrontare risultati usare la stessa suite e immagine di test; lo script verifica task, target e decoding del code generator.

## Riferimenti verificati

- [Qwen3-4B e modalità non-thinking](https://huggingface.co/Qwen/Qwen3-4B)
- [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [PEFT QLoRA](https://huggingface.co/docs/peft/developer_guides/quantization)
- [Podman con NVIDIA CDI](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)

Le dipendenze fissano una baseline di versioni; la validazione CUDA e il training completo vanno eseguiti sull'host GPU. Nessun risultato di miglioramento viene presupposto.
