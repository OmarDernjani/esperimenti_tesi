# DPO di Qwen3-8B su H200, con Podman

Questo esperimento addestra l'ottimizzatore dei prompt, non il code generator.
Tutto il codice Python viene eseguito nei container. GPU host predefinita: **3**;
nel container appare come `cuda:0`. Impostazioni persistenti in `podman.env` di
questa cartella, indipendenti da quelle della batteria SFT.

## Dati e significato dell'esperimento

Il punto di partenza sono le preferenze gia valutate in
`CodePromptOpt/data/dpo/train.jsonl` e `validation.jsonl`, con i task canonici
in `CodePromptOpt/data/tasks.jsonl`. Non vengono letti i dati DPO del test.
Per ogni coppia, chosen ha un punteggio maggiore di rejected per lo stesso
task e gruppo di confronto del dataset sorgente. Richiediamo provenienza delle
valutazioni, problema canonico identico e conservazione del problema nel chosen.
Non correggiamo i prompt automaticamente e non eseguiamo il codice generato qui.

Default: target `qwen3.5-coder`, margine minimo 0.1, al massimo quattro coppie
per task, selezionate per margine decrescente. L'audit locale ha selezionato:

| Split | Coppie | Task distinti |
|---|---:|---:|
| Training | 65 | 17 |
| Validation delle preferenze | 12 | 3 |

Il controllo dei token puo escludere altre coppie; sono registrate in
`excluded.jsonl`, senza troncamento. La validation funzionale resta sui dieci
task canonici gia usati, non solo sui tre con preferenze disponibili.
Il limite per task riduce la concentrazione; non rende tutti i task esattamente equiponderati.

**E un pilot con pochi problemi indipendenti.** Le preferenze misurano il coder
sorgente `qwen3.5-coder`: non sono nuove ricompense di Qwen3-Coder-30B.
Valutare con quest'ultimo misura il trasferimento delle preferenze. Per allineare
direttamente al 30B occorre prima costruire nuove coppie valutate dal 30B sui
soli problemi di training, conservando provenienza e condizioni comparabili.

Default: DPO direttamente da `Qwen/Qwen3-8B`, che e gia instruction-tuned.
La reference e la stessa base congelata, ottenuta disattivando LoRA.
E disponibile anche SFT -> DPO: la policy e una copia addestrabile dell'adapter
SFT, la reference una copia fissa dello stesso adapter. Non si usa la base senza
SFT come reference in quel caso. I log-prob della reference sono precalcolati;
non viene allocata una seconda copia completa dei pesi della base.

LoRA BF16, rank 16, alpha 32, attenzione q/k/v/o, dropout disabilitato,
LR 5e-6, beta 0.1, una epoca, batch 1, accumulo 8, contesto totale 4096.
Il prefisso Qwen3 usa `enable_thinking=False`, come la generazione esistente.
Modello/tokenizer usano il commit `b968826d9c46dd6066d109eabc6255188de91218`,
registrato anche nelle run SFT Qwen3-8B locali. Una revisione simbolica eventualmente
configurata viene risolta una volta prima di caricare tokenizer e pesi.

## Comandi sul server

Trasferire i sorgenti aggiornati, inclusa questa intera cartella e
`CodePromptOpt/finetuning/{common.py,generate.py}`. I bind mount usano i sorgenti
presenti sul server: ricostruire l'immagine non aggiorna file mancanti sull'host.
Eseguire dalla radice `~/esperimenti_tesi`.

Prima fermare il coder se ancora attivo (usare il comando del profilo in uso):

```bash
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh stop
nvidia-smi
df -h .
```

Se il container del coder non esiste, non occorre eseguire `stop`.
Controllare che la GPU 3 sia disponibile. Il wrapper blocca sovrapposizioni
con i nomi standard dei container del progetto, ma non prenota la GPU su un server condiviso.

Build, verifica CPU del trainer su un modello casuale minuscolo senza download,
preparazione delle preferenze:

```bash
bash CodePromptOpt/finetuning/dpo/podman.sh config
bash CodePromptOpt/finetuning/dpo/podman.sh build
bash CodePromptOpt/finetuning/dpo/podman.sh check
bash CodePromptOpt/finetuning/dpo/podman.sh prepare
```

`check` deve terminare con `PASS`. Verifica un aggiornamento DPO, la reference
SFT invariata e il salvataggio del solo adapter policy. `prepare` non scarica
modelli. Se la directory dati esiste gia, il comando si ferma: conservarla e
proseguire, oppure scegliere una nuova root persistente per un esperimento nuovo.

Controllo dei token e smoke di due optimizer step, separato dal training vero:

```bash
bash CodePromptOpt/finetuning/dpo/podman.sh tokens --seed 42
bash CodePromptOpt/finetuning/dpo/podman.sh await-job
bash CodePromptOpt/finetuning/dpo/podman.sh smoke --seed 42
bash CodePromptOpt/finetuning/dpo/podman.sh await-job
```

Training e generazione validation per il primo seed:

```bash
bash CodePromptOpt/finetuning/dpo/podman.sh train --seed 42
bash CodePromptOpt/finetuning/dpo/podman.sh await-job
bash CodePromptOpt/finetuning/dpo/podman.sh generate --seed 42 --split validation
bash CodePromptOpt/finetuning/dpo/podman.sh await-job
```

Gli altri due seed, in sequenza:

```bash
for seed in 43 44; do
  bash CodePromptOpt/finetuning/dpo/podman.sh train --seed "$seed" || break
  bash CodePromptOpt/finetuning/dpo/podman.sh await-job || break
  bash CodePromptOpt/finetuning/dpo/podman.sh generate --seed "$seed" --split validation || break
  bash CodePromptOpt/finetuning/dpo/podman.sh await-job || break
done
```

Ogni job GPU parte detached. Ctrl+C durante `logs` o `wait` interrompe
l'osservazione, non il training. In un ciclo shell, chiudere il terminale puo
impedire l'avvio dei job successivi: il job gia detached continua.
`await-job` salva i log su host e rimuove il container solo se exit 0.

```bash
bash CodePromptOpt/finetuning/dpo/podman.sh status
bash CodePromptOpt/finetuning/dpo/podman.sh logs
bash CodePromptOpt/finetuning/dpo/podman.sh wait
```

In caso di errore, il container e mantenuto. Leggere prima i log; dopo aver
salvato cio che serve rimuovere solo quel container fermo con
`podman rm cpo-dpo-gpu-3`. Una run parziale non viene sovrascritta: cambiare
`--run-name` per ripartire, oppure archiviare la sua directory su host.
Non viene effettuato resume dei pesi o dello stato optimizer.

## Variante: partire dall'adapter SFT

Alternativa al training diretto precedente, non un comando necessario.
Usare un adapter Qwen3-8B dello stesso modello/revisione della base.
Esempio per Best, seed 42, se il percorso sul server coincide:

```bash
bash CodePromptOpt/finetuning/dpo/podman.sh smoke --seed 42 \
  --run-name qwen3_8b-sft-dpo-best-s42 \
  --init-adapter CodePromptOpt/finetuning/outputs/runs/qwen3_8b-best-s42/adapter
bash CodePromptOpt/finetuning/dpo/podman.sh await-job
bash CodePromptOpt/finetuning/dpo/podman.sh train --seed 42 \
  --run-name qwen3_8b-sft-dpo-best-s42 \
  --init-adapter CodePromptOpt/finetuning/outputs/runs/qwen3_8b-best-s42/adapter
bash CodePromptOpt/finetuning/dpo/podman.sh await-job
bash CodePromptOpt/finetuning/dpo/podman.sh generate --seed 42 \
  --run-name qwen3_8b-sft-dpo-best-s42 --split validation
bash CodePromptOpt/finetuning/dpo/podman.sh await-job
```

Se i vecchi adapter sono in `outputs/h200/runs`, usare quel percorso.
Per gli altri seed scegliere l'adapter SFT corrispondente. La generazione legge
l'adapter iniziale dai metadati del training e crea anche una cartella
`<run-name>-initial-sft` per il confronto appaiato con la stessa inizializzazione.
L'adapter iniziale deve rimanere disponibile e invariato.

## Output e metriche

Root predefinita: `CodePromptOpt/finetuning/outputs/dpo-qwen3-8b-v1/`.

- `data/`: coppie selezionate, hash e audit dei dati.
- `tokens/` e `smoke/`: prove separate, non usate come training finale.
- `runs/qwen3_8b-dpo-s42/adapter/`: adapter DPO e tokenizer.
- `runs/.../{config,audit,history,metrics}.json`: commit, conteggi, loss e metriche.
- `runs/.../validation-{original,static,base,sft}-prompts.jsonl`: prompt da valutare.
- `logs/`: log salvati da `await-job`.

Per compatibilita col client esistente, la variante **`sft` contiene l'adapter
DPO** nelle cartelle DPO; il nome della run distingue l'esperimento.
`base` rimane Qwen3-8B senza adapter. Nelle cartelle `-initial-sft`, `sft`
contiene invece l'adapter SFT iniziale, prima di DPO.

`metrics.json` contiene valutazione prima/dopo e metriche di training.
`rewards/accuracies` misura l'ordinamento delle preferenze rispetto alla
reference; **non e l'accuracy del codice**. Il Pass@1 richiede coder e sandbox.
Non vengono salvati checkpoint intermedi o una copia completa dei pesi base.
Le run finali non vengono sovrascritte e non sono saltate silenziosamente.

## Passaggio al code generator e sandbox

Dopo tutti i job DPO, la memoria GPU e liberata automaticamente alla loro
terminazione. Conservare la cache 8B fino al termine della generazione dei
prompt. Per liberare poi lo spazio dei pesi base, senza cancellare gli adapter:

```bash
bash CodePromptOpt/finetuning/clear_cache.sh --model qwen3_8b
bash CodePromptOpt/finetuning/clear_cache.sh --model qwen3_8b --delete
```

Questo usa la cache standard `cpo-hf-cache`. Se `CPO_DPO_CACHE` e stato cambiato,
passare lo stesso volume come `CPO_CACHE` a `clear_cache.sh`.
La pulizia richiede che nessun container in esecuzione usi quel volume.

Per una nuova valutazione col 30B, `codegen.sh` imposta input e ID esperimento
persistenti da `dpo/podman.env`, senza export da ricordare nel terminale.
Il default e `qwen3-coder-30b-dpo8b-v1`, distinto dalle vecchie valutazioni.

```bash
bash CodePromptOpt/finetuning/dpo/codegen.sh build
bash CodePromptOpt/finetuning/dpo/codegen.sh start
bash CodePromptOpt/finetuning/dpo/codegen.sh logs
# Quando pronto, Ctrl+C dai log e verificare l'API:
bash CodePromptOpt/finetuning/dpo/codegen.sh ready
bash CodePromptOpt/finetuning/dpo/codegen.sh run --run-glob 'qwen3_8b-*'
# Attendere il completamento del client prima di fermare il coder:
bash CodePromptOpt/finetuning/dpo/codegen.sh stop
bash CodePromptOpt/finetuning/downstream/evaluate_suite.sh \
  CodePromptOpt/finetuning/outputs/codegen/qwen3-coder-30b-dpo8b-v1/validation \
  verified_tests.jsonl
```

Per il confronto con vecchie run usare lo stesso commit del coder, immagine,
decoding, task e test; fissare `QWEN3_CODEGEN_REVISION` al commit gia registrato
nel vecchio `server-config.json` prima di `start`.
Il riepilogo SFT `downstream.summarize_results` assume la matrice di 27 run SFT
e non va usato per questa matrice DPO. Per un confronto appaiato del seed 42:

```bash
CPO_IMAGE=localhost/cpo-codegen-client:1 bash CodePromptOpt/finetuning/podman.sh cpu \
  python -m CodePromptOpt.finetuning.compare \
  --baseline CodePromptOpt/finetuning/outputs/codegen/qwen3-coder-30b-dpo8b-v1/validation/qwen3_8b-dpo-s42/validation-base-results.jsonl \
  --candidate CodePromptOpt/finetuning/outputs/codegen/qwen3-coder-30b-dpo8b-v1/validation/qwen3_8b-dpo-s42/validation-sft-results.jsonl
```

Ripetere con baseline `validation-original-results.jsonl`. Per SFT -> DPO,
confrontare inoltre i risultati `sft` della cartella `-initial-sft` con quelli
`sft` della run DPO. Le ripetizioni dei controlli non sono task indipendenti.
Generare sul test (`generate --split test`) solo dopo aver fissato il protocollo;
richiede test verificati del relativo split per l'esecuzione nella sandbox.

Implementazione basata su [TRL DPO 0.17.0](https://huggingface.co/docs/trl/v0.17.0/dpo_trainer),
con dipendenze separate per non modificare l'ambiente SFT precedente.
