# Passare dal 7B a Qwen3-Coder-30B sulla H200

I nuovi file `h200_qwen3.sh` e `qwen3_coder_30b.env` definiscono un profilo separato. Trasferirli sul server con gli altri script aggiornati. Non modificano `h200.env` o i risultati del 7B. Usare sempre `h200_qwen3.sh` per il nuovo modello, anche per ready/logs/stop/free-cache.

## 1. Liberare GPU e, facoltativamente, disco del 7B

Dalla radice del repository sul server, dopo aver terminato le richieste del client:

```bash
cd /home/agostinelli/esperimenti_tesi

# Profilo VECCHIO: salva i log, ferma e rimuove il server 7B.
bash CodePromptOpt/finetuning/downstream/h200.sh stop
nvidia-smi -i 3

# Facoltativo: elimina soltanto il volume dei pesi base del code generator 7B.
# Risultati, risposte SQLite, adapter fine-tuned e immagine vLLM rimangono.
bash CodePromptOpt/finetuning/downstream/h200.sh free-cache
df -h .
```

Fermare il processo rilascia la memoria GPU; eliminare la cache recupera spazio su disco. Non serve un reset GPU o del server. Se il vecchio container e gia stato rimosso non occorre ripetere stop. Non eliminare un volume che serve ad altri container.

## 2. Avviare il nuovo profilo

```bash
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh config
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh build
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh start
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh logs
```

Il profilo usa:

- modello `Qwen/Qwen3-Coder-30B-A3B-Instruct`;
- GPU ereditata da `podman.env` (3 di default);
- server detached `cpo-vllm-qwen3-30b-h200`;
- porta host 8001, interna 8000; se occupata cambiare `QWEN3_CODEGEN_PORT` nel nuovo `.env`;
- cache dedicata `cpo-vllm-qwen3-30b-hf-h200`;
- esperimento `qwen3-coder-30b-h200-v1`;
- stesso input di `h200.env`, quindi gli optimizer non vengono ricaricati o riaddestrati.

Il modello ha 30,5B parametri totali e 3,3B attivati per token: A3B non significa che i pesi da caricare siano solo 3B. Il repository BF16 e circa 61 GB; serve inoltre spazio per immagine e cache di compilazione. Contesto 16384 e massimo 4 sequenze ereditati sono un punto di partenza: verificare startup e consumo reale con `nvidia-smi`. Non caricare il vecchio generatore o un optimizer contemporaneamente sulla GPU 3.

Il modello e non-thinking e non richiede `enable_thinking=False`. La scheda ufficiale documenta il serving vLLM: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct . Manteniamo temperatura 0 e budget di output 2048 per il confronto controllato con il 7B; le raccomandazioni generiche di sampling del produttore sono un protocollo distinto.

`start` scarica i pesi, risolve il commit, registra l'immagine e avvia il server. Attendere la fine del caricamento. Ctrl+C chiude solo la visualizzazione dei log.

```bash
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh ready

# Audit e pilot sui controlli original/statico di un solo run.
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh plan --runs qwen3_14b-best-s42 --variants original static
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh run --runs qwen3_14b-best-s42 --variants original static
```

Il pilot usa i dieci task validation (HumanEval+ e APPS) con il protocollo originale di input. I problemi di nome funzione/ripetizioni non sono stati corretti implicitamente. Questo esperimento isola il cambio di code generator. Per correggere anche il contratto di interfaccia bisogna registrare un altro protocollo/experiment_id, senza mescolare i due effetti.

Se startup fallisce per memoria, leggere i log prima di intervenire. Eventuali cambi a `CODEGEN_CONTEXT`, `CODEGEN_GPU_UTIL` o `CODEGEN_MAX_SEQS` in `h200.env` modificano i parametri registrati: scegliere un nuovo ID nel profilo oppure ripristinare quelli originali. Non cancellare config/risposte per aggirare il controllo di identita.

## 3. Batteria completa, quando il pilot e verificato

### Se il download fallisce con CAS/Xet

Un traceback `File reconstruction error: CAS Client Error` / `error sending request ... cdn.hf.co/xorbs/...` indica un fallimento nel download dei pesi. Non equivale a CUDA OOM. Il profilo imposta ora `HF_HUB_DISABLE_XET=1`, `HF_HUB_DOWNLOAD_TIMEOUT=120`, `HF_HUB_ETAG_TIMEOUT=60`; `serve_vllm.sh` inoltra queste variabili al processo del container prima dell'import di Hugging Face.

Aggiornare i due file sul server, poi:

```bash
# Conserva log e rimuove il container fallito; NON elimina il volume.
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh stop
df -h .
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh start
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh logs
```

I file completati nella cache possono essere riutilizzati. Non usare `free-cache` in questo caso. Se il problema continua anche senza hf-xet, verificare l'accesso HTTPS dal container ai domini Hugging Face/CDN, DNS, proxy e certificati; il messaggio allegato non identifica quale di questi sia la causa di rete. Non disabilitare TLS. Un export preesistente di `HF_HUB_DISABLE_XET=0` nella shell prevale sul default: usare `unset HF_HUB_DISABLE_XET` prima di rilanciare per applicare il valore del profilo.

Riferimento: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisablexet . Cambiare il trasporto non richiede un nuovo experiment_id, poiche commit/pesi e parametri di generazione restano gli stessi.

```bash
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh run --run-glob 'qwen3_4b-*' 'qwen3_8b-*' 'qwen3_14b-*'
```

Le risposte gia riuscite del pilot vengono riutilizzate. Le altre varianti vengono aggiunte senza riscrivere i prompt o i risultati del 7B. Il client e in foreground e supporta ripresa con lo stesso comando.

Output:

```text
CodePromptOpt/finetuning/outputs/codegen/qwen3-coder-30b-h200-v1/validation/
```

## 4. Test e pulizia

```bash
# A richieste finite, libera la GPU e conserva log/risultati.
bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh stop

# Valuta con gli stessi test e la stessa sandbox dei risultati 7B.
bash CodePromptOpt/finetuning/downstream/evaluate_suite.sh \
  CodePromptOpt/finetuning/outputs/codegen/qwen3-coder-30b-h200-v1/validation verified_tests.jsonl

# Facoltativo, a esperimento terminato: elimina i pesi base del solo 30B.
# bash CodePromptOpt/finetuning/downstream/h200_qwen3.sh free-cache
```

Se hai eseguito soltanto il pilot, il wrapper valuta i due file presenti; evita di rilanciare in seguito sulla stessa cartella con job gia esistenti. Puoi attendere di completare tutta la code generation prima della sandbox, oppure valutare successivamente i singoli file nuovi con `podman.sh evaluate` e percorsi jobs distinti.
