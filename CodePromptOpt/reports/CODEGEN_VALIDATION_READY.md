# Risultati code generation e preparazione dei test

Input locale: `CodePromptOpt/finetuning/outputs/codegen/qwen25-coder-7b-h200-v1/validation/`.

- 27 run Qwen (4B, 8B, 14B; Best/Random/Top-k; tre seed), 108 file, 1080 risposte.
- `status.json`: 108 file completi, nessuna richiesta fallita o file incompleto.
- 1079 risposte con `finish_reason=stop`, una con `finish_reason=length`.
- Controllo sintattico AST, senza esecuzione: 59 righe non sono Python sintatticamente valido (18 base, 41 SFT); zero in original/static.
- Le righe ripetute fra controlli non sono generazioni indipendenti: mantenere gli ID delle richieste condivise.

Questi controlli non misurano la correttezza funzionale.

## Suite recuperate

`verified_tests.jsonl` nella radice del repository contiene ora gli harness per tutti i 10 task validation: 5 HumanEval+ e 5 APPS. `verified_tests.jsonl.mapping.json` documenta task, conteggi, corrispondenze, revisioni e checksum delle fonti. Non sono stati inventati test o risultati attesi e non sono state usate soluzioni generate come oracolo.

- HumanEval+: ID originale, corrispondenza del problema dopo rimozione del solo whitespace esterno, entry point uguale. Viene invocata la funzione `check(candidate)` del dataset originale.
- APPS: corrispondenza esatta e univoca del testo originale in tutti i record train della revisione sorgente; ID documentato tramite URL del problema. Non si ricostruisce un ID dal numero locale degli esperimenti.
- Fonti fissate: `evalplus/humanevalplus` commit `d32357cf319e50e9c8d8dab5ea876c72b0fd321b`, `codeparrot/apps` commit `21e74ddf8de1a21436da12e3e653065c5213e9d1`.
- Sono usati tutti i test disponibili nei record sorgenti selezionati, non la suddivisione interna dev/test storica. I task restano quelli della validation held-out.
- APPS conserva la convenzione di `utils.py` per output singleton, normalizzazione tuple/set e confronto stdout con whitespace finale per riga rimosso. Due task sono stdin/stdout, tre call-based.

I soli record sorgenti selezionati sono in `outputs/benchmark_sources/validation/`; il download viene letto in streaming senza conservare l'intero dataset APPS. Per ricreare il file da questi record, senza rete:

```bash
python -m CodePromptOpt.finetuning.downstream.prepare_tests
```

Per recuperare nuovamente le fonti pubbliche aggiungere `--download`. La preparazione non esegue il codice generato o i test sorgenti.

## Eseguire sul server Linux in sandbox

Trasferire `verified_tests.jsonl` nella radice del repository sul server, il report mapping per tracciabilita e gli script aggiornati di `CodePromptOpt/finetuning/`. Non occorre rifare code generation o training.

```bash
cd /home/agostinelli/esperimenti_tesi
bash CodePromptOpt/finetuning/downstream/client.sh build
bash CodePromptOpt/finetuning/downstream/evaluate_suite.sh \
  CodePromptOpt/finetuning/outputs/codegen/qwen25-coder-7b-h200-v1/validation \
  verified_tests.jsonl
```

L'immagine benchmark viene costruita automaticamente da `downstream/Test.Containerfile` e include NumPy 1.26.4, richiesto dai test HumanEval+. Viene verificato l'import prima di attribuire fallimenti ai modelli. Il codice non viene eseguito sul PC Windows o sull'host Linux: ogni caso passa per Podman senza rete/GPU, con filesystem read-only e limiti di risorse. Il timeout totale per task rimane 20 secondi e quello interno per caso APPS e 10 secondi; limiti e immagine devono essere identici per tutti i confronti.

I risultati saranno i file `*-results.jsonl` e `*.summary.json` accanto ai code JSONL. Non sono ancora presenti punteggi pass@1: l'esecuzione effettiva richiede il server Podman.
