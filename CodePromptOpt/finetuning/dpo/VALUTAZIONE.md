# Generare il codice e confrontare DPO con SFT

Eseguire sul server dalla radice del repository, dopo aver trasferito i sorgenti
aggiornati. Non ci sono ancora prompt o risultati DPO nel workspace locale:
non e ancora possibile dire se DPO ha migliorato il Pass@1.

## 1. Prompt DPO

Se non sono gia presenti i quattro file `validation-*-prompts.jsonl` nella run:

```bash
bash CodePromptOpt/finetuning/dpo/podman.sh generate --seed 42 --split validation
bash CodePromptOpt/finetuning/dpo/podman.sh await-job
```

Se sono gia completi, saltare questi comandi. Una generazione parziale deve
essere archiviata prima di ripartire; non viene sovrascritta automaticamente.
Per SFT -> DPO passare lo stesso `--run-name` scelto nel training.

## 2. Code generator sulla GPU 3

Nessun training o vecchio server deve occupare la GPU 3. Se il vecchio coder e
ancora in esecuzione, fermarlo con il suo wrapper prima di questo avvio.

```bash
bash CodePromptOpt/finetuning/dpo/codegen.sh build
bash CodePromptOpt/finetuning/dpo/codegen.sh start
bash CodePromptOpt/finetuning/dpo/codegen.sh logs
```

Quando il server e pronto, Ctrl+C interrompe solo la visualizzazione dei log.

```bash
bash CodePromptOpt/finetuning/dpo/codegen.sh ready
bash CodePromptOpt/finetuning/dpo/codegen.sh run --runs qwen3_8b-dpo-s42
```

Il client genera codice con originale, statico, optimizer base e optimizer DPO.
Il client va lasciato terminare prima di fermare il coder; la cache delle
risposte consente di riprendere la generazione interrotta con lo stesso comando.

`start` legge il precedente `qwen3-coder-30b-h200-v1/server-config.json` e usa la
stessa revisione del coder, immagine vLLM locale, contesto e impostazioni server.
Se l'immagine originale non e disponibile si ferma, senza sostituirla con una
nuova versione. Il percorso e configurabile in `dpo/podman.env` tramite
`CPO_DPO_BASELINE_CONFIG`. L'immagine gia locale non viene riscaricata.

## 3. Sandbox e memoria

Solo dopo che il client ha completato la generazione:

```bash
bash CodePromptOpt/finetuning/dpo/codegen.sh stop
bash CodePromptOpt/finetuning/dpo/evaluate.sh sandbox
```

`stop` libera la VRAM; i pesi scaricati restano nella cache su disco.
La sandbox usa `verified_tests.jsonl`, non richiede GPU e non va eseguita
direttamente sul PC senza isolamento. Non cancellare gli output degli adapter.
Il batch sandbox richiede directory risultati/jobs nuove: non e un comando di
resume delle valutazioni gia avviate.

## 4. Tabella DPO vs SFT

```bash
bash CodePromptOpt/finetuning/dpo/evaluate.sh report --seeds 42
```

Richiede sul server anche i precedenti risultati SFT del coder 30B:
`outputs/codegen/qwen3-coder-30b-h200-v1/validation/qwen3_8b-{best,random,topk}-s42/`.
Per ogni condizione servono `*-code.jsonl`, `*-results.jsonl` e i relativi
`*.summary.json`. Se sono solo sul PC, copiarli sul server mantenendo le cartelle,
oppure riportare qui i risultati DPO ed eseguire lo script Python di confronto localmente.

File prodotti in `CodePromptOpt/reports/`:

- `dpo_vs_sft_qwen3_8b.md`: tabella leggibile, delta e intervalli bootstrap.
- `dpo_vs_sft_qwen3_8b.csv`: Pass@1 in frazioni 0..1, con valori per seed.
- `dpo_vs_sft_qwen3_8b.json`: protocollo e confronti completi.

La tabella confronta **Originale, Statico, Base, SFT Best, SFT Random, SFT Top-k,
DPO**. Un delta positivo DPO - SFT indica un miglioramento osservato.
Lo script rifiuta task incompleti, seed mancanti, revisioni o protocolli diversi.
I seed sono accoppiati: DPO 42 si confronta con SFT 42; non con la media SFT
42/43/44. Il bootstrap ricampiona task, dopo la media dei seed per task.

Per tutti i seed, dopo aver generato i prompt di 42, 43 e 44:

```bash
bash CodePromptOpt/finetuning/dpo/codegen.sh run --runs \
  qwen3_8b-dpo-s42 qwen3_8b-dpo-s43 qwen3_8b-dpo-s44
# Fermare il coder, poi valutare solo le directory nuove se 42 e gia valutato.
bash CodePromptOpt/finetuning/dpo/evaluate.sh report --seeds 42 43 44
```

Per nomi run personalizzati usare `report --run-pattern 'NOME-s{seed}'`.
Per confrontare SFT -> DPO con la precisa inizializzazione SFT usare inoltre
`finetuning.compare` sui risultati della cartella `-initial-sft`, come nella
guida principale. Con soli dieci task, variazioni del Pass@1 non sono una
dimostrazione definitiva: mantenere separato il test finale.
