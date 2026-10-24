# Audit dei prompt generati

Input: `CodePromptOpt\finetuning\outputs\runs`. Split: `validation`. Modelli: qwen3_4b, qwen3_8b, qwen3_14b.

Phi-4 escluso intenzionalmente: non addestrato. Nessun file di input modificato.

Copertura: 27 run, 108 file, 1080 righe; 0 problemi di copertura file; 0 cartelle run mancanti.

| Modello | Variante | Righe | Problema esatto | Problema normalizzato | Troncati | Ripetizioni* | Firma assente** |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3_14b | base | 90 | 0 | 9 | 0 | 0 | 0/45 |
| qwen3_14b | original | 90 | 90 | 90 | 0 | 0 | 0/45 |
| qwen3_14b | sft | 90 | 12 | 21 | 32 | 31 | 5/45 |
| qwen3_14b | static | 90 | 90 | 90 | 0 | 0 | 0/45 |
| qwen3_4b | base | 90 | 0 | 0 | 0 | 0 | 0/45 |
| qwen3_4b | original | 90 | 90 | 90 | 0 | 0 | 0/45 |
| qwen3_4b | sft | 90 | 9 | 10 | 24 | 24 | 21/45 |
| qwen3_4b | static | 90 | 90 | 90 | 0 | 0 | 0/45 |
| qwen3_8b | base | 90 | 0 | 0 | 0 | 0 | 9/45 |
| qwen3_8b | original | 90 | 90 | 90 | 0 | 0 | 0/45 |
| qwen3_8b | sft | 90 | 1 | 2 | 22 | 21 | 28/45 |
| qwen3_8b | static | 90 | 90 | 90 | 0 | 0 | 0/45 |

*Euristica: una sequenza di 8 parole compare almeno 8 volte; non e una prova di degenerazione.
**Si controlla il nome della funzione solo quando il dataset lo documenta. La presenza del nome non prova che interfaccia e semantica siano corrette.

Normalizzazione: collasso degli spazi e dei newline, senza riscrivere parole o simboli. Un problema non preservato puo essere riformulato correttamente: non equivale automaticamente a errore funzionale.

## Dettaglio SFT Qwen3-14B per run

| Run | Righe | Problema normalizzato | Troncati | Ripetizioni* |
|---|---:|---:|---:|---:|
| qwen3_14b-best-s42 | 10 | 2 | 1 | 1 |
| qwen3_14b-best-s43 | 10 | 3 | 1 | 1 |
| qwen3_14b-best-s44 | 10 | 2 | 3 | 3 |
| qwen3_14b-random-s42 | 10 | 5 | 5 | 5 |
| qwen3_14b-random-s43 | 10 | 1 | 7 | 6 |
| qwen3_14b-random-s44 | 10 | 2 | 4 | 4 |
| qwen3_14b-topk-s42 | 10 | 1 | 5 | 5 |
| qwen3_14b-topk-s43 | 10 | 3 | 3 | 3 |
| qwen3_14b-topk-s44 | 10 | 2 | 3 | 3 |

## Training

| Run | Primo eval loss | Miglior eval loss | Step finale |
|---|---:|---:|---:|
| qwen3_14b-best-s42 | 0.8242440223693848 | 0.6684814691543579 | 40 |
| qwen3_14b-best-s43 | 0.818213939666748 | 0.6660728454589844 | 40 |
| qwen3_14b-best-s44 | 0.8198819160461426 | 0.6734473705291748 | 40 |
| qwen3_14b-random-s42 | 0.8162193298339844 | 0.6702908277511597 | 40 |
| qwen3_14b-random-s43 | 0.815648078918457 | 0.6680351495742798 | 40 |
| qwen3_14b-random-s44 | 0.8208131194114685 | 0.6765407919883728 | 40 |
| qwen3_14b-topk-s42 | 0.8200411796569824 | 0.6693093180656433 | 40 |
| qwen3_14b-topk-s43 | 0.8184615969657898 | 0.665860116481781 | 40 |
| qwen3_14b-topk-s44 | 0.8185880780220032 | 0.6719850897789001 | 40 |

Questa analisi non esegue modelli o codice e non misura pass@1. Loss bassa, conservazione del testo e assenza di warning non dimostrano un miglioramento funzionale.
CSV: una riga per prompt. JSON: copertura, contatori, artefatti training e checksum dei file esaminati.
