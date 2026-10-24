# Valutazione funzionale sulla validation

27 run, 108 file risultati e riepiloghi, 1080 righe; 10 task indipendenti e 3 seed di training.
Code generator: `Qwen/Qwen2.5-Coder-7B-Instruct`. 249 richieste uniche; i controlli duplicati non sono repliche indipendenti.

## Controlli di integrita

- Tutti i file coprono lo stesso insieme di dieci task; i riepiloghi coincidono con i record.
- Modello, decoding, revisione, immagine vLLM, suite, immagine sandbox e timeout sono identici.
- Verificati 1080 check.py esportati contro gli harness locali.
- Il checksum dei test differisce fra Windows e Linux solo per CRLF/LF; quello normalizzato LF coincide.
- Timeout esterni registrati: 0. Un timeout interno APPS puo apparire come exit 1.
- Disaccordi sulle richieste riutilizzate: 0.

## Pass@1 medio sui tre seed di training

Valori percentuali. Ogni seed ha dieci task: la media non rappresenta trenta problemi distinti.

| Optimizer | Supervisione | Originale | Statico | Base | SFT | Delta SFT-base (pp) | SFT per seed 42/43/44 |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3_4b | best | 50.0 | 50.0 | 40.0 | 16.7 | -23.3 | 20/20/10 |
| qwen3_4b | random | 50.0 | 50.0 | 40.0 | 23.3 | -16.7 | 20/40/10 |
| qwen3_4b | topk | 50.0 | 50.0 | 40.0 | 30.0 | -10.0 | 20/50/20 |
| qwen3_8b | best | 50.0 | 50.0 | 20.0 | 20.0 | +0.0 | 20/20/20 |
| qwen3_8b | random | 50.0 | 50.0 | 20.0 | 16.7 | -3.3 | 10/30/10 |
| qwen3_8b | topk | 50.0 | 50.0 | 20.0 | 20.0 | -0.0 | 30/30/0 |
| qwen3_14b | best | 50.0 | 50.0 | 50.0 | 40.0 | -10.0 | 50/40/30 |
| qwen3_14b | random | 50.0 | 50.0 | 50.0 | 36.7 | -13.3 | 40/40/30 |
| qwen3_14b | topk | 50.0 | 50.0 | 50.0 | 40.0 | -10.0 | 40/50/30 |

## Confronto appaiato SFT-base

Intervalli bootstrap descrittivi (10000 ricampionamenti di task), dopo media dei seed dentro ogni task. Non si trattano i seed o le copie dei controlli come nuovi task.

| Optimizer | Supervisione | Delta (pp) | Intervallo 95% (pp) |
|---|---|---:|---|
| qwen3_4b | best | -23.3 | [-60.0, +10.0] |
| qwen3_4b | random | -16.7 | [-40.0, +3.3] |
| qwen3_4b | topk | -10.0 | [-33.3, +13.3] |
| qwen3_8b | best | +0.0 | [-10.0, +10.0] |
| qwen3_8b | random | -3.3 | [-26.7, +23.3] |
| qwen3_8b | topk | +0.0 | [-13.3, +16.7] |
| qwen3_14b | best | -10.0 | [-23.3, +0.0] |
| qwen3_14b | random | -13.3 | [-30.0, +0.0] |
| qwen3_14b | topk | -10.0 | [-26.7, +0.0] |

## Risultati per task

I numeratori sotto contano righe dei run: gli original/static sono ripetuti e non indipendenti.

| Task | Benchmark | Funzione/ID | Originale | Statico | Base | SFT |
|---|---|---|---:|---:|---:|---:|
| task_14ce6886318b93fabf08435a | HumanEval+ | count_up_to | 27/27 | 27/27 | 18/27 | 4/27 |
| task_2af336a553499320616b5b60 | APPS | task_2af336a553499320616b5b60 | 0/27 | 0/27 | 0/27 | 0/27 |
| task_2dd0255221de75accbb5e170 | HumanEval+ | compare_one | 27/27 | 27/27 | 18/27 | 10/27 |
| task_304f464ff2acd7b96b46f8a3 | APPS | task_304f464ff2acd7b96b46f8a3 | 0/27 | 0/27 | 0/27 | 0/27 |
| task_359b4894f04d41eaa4641e85 | APPS | task_359b4894f04d41eaa4641e85 | 0/27 | 0/27 | 0/27 | 0/27 |
| task_50f04de1119bcf6a77e863ee | APPS | task_50f04de1119bcf6a77e863ee | 0/27 | 0/27 | 0/27 | 0/27 |
| task_676f0878d4d712b47b342156 | HumanEval+ | intersperse | 27/27 | 27/27 | 18/27 | 20/27 |
| task_8ac553d23620fb6d969e2d79 | HumanEval+ | get_positive | 27/27 | 27/27 | 18/27 | 22/27 |
| task_c4b997fd5d2140e5f1841eba | HumanEval+ | string_to_md5 | 27/27 | 27/27 | 27/27 | 17/27 |
| task_db4b4fe60b637b446e605e53 | APPS | task_db4b4fe60b637b446e605e53 | 0/27 | 0/27 | 0/27 | 0/27 |

## Interpretazione e limiti

- Nessuna configurazione SFT migliora il rispettivo optimizer base in questa validation. Tutte restano sotto originale/statico (50%). Non emerge evidenza del beneficio cercato con il protocollo attuale.
- Originale/statico superano tutti e cinque i task HumanEval+ ma nessuno dei cinque APPS. Anche ogni altra variante fallisce tutti gli APPS: analizzare questi fallimenti prima di estendere le conclusioni.
- Gli harness sono coerenti con i file preparati e i test sintetici del runner, ma nei risultati non e registrato stderr: exit 1 non distingue algoritmo errato, interfaccia, eccezioni o timeout del sottoprocesso. Non si puo attribuire ogni fallimento a uno specifico motivo da questi record.
- Alcune soluzioni APPS originali hanno problemi visibili nel testo, come nomi funzione diversi dalla specifica o algoritmi che non gestiscono tutti i casi; questa ispezione non sostituisce una diagnosi runtime.
- Le ripetizioni/troncamenti dei prompt SFT rilevati nel precedente audit sono un limite plausibile, non una spiegazione causale dimostrata di ogni errore.
- Sono dieci task di validation, non il test finale. Non usare questi numeri come stima definitiva di generalizzazione o evidenza universale contro SFT. Conservare il test per un protocollo fissato.

CSV: valori per ogni seed. JSON: metadata, controlli, confronti appaiati per tutte le baseline e dettaglio task.
