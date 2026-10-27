# CodePromptOpt-Strategies / SGDP v1

Implementazione iniziale della direzione di ricerca: tutti i candidati storici,
annotazione differenziale, analisi esplorativa e prompt deterministici M0–M4.
Solo Python standard; nessun fine-tuning. Dalla radice del repository usare
`python` oppure `CodePromptOpt/.runtime/python/python.exe` su Windows.

## Riscontro sui dati reali

L'estrazione locale contiene **6.539 candidati APE/APO** (5.134 train, 597 validation,
808 test). Tutti hanno una baseline testuale; **nessuna baseline ha score dev**
nei log canonici: le 346 valutazioni baseline sono di test. Pertanto
`baseline_dev_score` e `improvement` restano null e `comparable` è false.
Non usare score test al posto di score dev. I flag di qualità riguardano 4.396
candidati: sono conservati nell'artefatto e nelle richieste di annotazione.

Il record contiene prompt candidato, selezione finale, fonte, ID delle valutazioni,
benchmark, difficoltà, split, modello e gruppo di confronto. Non chiamiamo ogni
candidato `best_prompt`: la selezione finale è un attributo osservato.
Annotazioni mancanti sono null, distinte dall'assenza annotata di strategie `[]`.

```powershell
python -m CodePromptOpt.strategies.pipeline dataset --output CodePromptOpt/strategies/artifacts/candidates.jsonl
python -m CodePromptOpt.strategies.pipeline annotation-requests --input CodePromptOpt/strategies/artifacts/candidates.jsonl --split train --output CodePromptOpt/strategies/artifacts/annotation_requests_train_original_v2.jsonl
```

I comandi sono stati eseguiti sui dati locali. Gli artefatti voluminosi sono
ignorati da Git e rigenerabili; il manifest registra hash degli input canonici.

## Recupero degli split dev storici

Il recupero dai due dataset pubblici fissati è stato eseguito su tutti i 90 task.
Il nuovo comando è:

```powershell
python -m CodePromptOpt.strategies.recover_dev --download
```

Omettere `--download` per usare le fonti già recuperate in
`strategies/artifacts/benchmark_sources/`. Il comando non importa `utils.py`,
non carica modelli e non esegue codice dei benchmark o dei candidati.

Risultato del 26 ottobre 2026: **334/346 gruppi**, relativi a **87/90 task**, hanno
una ricostruzione compatibile con i conteggi storici. Per APPS si ipotizza un cap
ufficiale pari a `n_dev + n_test` e un prefisso dev lungo `n_dev`; per HumanEval+
si richiede anche che la suite completa abbia esattamente `n_dev + n_test` casi.
Queste sono ipotesi esplicite, non verifiche dell'identità storica: nei log mancano
hash dei test, revisione originale del dataset e configurazione completa.
Il valore corrente di `DEV_SIZE` non viene usato per indovinare quello storico.

Restano esclusi 12 gruppi relativi a tre task HumanEval+: per HumanEval/1 e
HumanEval/28 il totale dei casi nella fonte differisce dai log; per HumanEval/50
l'entry point della fonte differisce dal metadato canonico. Le discrepanze sono
registrate, senza modificare sorgenti o metadati per farli coincidere.
`augmented_dev.json` è censito con hash ma non applicato: il solo indice locale
non dimostra che quel minibatch e quei casi fossero usati nei run storici.

Output in `strategies/artifacts/dev_recovery/`:

- `recovery_report.json`: copertura, discrepanze, ipotesi e hash degli input;
- `dev_suites.jsonl`: dev/test proposti per gruppo e hash della suite. I campi
  `dev_io_literal`/`test_io_literal` preservano anche eventuali tuple annidate;
- `baseline_dev_samples.jsonl`: codice storico disponibile per baseline,
  zero-shot e human-prompt, con un record per caso dev;
- `baseline_dev_tests.jsonl`: harness corrispondenti per la sandbox esistente.

Sono stati esportati **957 programmi**, suddivisi in **19.134 casi**. Gli ID dei
job sono distinti dagli ID dei task originali, conservati in `original_task_id`.
Il frazionamento permette di calcolare la frazione di casi superati per programma:
il `pass_at_1` globale prodotto dal collector esistente riguarda invece i job,
non i problemi. Non va presentato come accuratezza media per task.

Per eseguire i casi su Linux con Podman e l'immagine client già costruita:

```bash
CPO_IMAGE=localhost/cpo-codegen-client:1 bash CodePromptOpt/finetuning/podman.sh evaluate \
  CodePromptOpt/strategies/artifacts/dev_recovery/baseline_dev_samples.jsonl \
  CodePromptOpt/strategies/artifacts/dev_recovery/baseline_dev_tests.jsonl \
  CodePromptOpt/strategies/artifacts/dev_recovery/baseline_dev_results.jsonl \
  CodePromptOpt/strategies/artifacts/dev_recovery/jobs
```

L'esecuzione può richiedere tempo: ogni caso usa un container separato. I test
restano adattati al nuovo harness di sandbox; la gestione call-based supporta
anche `Solution`, mentre il runner storico chiamava direttamente la funzione.
Per confronti storici rigorosi serve validare anche questa equivalenza del runner.
Gli eventuali score ottenuti restano una **rivalutazione su suite ricostruita**,
senza sovrascrivere i punteggi canonici o dichiarare verificata l'identità storica.
Su questo PC non sono disponibili Bash e Podman: gli harness sono stati preparati
e controllati, ma non eseguiti. Nessuno score funzionale è stato generato.

```powershell
python -m unittest CodePromptOpt.strategies.test_recover_dev CodePromptOpt.strategies.test_strategies CodePromptOpt.finetuning.downstream.test_prepare_tests -v
```

## Annotazione LLM e revisione

Per il **pilot con due annotatori locali** (Qwen2.5-32B-Instruct e
Mistral-Small-24B-Instruct-2501), usare la procedura dedicata
[ANNOTATION_H200.md](ANNOTATION_H200.md). Include campionamento di 100 candidati,
avvio sequenziale, comandi di inferenza e report di accordo/disaccordo.

Le richieste confrontano **il problema originale (`original_problem`, input
zero-shot senza arricchimenti) con il candidato (`candidate_prompt`)**, senza
mostrare score, selezione finale, codice o test all'annotatore. La baseline
storica riscritta dall'optimizer non viene inviata all'annotatore: resta nel campo
`baseline_prompt` del dataset, insieme ai suoi score, come riferimento distinto
per le valutazioni quantitative. Il problema originale viene preso direttamente
da `tasks.jsonl`, quindi non serve che esista una generazione zero-shot nei log.

Si annotano istruzioni aggiunte o modificate rispetto al problema originale:
il semplice testo del problema copiato invariato non costituisce una strategia
aggiunta. Un richiamo esplicito aggiunto fuori dal testo copiato, come
"Check boundary cases", puo essere annotato anche se compare anche nella
baseline storica: quella baseline non e il riferimento di classificazione.
L'eventuale confronto con il prompt iniziale effettivo di APE/APO resta
un'analisi separata, quando quel prompt e documentato.

Il protocollo e versionato `original-problem-v2`; ogni richiesta ha un
`request_hash` che lega testo, istruzioni dell'annotatore e riferimento. Le
risposte del precedente confronto con `baseline_prompt` vengono rifiutate al
merge e devono essere rigenerate in nuovi file di output. I vecchi artefatti
rimangono disponibili e non vengono sovrascritti.

Tassonomia multilabel:
`constraint_explicitization`, `algorithmic_strategy_injection`,
`procedural_scaffolding`; controlli `semantic_change`, `unsupported_instruction`.
Ogni etichetta deve avere citazioni letterali dal candidato. L'esistenza di una
citazione verifica il riferimento testuale, non la correttezza dell'annotazione.
Serve una revisione umana stratificata per benchmark/optimizer/score, idealmente
con due annotatori e accordo per etichetta prima di fissare la tassonomia.

Con un endpoint già avviato compatibile Chat Completions e JSON mode:

```powershell
python -m CodePromptOpt.strategies.infer --input CodePromptOpt/strategies/artifacts/annotation_requests_train_original_v2.jsonl --output CodePromptOpt/strategies/artifacts/annotation_responses_train_original_v2.jsonl --model MODELLO_SERVITO --model-revision REVISIONE_EFFETTIVA
python -m CodePromptOpt.strategies.pipeline merge --input CodePromptOpt/strategies/artifacts/candidates.jsonl --responses CodePromptOpt/strategies/artifacts/annotation_responses_train_original_v2.jsonl --output CodePromptOpt/strategies/artifacts/annotated.jsonl
python -m CodePromptOpt.strategies.pipeline analyze --input CodePromptOpt/strategies/artifacts/annotated.jsonl --split train --output CodePromptOpt/strategies/artifacts/associations_train.json
```

`infer` registra risposte API integrali e protocollo, riprende le risposte già
salvate e rifiuta cambiamenti di input/modello/revisione. La revisione è dichiarata
dall'operatore, non verificata dal server. Autenticazione: `CODEGEN_API_KEY`, come
nel client downstream esistente. Nessuna chiamata è stata eseguita durante
l'implementazione. Un output valido è `{id, record_hash, response, metadata}`;
`response` contiene `strategies`, `strategy_evidence`, `review_notes`.
Ogni risposta deve inoltre conservare `annotation_version`,
`annotation_reference` e `request_hash` della richiesta; `infer` li copia
automaticamente. Le annotazioni accettate riportano questi riferimenti
nel record finale.

L'analisi usa score dev univoci, esclude flag e controlli, confronta presenza e
assenza **nello stesso gruppo**, separa benchmark/optimizer/modello, media prima
i gruppi per task e poi i task. La baseline comune si cancella nel contrasto tra
i due gruppi di candidati, anche se il suo score non è noto. Registra frequenze,
cooccorrenze e bootstrap per task (2.000 repliche, seed 42; intervallo nullo con
un solo task). Restano confondimento tra strategie e bias della ricerca:
queste non sono stime causali. Le annotazioni sono marcate non revisionate.
L'analisi esplorativa sullo split test è disabilitata.

## Selettore e builder

```powershell
python -m CodePromptOpt.strategies.pipeline selector-requests --split validation --output CodePromptOpt/strategies/artifacts/selector_requests_validation.jsonl
python -m CodePromptOpt.strategies.infer --input CodePromptOpt/strategies/artifacts/selector_requests_validation.jsonl --output CodePromptOpt/strategies/artifacts/selector_responses_validation.jsonl --model MODELLO_SERVITO --model-revision REVISIONE_EFFETTIVA
python -m CodePromptOpt.strategies.pipeline prompts --split validation --responses CodePromptOpt/strategies/artifacts/selector_responses_validation.jsonl --output CodePromptOpt/strategies/artifacts/prompts
```

La decisione contiene solo `strategies`, `evidence`, `confidence`, con citazioni
letterali dal problema per ogni categoria selezionata. L'hash lega la decisione al
problema originale. Testo libero, categorie sconosciute e citazioni inventate
vengono rifiutati. Il builder ammette solo istruzioni predefinite, aggiunte prima
del problema senza modificarne il testo Unicode, inclusi whitespace e newline.
L'encoding UTF-8 del suffisso resta identico a quello della stringa originale.
Non vengono ancora iniettati algoritmi specifici: le citazioni da sole non ne
provano l'applicabilità. I template v1 sono ipotesi da validare, non strategie
già dimostrate o derivate automaticamente dai log.

| Metodo/file | Definizione |
| --- | --- |
| original / M0 | Problema originale esatto |
| static / M1 | Istruzione generica fissa |
| all / M2 | Tutte e tre le categorie |
| dynamic / M3 | Categorie selezionate; scaffolding se il selettore sceglie zero |
| dynamic_abstain / M4 | Categorie selezionate; originale se zero o confidence sotto soglia |

La confidence non è calibrata: la soglia 0,5 è un parametro sperimentale da
fissare su validation, non una probabilità di correttezza. M3 e M4 condividono
la stessa decisione per isolare la politica di astensione; M3 ha un fallback
esplicito. Su validation si esportano anche C, A, P, CA, CP, AP: con original e
all completano il fattoriale di otto configurazioni. Senza `--responses` si
producono solo metodi statici/ablazioni; i metodi dinamici non sono simulati.

## Collegamento al generatore H200 esistente

I JSONL sono già nel formato del client downstream. Audit offline del pilot:

```powershell
python -m CodePromptOpt.finetuning.downstream.run --input-root CodePromptOpt/strategies/artifacts/prompts --output-root CodePromptOpt/strategies/artifacts/codegen --runs sgdp-v1 --variants original static all ablation_C ablation_A ablation_P ablation_CA ablation_CP ablation_AP --plan-only
```

Per M3/M4 aggiungere `dynamic dynamic_abstain` dopo aver prodotto le decisioni.
Per generare realmente rimuovere `--plan-only`, usare il config del server
effettivamente servito e assegnare un nuovo `experiment_id` SGDP. Riutilizzare
la procedura [H200 downstream](../finetuning/downstream/H200.md) per serving,
test verificati e sandbox: nessun codice generato viene eseguito dal modulo SGDP.

Prima del test finale fissare template, modello e revisione, decoding, soglia,
metodi e metriche. Rieseguire selector/prompts con `--split test` solo dopo questo
passaggio: le ablazioni non vengono esportate sul test. Il manifest dei prompt
registra configurazioni, task e hash; non sostituisce una preregistrazione.
I dieci task test già usati durante ricerche precedenti non diventano unseen:
per conclusioni confermative serve un insieme nuovo con mappatura benchmark
verificata. Il pilot downstream resta greedy pass@1; la cache esistente non
produce repliche indipendenti. Multiple generazioni, costi/token del selettore,
pass@k e nuova rivalutazione delle baseline dev richiedono un successivo
protocollo di campionamento e suite autentiche.

```powershell
python -m unittest CodePromptOpt.strategies.test_strategies CodePromptOpt.finetuning.downstream.test_run -v
```
