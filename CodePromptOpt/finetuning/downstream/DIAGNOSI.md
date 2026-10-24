# Prima di nuovi training: diagnosticare i fallimenti

La validation attuale non mostra un beneficio SFT. Sono presenti due problemi distinti:

1. Tutti i task APPS falliscono anche con i prompt originali e statici. Prima verificare interfacce, eccezioni, timeout interni e correttezza algoritmica. Questo non viene risolto automaticamente aumentando l'optimizer.
2. Alcuni optimizer producono ripetizioni, perdono la specifica o raggiungono il limite di token. Prima di un nuovo training va controllato questo comportamento su un piccolo pilot, senza valutare ancora il test finale.

I risultati precedenti vanno conservati. La diagnostica seguente riusa il codice gia generato, non richiede vLLM o GPU e scrive in una directory separata. Non esegue niente sul PC Windows.

## Sul server Linux con Podman

Trasferire gli script aggiornati e `verified_tests.jsonl`. Poi:

```bash
cd /home/agostinelli/esperimenti_tesi
bash CodePromptOpt/finetuning/downstream/client.sh build

# Seleziona soltanto i fallimenti APPS di originale/statico,
# eliminando le copie della stessa richiesta fra i vari run.
CPO_IMAGE=localhost/cpo-codegen-client:1 bash CodePromptOpt/finetuning/podman.sh cpu \
  python -m CodePromptOpt.finetuning.downstream.diagnose_failures prepare

# Riesegue il piccolo sottoinsieme in sandbox con diagnostica limitata.
bash CodePromptOpt/finetuning/downstream/evaluate_suite.sh \
  CodePromptOpt/finetuning/outputs/diagnostics/apps-controls-v1 verified_tests.jsonl

# Aggrega gli errori in JSON leggibile.
CPO_IMAGE=localhost/cpo-codegen-client:1 bash CodePromptOpt/finetuning/podman.sh cpu \
  python -m CodePromptOpt.finetuning.downstream.diagnose_failures summarize \
  --root CodePromptOpt/finetuning/outputs/diagnostics/apps-controls-v1 \
  --output CodePromptOpt/finetuning/outputs/diagnostics/apps-controls-v1/report.json
```

Se la cartella diagnostica esiste gia, usare un nuovo `--output` in prepare e il medesimo percorso negli altri due comandi. Trasferire poi qui `report.json`. Non serve copiare/rifare l'intera batteria di test o scaricare altri pesi.

I nuovi record contengono `diagnostic`: categoria euristica, codice di uscita del processo, durata e ultimi 32 KiB per stream. La cattura drena gli stream senza conservarli illimitatamente. I nuovi job restano isolati nei container gia configurati. I timeout esterni possono non avere traceback; quelli APPS interni vengono distinti quando visibili. Non sono state cambiate soluzioni, test, asserzioni o risultati attesi. I controlli vengono eseguiti mediante un wrapper diagnostico, quindi conservare separatamente questa nuova esecuzione.

## Come decidere il passo successivo

- Nome funzione errato/mancante: stabilire un protocollo nuovo che specifichi il nome richiesto dal benchmark a **tutte** le varianti. Non rinominare silenziosamente le risposte gia raccolte. I nomi APPS si possono recuperare dallo schema `fn_name` delle fonti verificate, senza usare input/output dei test nel prompt.
- Assertion fallita: esaminare algoritmo e requisiti, senza allentare i test per far passare il modello.
- Timeout interno: distinguere algoritmo troppo lento da limiti infrastrutturali prima di cambiare il budget; qualunque nuovo limite deve valere per tutti.
- Import/runtime mancante: sistemare il runtime o il protocollo di soluzione completa, conservando separata la rivalutazione.

Per gli optimizer, il prossimo controllo utile e un'ablation con problema originale sempre fornito al code generator e optimizer usato solo per le istruzioni aggiuntive. Deve essere un esperimento nuovo, applicato a base/SFT e controlli, senza riscrivere questi risultati: distingue perdita della specifica da qualita della strategia suggerita.

Solo dopo il controllo del protocollo conviene provare un piccolo pilot di training (per esempio Qwen3-4B Best, seed 42), verificare il template di training/inferenza e la qualita dei prompt per epoca. Un LR piu basso o meno epoche sono ipotesi da verificare, non correzioni gia dimostrate. Non riaddestrare subito tutti i modelli o aumentare soltanto il limite di generazione: puo prolungare le ripetizioni senza risolverle.
