# Annotazione con due modelli sull'H200

Il pilot usa gli stessi 100 candidati train per entrambi gli annotatori:
25 APPS/APE, 25 APPS/APO, 25 HumanEval+/APE e 25 HumanEval+/APO.
Il campione locale copre 65 task e include 62 candidati con flag di qualita.
E un campione diagnostico stratificato, non rappresentativo della frequenza
complessiva delle strategie. Score e flag servono al campionamento e non sono
mostrati ai modelli.

Annotatori, senza adapter:

- [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct);
- [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501).

Le famiglie diverse aiutano a esplorare differenze di annotazione; non rendono
indipendenti gli errori e nessuno dei due costituisce la verita di riferimento.
Lo script include per Mistral i flag di tokenizer/config/load indicati nella
scheda ufficiale. I due server vengono avviati in sequenza sulla stessa GPU.
BF16, contesto totale 16384, utilizzo memoria 0.80, una sequenza alla volta.
Sono impostazioni iniziali da verificare sul server, non misure di throughput.

## Comandi sul server

Aggiornare prima il repository sul server con questi nuovi file. Gli artefatti
locali non sono versionati: `prepare` li rigenera dal dataset canonico.
Dalla radice del repository, preferibilmente dentro `tmux`:

```bash
cd /home/agostinelli/esperimenti_tesi
nvidia-smi -i 3

# CPU client leggero, senza installare Python sull'host.
bash CodePromptOpt/strategies/h200_annotate.sh build
bash CodePromptOpt/strategies/h200_annotate.sh prepare

# Primo annotatore. Attendere che i log dichiarino il server pronto.
bash CodePromptOpt/strategies/h200_annotate.sh start qwen
bash CodePromptOpt/strategies/h200_annotate.sh logs
```

Ctrl+C chiude soltanto la visualizzazione dei log. Poi:

```bash
bash CodePromptOpt/strategies/h200_annotate.sh ready
bash CodePromptOpt/strategies/h200_annotate.sh annotate qwen
bash CodePromptOpt/strategies/h200_annotate.sh stop

# Secondo annotatore, sugli stessi identici input.
bash CodePromptOpt/strategies/h200_annotate.sh start mistral
bash CodePromptOpt/strategies/h200_annotate.sh logs
```

Attendere il server pronto, chiudere i log con Ctrl+C, poi:

```bash
bash CodePromptOpt/strategies/h200_annotate.sh ready
bash CodePromptOpt/strategies/h200_annotate.sh annotate mistral
bash CodePromptOpt/strategies/h200_annotate.sh stop

bash CodePromptOpt/strategies/h200_annotate.sh compare
```

Lo script usa GPU 3 e porta loopback 8001, separata dalla porta del coder.
`start` rifiuta l'avvio se trova processi compute sulla GPU; non arresta altri
job. Attendere che finiscano o fermarli con il loro wrapper appropriato prima
di proseguire. Per una GPU assegnata diversa:

```bash
export ANNOTATION_GPU=2
```

`ANNOTATION_PORT`, `ANNOTATION_ROOT`, `ANNOTATION_CACHE` e `VLLM_IMAGE` sono
personalizzabili. Non cambiare le impostazioni tra `start` e `annotate`.
L'immagine vLLM viene risolta al suo ID, il modello al commit Hugging Face.
Queste identita, i parametri e i flag Mistral sono salvati e inclusi nel
protocollo delle risposte. Dopo il primo avvio usare immagine/revisione
registrate per le riprese (`VLLM_IMAGE`, `ANNOTATION_REVISION`), oppure scegliere
un nuovo `ANNOTATION_ROOT` se cambia il protocollo. Il wrapper non scarica
pesi sul PC: sono nella cache Podman del server.

## Cosa viene annotato e confrontato

Ogni richiesta contiene problema originale e candidato APE/APO. Output:
tre categorie di strategia, due etichette di controllo, citazioni esatte e
`review_notes`, come descritto nel README. Gli annotatori non vedono l'output
dell'altro. Lo schema attuale non distingue ancora con un campo dedicato
"nessuna strategia" da "non classificabile": i casi dubbi vanno segnalati nelle
note e trattati nella revisione del pilot.

Il client salva le risposte una alla volta e riprende quelle completate se si
rilancia lo stesso comando. Un errore API, JSON non valido o output troncato
interrompe il run: conservare i log e diagnosticare prima di proseguire.
JSON sintatticamente valido ma etichette/evidenze non valide e conservato come
risposta grezza e segnalato dal confronto; non diventa una classificazione vuota.

Output sotto `CodePromptOpt/strategies/artifacts/annotation_pilot_v2/`:

| File | Contenuto |
| --- | --- |
| `manifest.json` | Campione, seed e hash sorgente |
| `records.jsonl` | Record del pilot per revisione e validazione |
| `requests.jsonl` | Stessi input per entrambi gli annotatori |
| `qwen/responses.jsonl` | Annotazioni e risposte API Qwen |
| `mistral/responses.jsonl` | Annotazioni e risposte API Mistral |
| `comparison/agreement.json` | Copertura valida, accordo esatto e Cohen kappa per etichetta |
| `comparison/human_review.jsonl` | Disaccordi, risposte mancanti/non valide e campi per revisione umana |

Il confronto controlla ID, versione, hash del riferimento, etichette ammesse
e citazioni letterali. Non confronta come disaccordo le sole differenze nelle
citazioni o nelle note se le etichette coincidono. I mancanti e gli invalidi
sono espliciti nel report; le metriche usano solo le coppie valide. Kappa nullo
indica dati mancanti o etichette costanti concordi, non un errore numerico.
Il report include anche prevalenze e accordo positivo per evidenziare etichette
rare per cui il solo accordo percentuale puo essere fuorviante.

Revisionare tutti i disaccordi e anche un campione degli accordi, includendo
casi senza etichette. Non accettare automaticamente il consenso come corretto
e non risolvere due voti discordi con una maggioranza inesistente. Il pilot
serve a chiarire la tassonomia, creare un riferimento umano e poi decidere
se annotare tutto il train con entrambi i modelli. Non produce risultati
scientifici sull'efficacia delle strategie.

Verifiche locali: test Python con risposte controllate e protocollo di campionamento.
Serving, disponibilita GPU e annotazioni reali richiedono l'esecuzione sull'H200.
