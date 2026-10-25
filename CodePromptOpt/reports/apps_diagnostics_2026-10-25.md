# Diagnostica APPS: controlli originale e statico

Fonte: `CodePromptOpt/finetuning/outputs/diagnostics/apps-controls-v1/report.json` e i dieci `diagnostic-results.jsonl`. Nessun codice generato e stato eseguito nuovamente sul PC; sono stati letti i traceback esportati dalla sandbox.

Sono dieci richieste uniche (due per ciascuno dei cinque task APPS), non dieci task indipendenti. Tutte continuano a fallire. Il checksum della suite coincide con quello della precedente valutazione.

| Causa rilevata | Richieste |
|---|---:|
| Asserzione fallita / output diverso dall'atteso | 5 |
| Timeout del sottoprocesso APPS dopo 10 secondi | 2 |
| Eccezione nel codice generato | 2 |
| Nome funzione/interfaccia non compatibile | 1 |

## Dettaglio per problema

| Task | Originale | Statico | Evidenza |
|---|---|---|---|
| `task_304f464ff2acd7b96b46f8a3` (differentiate) | ValueError | AssertionError | Il parser dell'originale tenta `int('x-6')`; lo statico restituisce un risultato differente dall'atteso. |
| `task_359b4894f04d41eaa4641e85` (almost_everywhere_zero) | Timeout 10 s | Timeout 10 s | Entrambe le soluzioni enumerano `range(1, n + 1)`: non scalano agli input grandi della suite. |
| `task_50f04de1119bcf6a77e863ee` (valid_mersenne) | OverflowError | AttributeError | L'originale esegue una demo casuale all'import e calcola una potenza troppo grande per la conversione float. Lo statico definisce `is_mersenne_prime`, mentre la suite richiede `valid_mersenne`. |
| `task_2af336a553499320616b5b60` (Toad Zitz) | Output mismatch | Output mismatch | Il programma termina ma l'output non coincide con quello della suite. |
| `task_db4b4fe60b637b446e605e53` (Vasya) | Output mismatch | Output mismatch | Il programma termina ma l'output non coincide con quello della suite. |

L'`AttributeError` menziona `Solution` perche il runner tenta prima la funzione richiesta a livello modulo e poi il metodo di una classe `Solution`. Il codice statico non fornisce nessuna delle due interfacce per `valid_mersenne`: non significa che la classe sia obbligatoria.

## Cosa possiamo concludere

- I traceback di queste dieci richieste documentano problemi nel codice/interfaccia o nel suo tempo di esecuzione. Non mostrano mancanza di NumPy, errori di Podman o test non trovati.
- Aumentare soltanto il timeout non e una soluzione motivata per la scansione lineare fino a n. Un limite diverso va comunque applicato a tutte le varianti e dichiarato come protocollo distinto.
- Il problema dell'interfaccia evidenzia un limite del protocollo: i task APPS importati possono non esplicitare nel testo il nome callable contenuto nello schema del benchmark. Il generatore deve ricevere questo contratto per tutte le varianti, senza usare i valori dei test.
- Non si devono rinominare risposte gia raccolte o correggere algoritmi per aggiornare retroattivamente pass@1. I punteggi precedenti rimangono validi per il protocollo che e stato eseguito, con questa limitazione documentata.
- La diagnostica riguarda solo i controlli APPS, non dimostra che ogni fallimento SFT abbia la stessa causa e non risolve le ripetizioni degli optimizer.

## Prossimo esperimento controllato

1. Preparare un protocollo v2 con problema originale sempre presente e contratto benchmark esplicito: nome funzione per call-based, stdin/stdout per script, niente demo eseguite all'import. Usare soltanto metadati di interfaccia, mai risultati/input della suite.
2. Ripetere prima originale/statico sui cinque APPS con lo stesso generatore e gli stessi test, in un nuovo experiment_id. Verificare il limite di interfaccia separatamente dalla qualita degli algoritmi.
3. Applicare il medesimo protocollo a optimizer base/SFT per distinguere perdita della specifica e qualita delle istruzioni aggiuntive.
4. Valutare eventualmente un generatore piu capace come asse sperimentale separato, identico per tutte le condizioni. Nessuna garanzia che cambiare modello faccia passare tutti i task.
5. Solo dopo questi controlli, studiare nuovi iperparametri SFT con un pilot piccolo sulla validation. Mantenere intatto il test finale.
