# Prompt con correttezza osservata per problema

Conteggi di testi di prompt distinti esatti per task. Un prompt viene contato se almeno una sua soluzione salvata ha score 1 sullo split indicato. Si aggregano tutti i metodi e le configurazioni: il successo non implica correttezza universale o successo su ogni modello. Le colonne per modello sono nel CSV.

Dev e test sono separati. «Entrambi» richiede la stessa completion salvata nello stesso candidato/run con score 1 su dev e test. Un prompt senza valutazione test non viene contato nel test; ciò non implica che sia errato. «Ottimizzati test» considera solo APE/APO. I numeri non applicano i filtri di ammissibilità SFT/DPO.

APPS/local-N è l’indice locale dell’esperimento, non l’ID ufficiale APPS. Questo report include i task test interni per audit: non usarlo per selezionare etichette o regolare il modello.

| Problema | Difficoltà | Dev 100% | Test 100% | Entrambi | Ottimizzati test |
|---|---|---:|---:|---:|---:|
| APPS/local-0 | introductory | 42 | 49 | 42 | 43 |
| APPS/local-1 | introductory | 25 | 48 | 25 | 42 |
| APPS/local-2 | introductory | 41 | 48 | 41 | 42 |
| APPS/local-3 | introductory | 43 | 49 | 43 | 43 |
| APPS/local-4 | introductory | 44 | 50 | 44 | 44 |
| APPS/local-5 | introductory | 38 | 47 | 38 | 41 |
| APPS/local-6 | introductory | 42 | 48 | 42 | 42 |
| APPS/local-7 | introductory | 43 | 49 | 43 | 43 |
| APPS/local-8 | introductory | 42 | 49 | 42 | 43 |
| APPS/local-9 | introductory | 17 | 86 | 17 | 80 |
| APPS/local-10 | introductory | 26 | 55 | 26 | 49 |
| APPS/local-11 | introductory | 43 | 49 | 43 | 43 |
| APPS/local-12 | introductory | 42 | 46 | 42 | 42 |
| APPS/local-13 | introductory | 35 | 47 | 35 | 41 |
| APPS/local-14 | introductory | 43 | 49 | 43 | 43 |
| APPS/local-15 | interview | 38 | 44 | 38 | 39 |
| APPS/local-16 | interview | 0 | 85 | 0 | 81 |
| APPS/local-17 | interview | 37 | 43 | 37 | 37 |
| APPS/local-18 | interview | 44 | 50 | 44 | 44 |
| APPS/local-19 | interview | 44 | 50 | 44 | 44 |
| APPS/local-20 | interview | 44 | 50 | 44 | 44 |
| APPS/local-21 | interview | 26 | 32 | 26 | 28 |
| APPS/local-22 | interview | 23 | 28 | 23 | 23 |
| APPS/local-23 | interview | 44 | 50 | 44 | 44 |
| APPS/local-24 | interview | 6 | 78 | 6 | 74 |
| APPS/local-25 | interview | 12 | 35 | 12 | 31 |
| APPS/local-26 | interview | 31 | 41 | 31 | 35 |
| APPS/local-27 | interview | 24 | 30 | 24 | 27 |
| APPS/local-28 | interview | 6 | 9 | 6 | 9 |
| APPS/local-29 | interview | 39 | 45 | 39 | 39 |
| APPS/local-30 | competition | 18 | 25 | 18 | 23 |
| APPS/local-31 | competition | 11 | 16 | 11 | 14 |
| APPS/local-32 | competition | 22 | 29 | 22 | 25 |
| APPS/local-33 | competition | 3 | 15 | 3 | 15 |
| APPS/local-34 | competition | 16 | 28 | 16 | 25 |
| APPS/local-35 | competition | 17 | 20 | 17 | 17 |
| APPS/local-36 | competition | 1 | 1 | 1 | 1 |
| APPS/local-37 | competition | 0 | 0 | 0 | 0 |
| APPS/local-38 | competition | 2 | 13 | 2 | 12 |
| APPS/local-39 | competition | 0 | 20 | 0 | 20 |
| APPS/local-40 | competition | 18 | 26 | 18 | 22 |
| APPS/local-41 | competition | 12 | 15 | 12 | 12 |
| APPS/local-42 | competition | 25 | 40 | 25 | 36 |
| APPS/local-43 | competition | 3 | 21 | 3 | 19 |
| APPS/local-44 | competition | 22 | 27 | 22 | 23 |
| HumanEval/30 |  | 44 | 50 | 44 | 44 |
| HumanEval/6 |  | 47 | 52 | 47 | 47 |
| HumanEval/72 |  | 44 | 50 | 44 | 44 |
| HumanEval/64 |  | 41 | 46 | 41 | 41 |
| HumanEval/59 |  | 44 | 50 | 44 | 44 |
| HumanEval/37 |  | 46 | 52 | 46 | 46 |
| HumanEval/28 |  | 43 | 49 | 43 | 43 |
| HumanEval/145 |  | 0 | 0 | 0 | 0 |
| HumanEval/24 |  | 50 | 56 | 50 | 50 |
| HumanEval/112 |  | 0 | 0 | 0 | 0 |
| HumanEval/8 |  | 0 | 0 | 0 | 0 |
| HumanEval/7 |  | 44 | 50 | 44 | 44 |
| HumanEval/25 |  | 42 | 48 | 42 | 42 |
| HumanEval/57 |  | 43 | 49 | 43 | 43 |
| HumanEval/61 |  | 44 | 49 | 43 | 43 |
| HumanEval/134 |  | 36 | 39 | 33 | 33 |
| HumanEval/162 |  | 44 | 50 | 44 | 44 |
| HumanEval/52 |  | 44 | 50 | 44 | 44 |
| HumanEval/156 |  | 43 | 49 | 43 | 43 |
| HumanEval/111 |  | 44 | 50 | 44 | 44 |
| HumanEval/58 |  | 43 | 49 | 43 | 43 |
| HumanEval/118 |  | 44 | 50 | 44 | 44 |
| HumanEval/73 |  | 44 | 50 | 44 | 44 |
| HumanEval/1 |  | 41 | 47 | 41 | 41 |
| HumanEval/42 |  | 44 | 50 | 44 | 44 |
| HumanEval/154 |  | 48 | 41 | 35 | 35 |
| HumanEval/90 |  | 44 | 50 | 44 | 44 |
| HumanEval/141 |  | 44 | 46 | 41 | 41 |
| HumanEval/41 |  | 44 | 50 | 44 | 44 |
| HumanEval/150 |  | 43 | 49 | 43 | 43 |
| HumanEval/89 |  | 43 | 49 | 43 | 43 |
| HumanEval/13 |  | 44 | 50 | 44 | 44 |
| HumanEval/11 |  | 43 | 49 | 43 | 43 |
| HumanEval/50 |  | 43 | 45 | 39 | 39 |
| HumanEval/12 |  | 43 | 49 | 43 | 43 |
| HumanEval/47 |  | 44 | 50 | 44 | 44 |
| HumanEval/137 |  | 42 | 46 | 40 | 40 |
| HumanEval/46 |  | 44 | 50 | 44 | 44 |
| HumanEval/79 |  | 40 | 46 | 40 | 40 |
| HumanEval/35 |  | 50 | 55 | 50 | 50 |
| HumanEval/107 |  | 0 | 0 | 0 | 0 |
| HumanEval/5 |  | 43 | 49 | 43 | 43 |
| HumanEval/96 |  | 41 | 47 | 41 | 41 |
| HumanEval/60 |  | 44 | 50 | 44 | 44 |
| HumanEval/70 |  | 44 | 50 | 44 | 44 |
