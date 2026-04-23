# Tesi — Metaprompting per code generation

Confronto di tre strategie di prompt engineering (baseline single-shot, APE, APO) applicate problema per problema su APPS e HumanEval+, con target e optimizer serviti da Ollama in locale.

Ogni problema è un'istanza di ottimizzazione a sé: l'optimizer enricchisce il prompt, il target LLM genera codice, il codice viene valutato in un sandbox subprocess. Per APPS c'è una fase preliminare di data augmentation: la reference solution del dataset viene usata come oracolo per validare test-case sintetici generati da un LLM, in modo da avere un dev split più ricco dei pochi casi ufficiali.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ollama deve essere attivo e avere i modelli che userai:

```bash
ollama pull qwen2.5-coder:14b
ollama pull llama3.1:8b
# (solo se usi gpt-oss come augmenter)
ollama pull gpt-oss:20b
```

Le variabili vanno in `.env` nella root del progetto — vedi la tabella sotto.

## Esecuzione

Due step. L'augmentation è offline e una tantum, il run è quello di `main.py`.

### 1. Data augmentation APPS

```bash
python augment_apps.py
```

Per ogni problema del minibatch: trova la prima reference solution che passa i test ufficiali al 100%, poi fa **otto round** di generazione con focus diverso (typical, boundary, degenerate, stress, ordering, duplicates, adversarial, diverse_random). Gli input generati dall'LLM vengono validati eseguendo la reference; quelli che crashano o scadono vengono scartati. Dedupe cross-round. Target tipico: 150-200 test kept per problema. Output: `augmented_dev.json`.

**Attenzione**: il file è fingerprint-locked sul minibatch corrente. Se cambi `N_PER_DIFFICULTY` dopo aver augmentato, scatta un warning su stderr e lo split cade sul fallback classico (cioè, di fatto, perdi l'augmentation). Fissa il valore **prima** di lanciare augment.

### 2. Run sperimentale

```bash
python main.py
```

Per ogni problema: zero-shot (controllo), human-prompt (template fisso scritto a mano), baseline (optimizer single-shot), APE, APO. Scrittura incrementale del JSON di risultati dopo ogni problema, quindi il run è sicuro da interrompere. Per riprendere un run parziale:

```bash
python resume_missing.py result/<file_parziale>.json
```

## Config principale (`.env`)

| Var | Cosa fa | Valore tipico |
|---|---|---|
| `MODEL_TARGET` | LLM che scrive codice | `qwen2.5-coder:14b` |
| `MODEL_OPTIMIZER` | LLM che fa prompt engineering e critiche | `llama3.1:8b` |
| `MODEL_AUGMENTER` | LLM per generare test augmented (fallback: OPTIMIZER) | `gpt-oss:20b` |
| `DATASET` | `apps` o `humaneval` | `humaneval` |
| `N_PER_DIFFICULTY` | problemi APPS per difficoltà | 15 |
| `N_HUMANEVAL` | problemi HumanEval+ | 15-50 |
| `MAX_TEST_CASES` | cap test ufficiali per problema APPS | 10 |
| `N_CANDIDATES` | target TOTALE di test generati per problema (augment) | 200 |
| `N_MIN_KEPT` | sotto questo threshold un problema è `too_few` | 30 |
| `AUG_TEMPERATURE` | temperature dell'augmenter (alta → più diversità) | 0.9 |
| `DEV_SIZE` | test case usati come dev da APE/APO | 30 |
| `TEST_AUG_CAP` | cap sul surplus augmented che va al test split | 50 |
| `N_VARIANTS` | proposals iniziali di APE | 10 |
| `APE_N_ITERS`, `APE_N_KEEP` | refinement Monte Carlo di APE | 2, 2 |
| `APO_NUM_GRADIENTS` (m) | critiche per beam member | 4 |
| `APO_NUM_EDITS` (q) | edits per gradient | 1 |
| `APO_NUM_PARAPHRASES` (r) | paraphrases per edit | 1 |
| `APO_BEAM_WIDTH` | larghezza beam APO | 3 |
| `APO_MAX_ITERS` | iterazioni APO | 2 |
| `NUM_CTX` | context window di Ollama | 8192 |
| `EXEC_TIMEOUT` | cap esecuzione per test case (s) | 10 |

Con `DEV_SIZE=30`, `TEST_AUG_CAP=50`, `N_VARIANTS=10` e `APO_BEAM_WIDTH=3` il final pool di APE è 12 candidati, quello di APO è 24 — quindi `pass@{1,3,5,10}` sempre calcolabili.

## Struttura

```
.
├── main.py                     entry point: loop problemi × pipeline
├── utils.py                    dataset loaders, chain builders, sandbox eval,
│                               dev/test split, pass@k
├── augment_apps.py             generator offline multi-round, reference oracle
├── algorithms/
│   ├── baseline.py             optimizer single-shot
│   ├── ape.py                  APE (Zhou et al. 2022)
│   └── apo.py                  APO / ProTeGi (Pryzant et al. 2023)
├── resume_missing.py           helper per ripartire da run parziali
├── notebooks/
│   └── analisi_esp.ipynb       analisi post-hoc, plot in outputs/
├── result/                     JSON dei run (gitignored)
└── outputs/                    figure prodotte dal notebook (gitignored)
```
