# ELEMENTI

Workspace di analisi e previsione sul **Lotto italiano**, basato su dati storici Franknet.

## Contenuto

| Cartella | Descrizione |
|----------|-------------|
| [`franknet_lotto/`](franknet_lotto/) | Mirror Franknet, dataset tabellare, script di analisi e simulazione |

## Quick start

```bash
cd franknet_lotto
python scripts/parse_draws.py          # rigenera CSV/JSON da years/*.HTM (se serve)
python scripts/vertibile_rule_test.py  # test regole cross + Monte Carlo
python scripts/find_profit_zones.py    # ricerca zone di profitto (sim ADM corretta)
```

Dati principali: `franknet_lotto/data/draws_wide.csv` (~10.800 estrazioni, 11 ruote).

## Metodo cross (sintesi)

Dalla quintina del giorno **T** si costruisce un pool (~59 numeri) con ricette cross (differenze adiacenti, complementi a 90, vertibili, gemelli ±11). Il metodo `cross_opt_v1` è la variante migliore per hit-rate sul pool.

**Simulazione scommesse:** 1 € intero per sorte secca su singola ruota (quote ADM). Script di riferimento: `simulate_sorti_proper.py`, `find_profit_zones.py`.

**Risultato principale (2015–2026):** unico edge statisticamente stabile → **NAZIONALE + ambo** (top-2, +622 € lordi su 1878 estrazioni). Dettaglio in `franknet_lotto/data/analysis/profit_zones.json`.

## Per agenti AI (cloud)

Leggi [`AGENTS.md`](AGENTS.md) per contesto, convenzioni e script chiave.

## Requisiti

- Python 3.10+
- Dipendenze ML opzionali: `franknet_lotto/requirements_ml.txt`

## Fonte dati

[Archivio Franknet (Altervista)](https://www.franknet.altervista.org/lotto/page.php) — uso statistico/offline; verificare sempre le estrazioni ufficiali ADM.
