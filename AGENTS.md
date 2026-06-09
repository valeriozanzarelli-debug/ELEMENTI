# AGENTS.md — guida per agenti AI

## Obiettivo del progetto

Analizzare estrazioni Lotto (Franknet) e validare un **metodo rule-based** basato su operazioni cross sulla quintina precedente (diff, complemento 90, vertibile, gemelli).

## Struttura

```
ELEMENTI/
  franknet_lotto/
    data/
      draws_wide.csv      # dataset principale (date × 11 ruote × 5 numeri)
      meta.json
      analysis/           # output JSON/CSV delle analisi
    scripts/              # tutta la logica Python
    years/*.HTM           # mirror HTML sorgente
```

## Script importanti

| Script | Uso |
|--------|-----|
| `vertibile_rule_test.py` | Pool generators, `pool_cross_opt_v1`, Monte Carlo |
| `cross_method_report.py` | Report hit pool vs baseline |
| `cross_opt_train_test.py` | Train/test 500+500, grid search |
| `simulate_sorti_proper.py` | Simulazione CORRETTA ADM (1€ intero/sorte) |
| `find_profit_zones.py` | Scansione profitto per ruota/anno/sorte |
| `simulate_wheel_daily_report.py` | Report giornaliero TORINO/ROMA/BARI |

## Regole da rispettare

1. **Simulazione ambo:** 1 € **intero** sulla coppia secca (top-2). Non splittare 1€ su C(8,2) coppie — moltiplicatore 250× vale solo per puntata intera.
2. **Quote ADM singola ruota (1€):** estratto 11.23×, ambo 250×, terno 4500×, quaterna 120000×, cinquina 6M×.
3. **Metodo default:** `cross_opt_v1` + `score_cross_numbers` + `top_n_from_pool`.
4. **PowerShell:** usare `;` non `&&`. `PYTHONUNBUFFERED=1` per output live.
5. **Minimizzare scope:** non refactorare codice non richiesto.

## Stato analisi (2026-06)

- Pool `cross_opt_v1` batte p95 Monte Carlo su BARI/ROMA (finestre recenti).
- Profitto reale lungo periodo: solo **NAZIONALE ambo** (vedi `data/analysis/profit_zones.json`).
- TORINO 2025 ambo: +1042€ singolo anno ma bilancio 2015–26 negativo.
- Terno/quaterna/cinquina: hit troppo rari; profitti da singoli colpi = fortuna.

## Ruote

`BARI, CAGLIARI, FIRENZE, GENOVA, MILANO, NAPOLI, PALERMO, ROMA, TORINO, VENEZIA, NAZIONALE`

## Comandi tipici

```bash
cd franknet_lotto
python scripts/find_profit_zones.py
python scripts/simulate_sorti_proper.py --years 2025,2026
python scripts/nazionale_ambo_detail.py
```
