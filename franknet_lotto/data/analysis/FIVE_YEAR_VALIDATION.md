# Validazione definitiva — teoria cross, ultimi 5 anni

Finestra: **2021-03-17 → 2026-03-17**, tutte le 11 ruote (10164 transizioni totali).

## Test A — Pool vs null esatto (ipergeometrico, a parità di pool)

Per ogni regola: hit osservati vs attesi da un pool CASUALE della stessa dimensione. Z-score e p-value one-sided; `Bonf` = ruote significative dopo Bonferroni.

| Regola | Pool medio | Hit | Attesi | Margine | Z | p | Ruote>null | Bonf |
|---|---|---|---|---|---|---|---|---|
| control_random25 | 25 | 14250 | 14116.3 | +133.7 | 1.35 | 0.0877 | 8/11 | 0 |
| smart_hot30 | 25 | 14227 | 14116.3 | +110.7 | 1.12 | 0.1310 | 6/11 | 0 |
| vert_diff_only | 25.1 | 14265 | 14184.8 | +80.2 | 0.82 | 0.2056 | 7/11 | 0 |
| user_full_twin11 | 68.3 | 38645 | 38590.2 | +54.8 | 0.59 | 0.2775 | 6/11 | 0 |
| user_full | 68.6 | 38784 | 38729.6 | +54.4 | 0.59 | 0.2780 | 6/11 | 0 |
| hybrid | 69.8 | 39468 | 39418.2 | +49.8 | 0.55 | 0.2912 | 6/11 | 0 |
| comp_best_diff | 25.2 | 14290 | 14241.9 | +48.1 | 0.49 | 0.3130 | 6/11 | 0 |
| diff_chain | 33.3 | 18809 | 18780.0 | +29.0 | 0.27 | 0.3920 | 6/11 | 0 |
| scored_top20 | 20 | 11304 | 11293.7 | +10.3 | 0.11 | 0.4552 | 6/11 | 0 |
| intersection_2of3 | 31.7 | 17893 | 17891.6 | +1.4 | 0.01 | 0.4947 | 6/11 | 0 |
| cross_opt_v1 | 58.7 | 33119 | 33124.0 | -5.0 | -0.05 | 0.5192 | 5/11 | 0 |
| special_4590 | 62.2 | 35139 | 35148.7 | -9.7 | -0.10 | 0.5384 | 6/11 | 0 |
| scored_top19 | 19 | 10719 | 10728.3 | -9.3 | -0.10 | 0.5412 | 5/11 | 0 |
| cross_only | 55.5 | 31333 | 31345.4 | -12.4 | -0.12 | 0.5467 | 5/11 | 0 |
| cross_twin11 | 55.4 | 31240 | 31255.2 | -15.2 | -0.14 | 0.5571 | 4/11 | 0 |
| cross_opt_v2 | 56.9 | 32110 | 32125.1 | -15.1 | -0.14 | 0.5572 | 5/11 | 0 |
| scored_top25 | 25 | 14095 | 14116.3 | -21.3 | -0.22 | 0.5855 | 5/11 | 0 |
| twin_trigger | 53.4 | 30093 | 30130.0 | -37.0 | -0.34 | 0.6350 | 5/11 | 0 |
| scored_top35 | 35.0 | 19711 | 19759.4 | -48.4 | -0.45 | 0.6739 | 5/11 | 0 |
| smart_cold_ritardatari | 25 | 14071 | 14116.3 | -45.3 | -0.46 | 0.6769 | 6/11 | 0 |
| user_tight_twin11 | 53.2 | 29983 | 30035.4 | -52.4 | -0.49 | 0.6873 | 5/11 | 0 |
| user_tight | 53.6 | 30214 | 30273.7 | -59.7 | -0.56 | 0.7114 | 4/11 | 0 |
| positional | 41.3 | 23252 | 23325.1 | -73.1 | -0.67 | 0.7484 | 5/11 | 0 |
| minimal_comp_vert | 24.3 | 13616 | 13695.9 | -79.9 | -0.82 | 0.7936 | 5/11 | 0 |
| smart_markov_lift | 25 | 13793 | 14116.3 | -323.3 | -3.28 | 0.9995 | 1/11 | 0 |

Regole significative dopo correzione: **nessuna**.

Nota: la regola col punteggio migliore è il **controllo casuale** (pool di 25 numeri estratti a caso). Se un pool puramente casuale “batte” tutte le ricette cross, la classifica misura solo rumore.

## Test D — Il “ritrova 3-4 numeri su 5” è copertura

- Hit medi/estrazione del pool teoria (±1): **4.002**
- Attesi per la SOLA dimensione del pool: **4.004**

## Test B — Betting 1 € intero/sorte, 5 anni, 11 ruote

### cross_opt_v1

| Sorte | Giocate | Hit | Attesi (caso) | p-value | Netto € |
|---|---|---|---|---|---|
| estratto | 10164 | 541 | 564.67 | 0.8525 | -4088.57 |
| ambo | 10164 | 33 | 25.38 | 0.0826 | -1914.00 |
| terno | 10164 | 2 | 0.87 | 0.2148 | -1164.00 |
| quaterna | 10164 | 0 | 0.02 | 1.0 | -10164.00 |
| cinquina | 10164 | 0 | 0.0 | 1.0 | -10164.00 |

### smart_markov_lift

| Sorte | Giocate | Hit | Attesi (caso) | p-value | Netto € |
|---|---|---|---|---|---|
| estratto | 10164 | 583 | 564.67 | 0.2192 | -3616.91 |
| ambo | 10164 | 22 | 25.38 | 0.7757 | -4664.00 |
| terno | 10164 | 1 | 0.87 | 0.579 | -5664.00 |
| quaterna | 10164 | 0 | 0.02 | 1.0 | -10164.00 |
| cinquina | 10164 | 0 | 0.0 | 1.0 | -10164.00 |

## Test C — Selection bias (la “cella fortunata”)

- Best cell osservata: **CAGLIARI 2022 terno = +4343.00 €**
- Max nullo su 198 celle con giocate casuali (5000 rep): mediana +4291 €, p95 +4456 €
- Percentile del best osservato nel max nullo: **72.9%**
- Celle in profitto: osservate 30, attese per caso 25.86

## Verdetto

Nessuna regola (cross o smart) batte il null esatto dopo correzione per test multipli; il betting 5 anni e' in perdita per tutti i metodi; le celle in profitto sono compatibili con la selezione a posteriori del massimo su una griglia di celle casuali.
