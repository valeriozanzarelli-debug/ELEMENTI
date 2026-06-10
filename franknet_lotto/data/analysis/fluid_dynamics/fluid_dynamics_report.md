# Fluidodinamica estrazione — ordine pos1..pos5

## Idea

Nessuno guardava la quintina come **tubo di flusso**: il peso del numero dipende da *dove*
esce nell'ordine, non solo dal valore. Il metodo cross usa `adj_diffs` = |n_{k+1}-n_k|
(ordine cieco); qui v_k = n_{k+1}-n_k rivela direzione e accelerazione del flusso.

## Asimmetria gap (media signed velocity per ruota)

| gap | mean_signed |
|-----|-------------|
| pos1_to_pos2 | 0.2131 |
| pos2_to_pos3 | 0.3822 |
| pos3_to_pos4 | -0.4588 |
| pos4_to_pos5 | 0.6285 |

## Entropia posizione (media ratio su ruote)

| pos | entropy_ratio |
|-----|---------------|
| 1 | 0.990494 |
| 2 | 0.989935 |
| 3 | 0.989744 |
| 4 | 0.990032 |
| 5 | 0.989613 |

## Test 500 transizioni — pool pieno (confronto equo)

### Pool pieno (hit su 5 numeri)

| ruota | pesi | cross | fluid | Δ fluid |
|-------|------|-------|-------|---------|
| BARI | (1, 1, 1, 1, 1) | 1633 | 2056 | +423 |
| CAGLIARI | (1, 1, 1, 1, 1) | 1632 | 2088 | +456 |
| FIRENZE | (1, 1, 1, 1, 1) | 1631 | 2080 | +449 |
| GENOVA | (1, 1, 1, 1, 1) | 1631 | 2091 | +460 |
| MILANO | (1, 1, 1, 1, 1) | 1607 | 2100 | +493 |
| NAPOLI | (1, 1, 1, 1, 1) | 1614 | 2100 | +486 |
| PALERMO | (1, 1, 1, 1, 1) | 1644 | 2092 | +448 |
| ROMA | (1, 1, 1, 1, 1) | 1697 | 2141 | +444 |
| TORINO | (1, 1, 1, 1, 1) | 1619 | 2085 | +466 |
| VENEZIA | (1, 1, 1, 1, 1) | 1616 | 2063 | +447 |
| NAZIONALE | (1, 1, 1, 1, 1) | 1673 | 2104 | +431 |

### Top-8 scored (rilevante per ambo)

| ruota | cross top8 | cross_w top8 | fluid top8 | Δ fluid |
|-------|------------|--------------|------------|---------|
| BARI | 212 | 212 | 217 | +5 |
| CAGLIARI | 226 | 226 | 227 | +1 |
| FIRENZE | 199 | 199 | 203 | +4 |
| GENOVA | 215 | 215 | 213 | -2 |
| MILANO | 211 | 211 | 219 | +8 |
| NAPOLI | 221 | 221 | 219 | -2 |
| PALERMO | 208 | 208 | 210 | +2 |
| ROMA | 250 | 250 | 237 | -13 |
| TORINO | 215 | 215 | 216 | +1 |
| VENEZIA | 226 | 226 | 225 | -1 |
| NAZIONALE | 234 | 234 | 228 | -6 |

Report JSON: `fluid_dynamics_report.json`