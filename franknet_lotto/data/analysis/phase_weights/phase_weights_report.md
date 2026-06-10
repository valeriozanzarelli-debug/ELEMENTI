# Pesi per periodo di estrazione

Seme = **primo numero** (pos1). Fase = giorno settimana × ciclo estrazione × decile seme.

## Ruote (top-8 test)

| ruota | schema | cross | phase | Δ | PLAY fasi |
|-------|--------|-------|-------|---|-----------|
| BARI | weekday | 212 | 215 | +3 | 0 |
| CAGLIARI | draw_mod9 | 226 | 231 | +5 | 1 |
| FIRENZE | anchor_dr | 199 | 206 | +7 | 0 |
| GENOVA | draw_mod9 | 215 | 223 | +8 | 0 |
| MILANO | draw_mod4 | 211 | 217 | +6 | 0 |
| NAPOLI | weekday | 221 | 223 | +2 | 0 |
| PALERMO | anchor_decile | 208 | 211 | +3 | 1 |
| ROMA | draw_mod7 | 250 | 259 | +9 | 0 |
| TORINO | draw_mod4 | 215 | 221 | +6 | 1 |
| VENEZIA | draw_mod7 | 226 | 232 | +6 | 0 |
| NAZIONALE | weekday | 234 | 241 | +7 | 1 |

## Fasi PLAY (estratto)

- **NAZIONALE** sab (wd_5) train=256 signal=PLAY test_hits=61
- **TORINO** ciclo4 pos 2 (d4_2) train=200 signal=PLAY test_hits=52
- **CAGLIARI** ciclo9 pos 7 (d9_7) train=87 signal=PLAY test_hits=23
- **PALERMO** seme decile 4 (ad_4) train=80 signal=PLAY test_hits=19

## Legenda weekday
lun=0 … dom=6 (Python). Lotto classico: mar/gio/sab ≈ wd 1,3,5.