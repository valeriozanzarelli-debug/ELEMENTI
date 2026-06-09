# Master audit (quick)

- Inizio: 2026-03-20T01:31:03.201856+00:00
- Fine: 2026-03-20T01:32:22.015348+00:00
- Fasi OK: 9, fallite: 0
- Tempo totale fasi: 78.799 s

## Fasi

| Fase | Script | Esito | Secondi |
|------|--------|-------|---------|
| parse_draws | parse_draws.py | ok | 1.109 |
| coupon_cycle_stream | coupon_cycle_stream.py | ok | 0.581 |
| analyze_coupon_cycles | analyze_coupon_cycles.py | ok | 0.523 |
| first_hit_chain_analysis | first_hit_chain_analysis.py | ok | 0.666 |
| structure_sieve_milano_n | structure_sieve.py | ok | 33.101 |
| heavy_null_engine | heavy_null_engine.py | ok | 39.085 |
| single_cell_series | single_cell_derived_series.py | ok | 2.065 |
| discover_structure | discover_sequence_structure.py | ok | 1.033 |
| longterm_distribution | longterm_distribution_windows.py | ok | 0.636 |

JSON completo: `C:\Users\valer\Desktop\ELEMENTI\franknet_lotto\data\analysis\MASTER_AUDIT_quick.json`

Report aggregato: nessuna conclusione di prevedibilita senza validazione prospettica; i p-value sono quelli prodotti dagli script citati.