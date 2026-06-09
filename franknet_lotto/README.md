# Archivio Franknet — Estrazioni Lotto (mirror locale)

**Fonte:** [Archivio Estrazioni del Lotto — Franknet (Altervista)](https://www.franknet.altervista.org/lotto/page.php)  
**Mirror creato:** 2026-03-20  

## Struttura cartelle

| Percorso | Contenuto |
|----------|-----------|
| `page.php` | Pagina indice originale (link agli anni, script pubblicitari/cookie come sul sito). |
| `years/` | Un file per anno: `1871.HTM` … `2026.HTM` (156 file). |
| `assets/` | Risorse condivise (es. sfondo delle pagine annuali), se presenti. |
| `manifest.json` | Elenco anni scaricati con dimensione file in byte (utile da script). |

## Formato dei file `years/YYYY.HTM`

- HTML classico: titolo `Estrazioni Lotto YYYY`.
- I dati sono in blocchi `<pre>…</pre>` con intestazione fissa delle ruote:
  **BARI, CAGLIARI, FIRENZE, GENOVA, MILANO, NAPOLI, PALERMO, ROMA, TORINO, VENEZIA, NAZIONALE**.
- Ogni riga di estrazione: data (es. `02 gen`), poi 11 gruppi di 5 numeri (una ruota), ultima colonna numerica progressiva (estrazione nell’anno).
- L’anno è spezzato in più blocchi `<pre>` (tipicamente per trimestri o periodi).

Per parsing automatizzato conviene estrarre il testo dai `<pre>`, saltare le righe di header colorato, e splittare per spazi multipli.

## Note legali e uso

I dati sono pubblicati sul sito di terzi; questo mirror serve solo a lavoro offline / analisi. Verificare sempre le estrazioni ufficiali presso l’ente competente per uso non statistico.

## Dati tabellari (`data/`)

Dopo il mirror, generare CSV/JSON con:

```bash
python scripts/parse_draws.py
```

Si ottengono `data/draws_wide.csv`, `data/draws_long.csv`, `data/draws.json` e `data/meta.json` (vedi `data/README.md`).

Analisi **finestre temporali** (uniformità per decennio e ruota): `python scripts/longterm_distribution_windows.py` → `data/analysis/uniformity_by_decade_wheel.csv`.

**Una casella** (ruota+posizione), trasformazioni e n-grammi: `python scripts/single_cell_derived_series.py` → `data/analysis/single_cell/` (vedi `data/README.md`).

**Struttura nella serie** (Markov, AR(1), motivi, shuffle-null): `python scripts/discover_sequence_structure.py` → `…_structure_report.md` / `.json`.

**Mappa temporale radici digitali** (finestre, transizioni 9×9, walk-forward): `python scripts/digital_root_temporal_map.py` → `data/analysis/digital_root/<ruota>/`.

**Modelli più complessi** (HistGradientBoosting + Random Forest, feature multilag): `pip install -r requirements_ml.txt` poi `python scripts/boosted_dr_predictor.py` → `data/analysis/ml_models/`.

**Setaccio multi-metrica + nulli a permutazione** (autocorr, spettro, Higuchi, Hurst R/S, DFA, run test…): `python scripts/structure_sieve.py` → `data/analysis/structure_sieve/` (non è decrittazione; vedi docstring).

**Cicli copertura 1..90** (tutte le ruote) + stream deterministico (cifre + SHA-256 per ciclo): `python scripts/coupon_cycle_stream.py` → `data/analysis/coupon_cycles/`.

**Analisi cicli/cifre** (chi² vs uniforme i.i.d., autocorr, lunghezze): `python scripts/analyze_coupon_cycles.py` → `coupon_cycle_analysis.json`.

**Ordine prima comparsa + catena somme** (per ciclo 1..90): `python scripts/first_hit_chain_analysis.py` → `data/analysis/first_hit_chain/`.

**Null pesanti + FDR** (MC cifre vs uniforme i.i.d.; permutazione su transizioni decile→decile da `first_hit_steps`; Benjamini–Hochberg): `python scripts/heavy_null_engine.py --tier quick|standard|full` → `data/analysis/heavy_null/heavy_null_<tier>_report.json`.

**Super-audit** (orchestra parse, cicli, first-hit, setaccio, heavy null, single-cell, discover, longterm; opzionale digital map e ML): `python scripts/super_audit.py --tier quick|standard|full` → `data/analysis/MASTER_AUDIT_<tier>.json` e `.md`.

## Riproducibilità

Per aggiornare il mirror, riscaricare `page.php` e tutti i `YYYY.HTM` dalla stessa base URL `https://www.franknet.altervista.org/lotto/`, poi rieseguire `parse_draws.py`.
