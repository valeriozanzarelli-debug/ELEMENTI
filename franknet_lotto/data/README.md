# Dati estrazioni (formato analisi)

Generati con `python scripts/parse_draws.py` dalla cartella `franknet_lotto`.

| File | Uso tipico |
|------|------------|
| **`draws_wide.csv`** | Una riga per estrazione; colonne `BARI_1`…`NAZIONALE_5`, celle vuote se ruota assente o `--` nel sorgente. |
| **`draws_long.csv`** | Forma normale: una riga per ogni ruota con numeri; ruote interamente vuote (es. NAZIONALE prima dell’introduzione) omesse. |
| **`draws.json`** | Stesso contenuto del wide, con oggetto `wheels` solo per ruote con almeno un numero. |
| **`meta.json`** | Conteggi e ordine delle ruote. |

Colonne chiave: `year`, `date` (ISO `YYYY-MM-DD`), `draw_index_year` (progressivo nell’anno nel file sorgente), `num_wheels` (10 o 11 secondo il formato dell’anno).

## Previsione e “certezza”

Il gioco del lotto ufficiale è progettato come **casuale** (estrazione controllata, combinazioni equiprobili a parità di regole). **Non esiste un algoritmo che garantisca previsioni corrette delle uscite future**: qualsiasi regola complessa applicata allo storico può al massimo:

- riassumere **frequenze** e **correlazioni empiriche** (spesso compatibili con il caso);
- costruire **modelli probabilistici** che assegnano **probabilità**, non certezze.

Se trovate schemi nei dati, vanno verificati con test statistici e tenuto conto del **data mining su molte ipotesi** (rischio di “pattern” casuali). Per decisioni economiche o di gioco, fare riferimento solo a fonti ufficiali e al gioco responsabile.

## Stato probabilistico nel tempo (analisi esplorativa)

Obiettivo sensato: vedere se la **distribuzione empirica** dei numeri (per ruota) resta **vicina all’uniforme** quando si confrontano **finestre lungo l’asse temporale** (es. decenni). Se in qualche periodo le frequenze si discostano più che altrove, può essere:

- **fluttuazione campionaria** (normale su molte finestre e molte ruote);
- **cambiamenti di regole / formato** (ruote assenti, `--`, introduzione della Nazionale, ecc.);
- **problemi di qualità o omogeneità** dell’archivio storico.

Non implica che il futuro sia prevedibile: indica solo **come appariva la distribuzione osservata** in quella finestra.

Dopo `parse_draws.py`, dalla cartella `franknet_lotto`:

```bash
python scripts/longterm_distribution_windows.py
```

Si ottiene `data/analysis/uniformity_by_decade_wheel.csv` con, per ogni **decennio × ruota** (con abbastanza estrazioni): conteggi, **χ² rispetto all’uniforme** su 90 caselle, **entropia** di Shannon (in bit) e rapporto alla massima `log2(90)`. Utile per grafici (es. χ² o entropia vs tempo) e per confronti tra ruote.

## Una “casella” nel tempo (differenze, cifre, complemento)

Per studiare **ripetizioni** o **dipendenze scalate nel tempo** su **una sola ruota e una sola posizione** (es. primo numero di MILANO), dalla cartella `franknet_lotto`:

```bash
python scripts/single_cell_derived_series.py --wheel MILANO --position 1 --from-year 1900 --lag 5
```

Parametri utili: `--wheel`, `--position` (1…5 = `n1`…`n5`), `--from-year`, `--lag` (ritardo in estrazioni per colonne `*_lagN`, es. confrontare con **5 uscite prima**), `--max-autocorr-lag`.

**Output** in `data/analysis/single_cell/`:

| File | Contenuto |
|------|-----------|
| `*_series.csv` | Serie temporale ordinata: `n`, `diff_signed` / `diff_abs` (rispetto all’estrazione precedente sulla stessa casella), `circ_dist` (distanza ciclica 1…90), `digit_sum` (somma cifre, es. 65→11), `digital_root` (radice digitale fino a una cifra), `complement_90` (=90−n), `complement_digit_sum` / `complement_digital_root`, più colonne **lag** (`n_lag5`, `diff_signed_lag5`, …). |
| `*_diffsigned_*grams.csv` | N-grammi più frequenti sulla sequenza delle **differenze** (bigrammi, trigrammi, 4-grammi). |
| `*_complement90_*grams.csv`, `*_complement_digitsum_*grams.csv`, `*_digitsum_*grams.csv` | N-grammi su altre trasformazioni. |
| `*_autocorrelation.json` | Correlazione di Pearson tra \(x_t\) e \(x_{t+h}\) per vari `h` su `n`, differenze, complemento, ecc. Valori vicini a **0** sono attesi se la serie somiglia a rumore i.i.d.; picchi isolati vanno interpretati con cautela (molti lag testati). |

L’idea è avere **liste lunghe** di interi derivati su cui cercare pattern; ogni sequenza va letta insieme alle cautele su **multiple testing** e casualità del gioco.

## Ricerca automatica di struttura (algoritmi semplici + confronto null)

Dopo aver generato un `*_series.csv`, dalla cartella `franknet_lotto`:

```bash
python scripts/discover_sequence_structure.py --series-csv data/analysis/single_cell/MILANO_pos1_y1900_series.csv
```

(Ometti `--series-csv` per usare il default MILANO pos1 dal 1900.)

Lo script produce **`…_structure_report.json`** e **`…_structure_report.md`** con:

- **Catene di Markov** (ordini 0–3) su `digital_root`, `digit_sum`, differenze in **12 bin quantili**, `n mod 10`; split cronologico train/test.
- **AR(1)** su `n` grezzo (coefficiente, R² su train, MAE su test vs previsione “sempre la media”).
- **Motivi contigui** ripetuti (bigrammi/trigrammi di differenze; 3–5 grammi su radice digitale).
- **Test a permutazione**: stesso Markov su serie **mescolate** — se il guadagno reale non supera quello tipico dello shuffle, la “struttura” è debole.

Interpretazione: non “trova il vero algoritmo del lotto”, ma **misura** se modelli semplici si comportano meglio del caso su dati tenuti fuori dal training (e rispetto a null di shuffle).

## Mappa temporale delle radici digitali (per ruota)

Dopo `parse_draws.py`, dalla cartella `franknet_lotto`:

```bash
python scripts/digital_root_temporal_map.py --wheel MILANO --from-year 1900 --window 400 --step 100 --min-train 600
```

Sotto `data/analysis/digital_root/<ruota>/` trovi:

- **`pos1`…`pos5/`** — per ogni posizione sulla ruota:
  - `rolling_transitions_long.csv`: **P(radice successiva | radice corrente)** per ogni finestra temporale (heatmap animabile).
  - `rolling_marginals.csv`: entropia e moda nella finestra.
  - `walkforward_markov1.csv` / `walkforward_markov2.csv`: test **senza look-ahead** (accuratezza cumulativa nel tempo).
  - `last_window_transition_matrix.json`: matrice 9×9 dell’**ultima** finestra (istantanea “regime” recente).
- **`joint_5tuple_top40.csv`**: tuple (5 radici) più frequenti su quella ruota.
- **`summary.json`**: confronto con baseline (moda globale) e con 1/9; χ² su tabella decennio × radice (non omogeneità nel tempo).
- **`README_MAP.md`**: guida e limiti per uso previsionale.

Le **combinazioni** complete hanno spazio enorme; i CSV servono a **visualizzare** e **misurare** eventuali deviazioni, non a garantire previsioni corrette.

## Modelli complessi (scikit-learn)

Installazione una tantum:

```bash
python -m pip install -r requirements_ml.txt
```

Script:

```bash
python scripts/boosted_dr_predictor.py --wheel MILANO --target-position 1 --lag-draws 5
```

Opzioni: `--test-fraction`, `--walkforward-refit N` (ricalcolo periodico sulla coda della serie).

Output: `data/analysis/ml_models/<ruota>_pos<k>_lag<L>_boosted_report.json` con:

- **HistGradientBoosting** e **Random Forest** (50 feature = 5 estrazioni × 5 numeri × `{n, dr}` per ogni ritardo);
- **accuracy** e **log-loss** sul test cronologico; confronto con probabilità **uniformi** su 9 classi (`log(9)`: log-loss più bassa = migliore);
- **importanze** (Random Forest): su dati quasi i.i.d. tendono a essere **quasi piatte** (nessuna variabile domina).

Su MILANO pos1 (esempio reale): l’accuracy del boosting può superare leggermente la moda, ma la **log-loss** può restare **peggiore** dell’uniforme → modello **sovradattato** o male calibrato rispetto a un “indovina 1/9”. È un risultato utile: la complessità non implica vantaggio previsionale.
