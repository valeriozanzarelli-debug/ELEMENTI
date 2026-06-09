# Mappa temporale — radici digitali (MILANO, da 1900)

## Cartelle `pos1` … `pos5`

- **`rolling_transitions_long.csv`**: per ogni finestra scorrevole, tutte le coppie (from_dr → to_dr) con conteggi e **P(to | from)**. Usala per heatmap 9×9 animate nel tempo (`slice_idx` / date).
- **`rolling_marginals.csv`**: entropia e moda della distribuzione marginale delle radici nella finestra (quanto è “piatta” 1…9).
- **`walkforward_markov1.csv` / `walkforward_markov2.csv`**: previsione **solo dal passato** (nessun look-ahead): ad ogni estrazione si aggiorna la catena e si predice la radice successiva. Colonna `rolling_accuracy` = accuratezza cumulativa dal primo step di test.
- **Baseline** in `summary.json`: stesso schema ma predici sempre la **moda globale** del passato; **uniforme** = 1/9 ≈ 0.111.

## File comuni alla ruota

- **`joint_5tuple_top40.csv`**: combinazioni di 5 radici (una riga estrazione) più frequenti nello storico — spazio 9⁵, atteso molto sparso sotto indipendenza.

## Uso per “previsione”

Qualsiasi modello va **rivalidato** con walk-forward o cross-validation temporale. Le combinazioni complete hanno cardinalità enorme: la probabilità sotto ipotesi i.i.d. uniforme sui numeri del lotto **non** coincide con uniforme sulle radici, ma **nessuna analisi storica** giustifica certezza sulle uscite future.

Chi² decenni (in `summary.json`): valori alti ⇒ marginale delle radici **non omogenea** tra decenni (regole, archivio, non-stazionarietà); non implica prevedibilità.

Generato da: `scripts/digital_root_temporal_map.py`
