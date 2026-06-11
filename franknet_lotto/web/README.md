# Super Motore Lotto — Web

## Avvio

```bash
cd franknet_lotto/scripts
python3 super_engine_server.py
```

Apri **http://localhost:8765**

## Uso

1. Scegli la **ruota** (su cui giochi)
2. Inserisci l'**ultima cinquina di quella ruota** nell'ordine di estrazione (pos1 → pos5)
3. Clicca **Calcola** → ambo, terno, quaterna, cinquina

**Data e n° concorso non servono** — il calcolo usa solo i 5 numeri.

Il metodo è **cross_opt_v1** (complementi, vertibili, diff, incroci). Funziona anche offline nel browser.
