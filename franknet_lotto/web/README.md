# Super Motore Lotto — Web

## Avvio

```bash
cd franknet_lotto/scripts
python3 super_engine_server.py
```

Apri **http://localhost:8765**

## Uso

1. Scegli la **ruota**
2. Inserisci l'**ultima cinquina** nell'ordine di estrazione (pos1 → pos5)
3. Clicca **Calcola** → ottieni ambo, terno, quaterna, cinquina

Senza server Python funziona solo il metodo **cross** (sufficiente per NAZIONALE ambo).

Con server attivo si aggiunge il layer **science** (armoniche + fase + k-NN su storico Franknet).
