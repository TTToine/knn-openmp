Prepara ed esegui un esperimento di benchmark: $ARGUMENTS

1. Verifica che la build Release sia aggiornata e che `ctest` passi.
2. Stima la durata dell'esperimento (runs × configurazioni × tempo di un run) e
   chiedimi conferma prima di lanciarlo se supera 2 minuti, ricordandomi di mettere
   il portatile in carica e in modalità prestazioni.
3. Lancia `scripts/run_benchmarks.py` con i parametri adatti, poi
   `scripts/plot_results.py` sui CSV prodotti.
4. Commenta i risultati: speedup, efficienza, deviazione standard, anomalie.
   Formula ipotesi sulle cause (Amdahl, banda di memoria, hyperthreading,
   load imbalance, overhead di scheduling) e suggerisci come verificarle.
5. Aggiungi la voce corrispondente in `report/notes.md`.
