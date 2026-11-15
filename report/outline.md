# Report outline (dalle linee guida del docente)

1. **Abstract** — problema, approccio, risultato chiave (miglior speedup).
2. **Introduction** — k-NN, perché è costoso (O(N_test · N_train · D)), obiettivi.
3. **Algorithm** — versione sequenziale, sezioni parallelizzabili.
4. **Parallel implementation** — strategia, direttive OpenMP, sincronizzazione,
   vettorizzazione, layout dati, problemi incontrati e soluzioni.
5. **Experimental setup** — hardware (`scripts/system_info.sh`), compilatore e flag,
   dataset, parametri, metodologia (warm-up, ≥ 5 run, mediana ± std).
6. **Results** — validazione della correttezza, curve di speedup per più dimensioni,
   strong (e weak) scaling, effetto di schedule/chunk, confronto varianti,
   analisi dei colli di bottiglia. Grafici con assi, legenda e didascalia.
7. **Discussion** — confronto con Amdahl/Gustafson, overhead, limiti, lezioni apprese.
8. **Conclusions** — configurazione migliore, sviluppi futuri.
9. **References**

Presentazione: 10-12 min (problema 1-2, approccio 3-4, risultati 4-5, conclusioni 1-2).
