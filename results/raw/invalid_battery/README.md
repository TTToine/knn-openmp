# Misure NON valide (portatile a batteria)

Durante la campagna `run_all.sh` del 2026-11-15 il caricatore è stato staccato
(batteria 100% in carica all'avvio, 71% in scarica alla fine). A batteria Windows
riduce i limiti di potenza della CPU: lo stesso `seq` (k=5, euclidea, 384 query)
passa da 19.1 s (inizio campagna) a 28.8 s (fine). Gli esperimenti eseguiti dalle
19:04 in poi (tiles, cachefit, vec, kmetric) sono 14%-2x più lenti e con varianza
alta: conservati solo per trasparenza, **non usarli nel report**. Da rifare con il
portatile in carica. Le accuratezze (`../accuracy_full_*.csv`) restano valide: non
dipendono dalla velocità della CPU.
