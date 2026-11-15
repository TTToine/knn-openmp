Rivedi la variante $ARGUMENTS dal punto di vista delle prestazioni e della correttezza.

Controlla: race condition, variabili private/shared, false sharing, allocazioni dentro
la regione cronometrata o dentro i loop caldi, accessi in memoria non contigui,
loop che il compilatore non vettorizza (usa la build con KNN_VEC_REPORT), overhead
di scheduling. Per ogni problema indica gravità, causa e correzione proposta, ma non
modificare il codice finché non confermo.
