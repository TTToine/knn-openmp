# Preparazione all'orale — traccia per la simulazione d'esame (fase F)

**Stato: da fare.** La simulazione è interattiva (una domanda alla volta, risposta
dello studente, correzione). Questo file è la traccia: domande, risposta modello
breve, dove sta la prova nel repository. Dopo la simulazione aggiungere in fondo i
punti in cui lo studente ha esitato.

I numeri marcati **[M]** vengono dalla campagna valida (`threads_20261115_184752`);
quelli marcati **[P]** sono previsioni non ancora misurate (`HANDOFF.md`): all'orale
vanno sostituiti con le misure, o presentati come previsioni.

## A. Parallelizzazione di base

1. **Perché il k-NN è "embarrassingly parallel"? Dove non lo è?**
   Ogni query è indipendente: legge training e query (sola lettura) e scrive solo
   il suo `out[i]`. Nessuna sincronizzazione. Non lo sono: il top-k *di una query*
   (stato sequenziale: per questo è privato del thread) e qualunque statistica
   globale (matrice di confusione → `tools/sync_strategies.cpp`).
2. **Quali variabili sono shared e quali private in `omp_naive`? Perché non c'è
   una race?** `default(none)`: `train`, `test`, `params`, norme = shared in sola
   lettura; `out` shared ma ogni elemento scritto da un solo thread; `top`, `q`,
   `dist` dichiarati nel loop = privati. Verifica: ThreadSanitizer (review §4.1).
3. **C'è false sharing su `out`?** In teoria sì (linee di confine tra thread), in
   pratica una scrittura per query contro ~10⁸ operazioni. Per *vedere* il false
   sharing: `sync_strategies`, contatori `packed` contro `padded` [P].
4. **critical, atomic, reduction: differenze e quando usarle?** critical = lock
   globale, serializza; atomic = una sola istruzione atomica, niente lock ma la
   linea di cache rimbalza; reduction = copia privata + combinazione finale,
   nessuna condivisione durante il loop. Per un istogramma: reduction.
5. **Perché `schedule(static)` basta qui? Quando servirebbe `dynamic`?** Carico
   uniforme (ogni query costa uguale) → static ha overhead nullo. Dynamic serve
   con carico variabile: il mean shift del final-term (iterazioni per pixel).

## B. Vettorizzazione

6. **GCC vettorizza `omp_naive`; perché non è veloce?** Vettorizza `sub` e `mul`,
   ma senza `-ffast-math` la somma deve restare in ordine: 8 `vaddss` dipendenti
   per iterazione (fold-left). Limite: latenza dell'addizione, 4 cicli per elemento.
7. **Cosa autorizza `omp simd reduction(+:s)`? Cambia il risultato?** Autorizza a
   riassociare: 8 somme parziali, una per corsia. Cambia l'ordine delle somme →
   ultimi bit diversi; per questo ogni variante è validata (0 mismatch su 10000
   query, ma non è garantito a priori).
8. **Intrinsics a mano o `omp simd`?** Per il kernel 1×1 e 4×1 il codice macchina
   è lo stesso (`tools/asm_loops.py`). Gli intrinsics servono dove la direttiva
   non arriva: accumulatori multipli sulla stessa somma, micro-kernel MR×NR
   (la riduzione su array viene vettorizzata male), early abandoning.
9. **AoS o SoA?** SoA vettorizza su 64 campioni invece che su 784 feature: niente
   riduzione orizzontale, stesso ordine di somma. Più veloce in L2, più lento su
   MNIST intero (784 flussi di memoria); AoSoA corregge il pattern di accesso.

## C. Memoria e modello di prestazioni

10. **`omp_simd` è compute-bound o memory-bound? Come lo dimostri?** Memory-bound:
    ogni query rilegge 188 MB; a 1 thread 10.9 GB/s **[M]** = limite di un core
    (legge di Little sui line fill buffer). Prova: cache-fit (stesso kernel con
    training in L2/L3/DRAM) e roofline.
11. **Lo scaling "quasi ideale" di `omp_simd` fino a 8 thread: la memoria scala?**
    No: 99.6 GB/s apparenti **[M]** = 4.2× il picco DRAM (23.5 GB/s, single
    channel). I thread scorrono le stesse righe quasi insieme: uno le porta in L3,
    gli altri le trovano lì ("convoglio"). Verifica per intervento:
    `omp_simd_stagger` [P].
12. **Perché crolla a 12 thread?** Fatto: a 12 thread tutte le varianti senza
    blocking leggono 22–26 GB/s = picco DRAM (flussi indipendenti). Causa: tre
    ipotesi (review §1-B), non ancora discriminate. *Dire che non lo sappiamo
    ancora è la risposta giusta.*
13. **Cosa fa il cache blocking e quanto vale?** Blocchi di 16 query × tile di 32
    righe (100 KB in L2): una tile letta dalla DRAM una volta, riusata 16 volte.
    Traffico DRAM ÷ 16. Non lo elimina: per `omp_blocked4` la lettura è metà del
    tempo già a 1 thread (modello a due termini, `perf_model.md` §4).
14. **Cos'è il register blocking / micro-kernel?** 4 query × 1 riga: la riga è
    caricata una volta e usata 4 volte, 4 catene FMA indipendenti nascondono la
    latenza. MR×NR generalizza. Per l'euclidea il 4×1 è già al limite delle porte
    (2 µop per elemento, 2 porte); per il prodotto scalare servono 8 accumulatori
    [P]. Il 6×2 usa 16 registri su 16 → spill (visto nell'assembly).
15. **Hyperthreading: aiuta?** Due thread sullo stesso core condividono porte FP,
    L1 e L2. Aiuta se il thread aspetta (latenza, memoria), non se le porte sono
    già sature. Con 2 hyperthread il working set del blocking supera la L2.
16. **Amdahl sui nostri numeri?** Frazione seriale concreta: `row_norms()` del
    coseno (un thread, dentro il tempo). `omp_blocked4_pn` la parallelizza [P].
    Ma il limite vero non è Amdahl: è la banda di memoria condivisa.
17. **Speedup rispetto a cosa?** Due metriche distinte: T_seq/T_p (definizione
    della traccia: SIMD + thread) e T_var(1)/T_var(p) (solo thread). Efficienza =
    la seconda / p; la prima / p supererebbe 1.
18. **Weak scaling e Gustafson?** Lavoro per thread costante; efficienza debole
    T(1)/T(p); speedup scalato p − s(p − 1). Qui il limite è la DRAM: far crescere
    il problema non la allarga [P].

## D. Algoritmo e numerica

19. **Heap o insertion sort per il top-k?** Indifferente: gli inserimenti attesi
    sono k(1 + ln(n/k)) ≈ 52 su 60000 per k = 5; contati davvero da
    `omp_gemm_exact` (53.6). Il top-k è < 0.1% del lavoro.
20. **Early abandoning: perché è esatto?** Le somme parziali di quadrati sono non
    decrescenti *anche in float* (arrotondamento monotono, termini ≥ 0): parziale ≥
    soglia ⇒ distanza finale ≥ soglia ⇒ il top-k l'avrebbe rifiutata. Valuta il
    41% delle feature. Costo: un salto imprevedibile per candidato.
21. **Perché non early abandoning + micro-kernel?** Uno decide per coppia, l'altro
    fa avanzare 4–8 coppie insieme: fare meno lavoro contro farlo più in fretta.
22. **|a|² + |b|² − 2a·b: perché è più veloce e perché è pericolosa?** 1 µop per
    elemento invece di 2; ma cancellazione: errore assoluto ∝ |a|² + |b|², non
    alla distanza. Su MNIST innocua, con dati traslati di 1024 il 28% delle
    distanze è negativo e l'accuratezza crolla al 22%. Rimedio: usarla come
    filtro e ricalcolare esattamente i candidati (`omp_gemm_exact`).
23. **k e metrica: cosa cambia?** Accuratezze 96.8–97.4%; differenze tra k dentro
    l'incertezza statistica (±0.17 punti); coseno +0.3–0.4 punti, più solido.

## E. Metodologia

24. **Come sai che i tempi sono affidabili?** Warm-up scartato, ≥ 5 run, mediana e
    min–max, thread fissati sui core, ordine delle varianti ruotato, stato di
    alimentazione registrato. Metà della prima campagna è stata buttata perché il
    portatile era a batteria: documentato, non nascosto.
25. **Limiti dichiarati?** WSL2 (niente `perf`, frequenza non leggibile: sonda
    `hwprobe`), RAM single channel, convoglio descritto ma non modellato.

## Punti in cui lo studente ha esitato

(da compilare dopo la simulazione)
