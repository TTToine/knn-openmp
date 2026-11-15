# k-NN classifier — OpenMP + SIMD (Parallel Programming, mid-term)

Progetto d'esame (UniFi, Parallel Programming, 6 CFU, individuale).
Obiettivo: classificatore k-NN esatto (brute force) su MNIST, versione sequenziale
e versioni parallele OpenMP + vettorizzazione, con analisi delle prestazioni.
La valutazione pesa: correttezza 25%, prestazioni 25%, metodologia sperimentale 20%,
report/presentazione 20%, codice 10%. **L'analisi conta quanto il codice.**

## Come lavorare con me (Claude)
- Lo studente deve saper spiegare ogni riga all'orale. Quando introduci una direttiva
  OpenMP, un intrinsic o un'ottimizzazione, spiega in 2-3 frasi *perché* funziona e a
  quale concetto del corso si collega (false sharing, AoS/SoA, cache, Amdahl, ...).
- Una variante = una sola idea di ottimizzazione, così il report può confrontarle.
- Procedi a piccoli passi: implementa → `ctest` → benchmark breve → annota in
  `report/notes.md`. Non accumulare più modifiche non validate.
- Prima di lanciare benchmark lunghi (> 1-2 minuti) chiedi conferma: le misure vanno
  fatte con il portatile in carica, modalità prestazioni, nessun'altra app pesante.
- Commit piccoli e descrittivi (`feat: add omp_simd variant`, `bench: schedule sweep`).

## Ambiente
- Laptop: Intel i7-10750H (6 core / 12 thread, AVX2, no AVX-512), 16 GB RAM.
  Cache (verifica con `lscpu`): L1d 32 KB/core, L2 256 KB/core, L3 12 MB condivisa.
- Windows 11 + **WSL2 Ubuntu**, GCC con OpenMP (non usare MSVC: supporta solo OpenMP 2.0).

## Comandi
```bash
cmake -S . -B build && cmake --build build -j          # build Release (-O3 -march=native)
ctest --test-dir build --output-on-failure             # valida TUTTE le varianti
cmake -S . -B build-novec -DKNN_NO_VEC=ON && cmake --build build-novec -j   # senza auto-vec
cmake -S . -B build-vecrep -DKNN_VEC_REPORT=ON && cmake --build build-vecrep 2>&1 | grep knn_
bash scripts/download_mnist.sh                         # MNIST in data/mnist
./build/knn --list                                     # varianti registrate
OMP_SCHEDULE=static ./build/knn --data data/mnist --n-test 500 --variant omp_naive --threads 6
python3 scripts/run_benchmarks.py threads  --variants omp_naive        # strong scaling
python3 scripts/run_benchmarks.py schedule --variants omp_naive --threads 12
python3 scripts/plot_results.py results/raw/threads_*.csv              # grafici + summary
```

## Architettura
- `include/dataset.hpp`, `src/dataset.cpp` — `Dataset` row-major allineato a 64 B,
  loader IDX (MNIST), dati sintetici, `replicate()` per raggiungere ≥ 10 s sequenziali.
- `include/knn_common.hpp` — mattoni condivisi: distanze scalari, `TopK` (insertion
  sort, tie-break deterministico), voto di maggioranza, norme.
- `src/knn_seq.cpp` — **riferimento sequenziale: oracolo di correttezza e baseline.**
- `src/knn_omp.cpp` — varianti parallele, una funzione per idea.
- `src/variants.cpp` — registro delle varianti (nome → funzione).
- `src/main.cpp` — driver: carica i dati (non cronometrato), esegue warm-up + run,
  cronometra solo la chiamata k-NN con `omp_get_wtime`, valida contro il riferimento
  (cache in `.cache/`), scrive una riga CSV per run.
- `scripts/` — benchmark, grafici, info di sistema. `tests/smoke_test.sh` — validazione.

### Aggiungere una variante
1. Dichiarala in `include/knn.hpp`, implementala in `src/knn_omp.cpp`
   (stessa firma `KnnFn`).
2. Registrala in `src/variants.cpp`.
3. `ctest`: deve passare con 0 mismatch. Se la variante riordina le somme float
   (es. `omp simd reduction`), alcune predizioni possono cambiare per pareggi quasi
   esatti: misura quante, spiegalo nel report e usa `MAX_MISMATCH=...` consapevolmente.

## Regole invarianti
- **Non modificare `knn_seq.cpp`** né la semantica di `knn_common.hpp` senza chiedere:
  cambierebbe il riferimento. I kernel ottimizzati vanno nelle varianti.
- Mai `-ffast-math` globale. Mai I/O o caricamento dati dentro la regione cronometrata.
- Il riferimento è compilato con gli stessi flag delle varianti (confronto equo).
- Con `schedule(runtime)` imposta sempre `OMP_SCHEDULE`: se non è definita, libgomp
  usa `dynamic,1` (lo script lo fa già).
- I risultati grezzi in `results/raw/` si committano (riproducibilità); `data/`,
  `build*/`, `.cache/` no.

## Roadmap varianti (dalle slide della traccia)
1. [x] `omp_naive` — `parallel for` sui campioni di test, `schedule(runtime)`.
   Esperimenti: strong scaling 1-12 thread, confronto scheduling/chunk.
2. [x] `omp_simd` — `#pragma omp simd reduction(+:s)` sul loop delle distanze;
   confronto con/senza vettorizzazione (`build-novec`), report di vettorizzazione.
3. [ ] Layout dati AoS vs SoA — training set trasposto (feature-major): si vettorizza
   *su più campioni di training* invece che sulle dimensioni, senza riduzione
   orizzontale e con lo stesso ordine di somma dello scalare.
4. [ ] Blocking — tile di query × tile di training che restano in L2 (una riga MNIST
   = 3136 B, ~80 righe in 256 KB): misura l'effetto sulla cache.
5. [ ] Formulazione matriciale — |a|² + |b|² − 2 a·b con prodotto a blocchi;
   attenzione alla cancellazione numerica.
6. [ ] Top-k: insertion sort vs heap al crescere di k.
7. [ ] Extra: weak scaling, analisi bottleneck (compute vs memory bound), confronto
   con Amdahl.

## Domande aperte da discutere nel report
- Oltre 6 thread l'hyperthreading quanto aiuta? (stesse unità AVX condivise)
- Il problema è compute-bound o memory-bound per ogni variante? (stima GFLOP/s e GB/s)
- `out[i]` è un array di `uint8_t` scritto da thread diversi: c'è false sharing? perché
  è (o non è) rilevante qui?
