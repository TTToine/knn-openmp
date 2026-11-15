# Lab notebook

Annota qui ogni esperimento appena fatto: è la materia prima del report.
Formato suggerito (una voce per esperimento):

## AAAA-MM-GG — titolo
- **Ipotesi:** cosa mi aspetto e perché
- **Configurazione:** variante, thread, schedule, dati, commit (`git rev-parse --short HEAD`)
- **Comando:** `...`
- **File risultati:** `results/raw/...csv`
- **Risultato:** numeri chiave (speedup, tempi)
- **Interpretazione:** confermata/smentita? perché?
- **Prossimo passo:**

---

## 2026-11-15 — omp_simd: `omp simd reduction` sul loop delle distanze
- **Ipotesi:** senza `-ffast-math` la riduzione float in `sq_euclidean` resta a ordine
  fisso → catena di `add` dipendenti (latenza ~4 cicli ciascuna). `omp simd reduction(+:s)`
  autorizza la riassociazione: 8 somme parziali (una per lane AVX2) → fino a 8× sul kernel.
  Con più thread il guadagno dovrebbe calare: ogni query rilegge tutto il training
  (60000×784×4 B = 188 MB ≫ L3 12 MB) → possibile passaggio a memory-bound.
- **Configurazione:** omp_naive vs omp_simd, 1 e 6 thread, `OMP_SCHEDULE=static`, MNIST
  60000 train / 100 test, k=5, euclidea, 3 run + 1 warm-up. Base: commit `5320056` + modifiche omp_simd.
- **Comando:** `OMP_SCHEDULE=static ./build/knn --data data/mnist --n-test 100 --variant {omp_naive,omp_simd} --threads {1,6} --runs 3 --warmup 1`
- **File risultati:** nessuno (prova rapida, solo stdout); da ripetere con `run_benchmarks.py`.
- **Verifica vettorizzazione** (`build-vecrep` + `objdump` del corpo parallelo):
  - `omp_naive`: GCC *vettorizza comunque* il loop (report: "32 byte vectors"), ma solo
    `sub`/`mul` (`vsubps`, `vmulps`); la somma è una *fold-left reduction* in ordine
    (8 `vaddss` scalari per iterazione) → la catena di dipendenza resta.
  - `omp_simd`: `vfmadd231ps` (8 accumulatori in un registro, FMA), `vaddss` solo nella
    riduzione orizzontale finale.
- **Risultato** (mediana; GFLOP/s con 3 flop/elemento, 14.1 GFLOP totali):

  | variante  | thr | tempo   | GFLOP/s | speedup vs naive stesso thr |
  |-----------|-----|---------|---------|-----------------------------|
  | omp_naive | 1   | 5.048 s | 2.8     | 1.00×                       |
  | omp_simd  | 1   | 1.721 s | 8.2     | 2.93×                       |
  | omp_naive | 6   | 0.879 s | 16.1    | 1.00×                       |
  | omp_simd  | 6   | 0.364 s | 38.7    | 2.41×                       |

  Scalabilità 1→6 thread: naive 5.74×, simd 4.72×. Mismatch: 0 (MNIST, 100 query) e
  ctest con `MAX_MISMATCH=0` passato (dati sintetici, entrambe le metriche).
- **Interpretazione:** ipotesi confermata in parte. Il guadagno SIMD (2.9×) è molto sotto
  8× già con 1 thread. Banda implicita con 1 thread: 18.8 GB letti / 1.72 s ≈ 11 GB/s,
  plausibile limite di banda di un singolo core → probabilmente memory-bound già da 1
  thread. Con 6 thread la banda "apparente" sarebbe ~52 GB/s, sopra il picco DRAM (DDR4
  dual channel ≈ 47 GB/s teorici): i thread con `static` avanzano in sincronia sul
  training e si condividono le righe in L3. La scalabilità peggiore di simd (4.7× vs 5.7×)
  è coerente con un collo di bottiglia sulla memoria condivisa.
  0 mismatch nonostante il riordino: le distanze tra vicini diversi raramente differiscono
  di pochi ulp; da ricontrollare sulle 10000 query.
- **Prossimo passo:** (1) confermare memory-bound: stessa prova con training piccolo che
  sta in L2/L3 → il guadagno SIMD dovrebbe avvicinarsi a 8×; (2) confronto `build-novec`;
  (3) strong scaling 1-12 thread con lo script; (4) blocking (roadmap 4) per riusare il
  training dalla cache.

---

## 2026-11-15 — omp_simd è memory-bound? Training in L2 / L3 / DRAM
- **Ipotesi:** se il guadagno SIMD su MNIST completo (2.9×) è limitato dalla banda di
  memoria, riducendo il training finché sta in cache il guadagno deve salire verso 8×.
  omp_naive invece dovrebbe avere GFLOP/s quasi costanti (limitato dalla catena di `add`).
- **Configurazione:** commit `191293f`, `OMP_SCHEDULE=static`, k=5, euclidea, 3 run + 1
  warm-up. Righe MNIST = 3136 B. Casi: 64 train × 10000 test (200 KB, sta in L2 256 KB),
  2000 × 2000 (6.3 MB, sta in L3 12 MB), 60000 × 100 (188 MB, DRAM).
- **Comando:** `bash scripts/cachefit.sh`
- **File risultati:** `results/raw/cachefit_20261115_180257.csv`
- **Risultato** (mediana; GFLOP/s con 3 flop/elemento):

  | training     | thr | naive GFLOP/s | simd GFLOP/s | simd/naive |
  |--------------|-----|---------------|--------------|------------|
  | 64 (L2)      | 1   | 2.9           | 20.8         | **7.2×**   |
  | 2000 (L3)    | 1   | 2.6           | 12.9         | **5.0×**   |
  | 60000 (DRAM) | 1   | 2.5           | 7.2          | **2.8×**   |
  | 64 (L2)      | 6   | 17.5          | 109          | 6.2×       |
  | 2000 (L3)    | 6   | 16.6          | 80.7         | 4.9×       |
  | 60000 (DRAM) | 6   | 15.4          | 30.4         | 2.0× (*)   |

  (*) misura rumorosa: simd 6 thr std 0.41 s, min 0.35 / max 1.10 s (nella prova
  precedente 0.36 s). Anche naive DRAM 1 thr è ~10% più lento della prova precedente
  (5.55 vs 5.05 s): probabile variazione di frequenza/termica. Da ripetere con più run e
  portatile in condizioni controllate. 0 mismatch in tutti i casi.
- **Interpretazione:** ipotesi confermata. Con 1 thread il guadagno SIMD cala
  7.2× → 5.0× → 2.8× man mano che i dati escono da L2, poi da L3. omp_naive ha
  GFLOP/s quasi costanti (2.5-2.9) indipendentemente da dove stanno i dati → è
  **compute-bound**; omp_simd su MNIST completo è **memory-bound**.
  Banda implicita omp_simd 1 thread: L3 ≈ 17 GB/s, DRAM ≈ 9.6 GB/s.
  Modello a latenza (da verificare): naive fa 8 `vaddss` dipendenti per 8 elementi,
  latenza ~4 cicli → ~0.25 elementi/ciclo ≈ 3.4 GFLOP/s a ~4.5 GHz (misurato 2.9).
  simd ha un solo accumulatore vettoriale → 1 `vfmadd231ps` ogni ~4 cicli → ~2
  elementi/ciclo ≈ 27 GFLOP/s (misurato 20.8 in L2). Quindi anche in L2 simd è limitato
  dalla latenza dell'FMA, non dal throughput (2 FMA/ciclo possibili).
- **Prossimo passo:** il blocking (roadmap 4) attacca proprio il limite di memoria:
  riusare ogni tile di training per più query mentre è in cache. Più accumulatori
  indipendenti (unrolling della riduzione) attaccherebbero il limite di latenza in L2.

---
