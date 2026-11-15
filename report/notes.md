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
