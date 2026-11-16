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
  thread. Con 6 thread la banda "apparente" sarebbe ~52 GB/s, sopra il picco DRAM:
  i thread con `static` avanzano in sincronia sul training e si condividono le righe
  in L3. **[Corretto 2026-11-15]** il portatile ha *un solo* modulo DDR4-2933 (single
  channel, verificato con `Win32_PhysicalMemory`): picco ≈ 2933e6 × 8 B = **23.5 GB/s**,
  non 47. I 52 GB/s apparenti sono quindi 2.2× il picco → la condivisione in L3 è
  ancora più evidente. La scalabilità peggiore di simd (4.7× vs 5.7×)
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

## 2026-11-15 — Con vs senza auto-vettorizzazione (`build-novec`)
- **Ipotesi:** `-fno-tree-vectorize` (a) non tocca omp_simd, perché `#pragma omp simd`
  forza la vettorizzazione del loop; (b) cambia poco omp_naive: con auto-vec GCC
  vettorizza solo `sub`/`mul` ma somma in ordine (fold-left), senza ha una FMA scalare
  per elemento. In entrambi i casi il collo di bottiglia è una operazione dipendente per
  elemento (~4 cicli di latenza) → stesso throughput.
- **Configurazione:** commit `3dcb380`, 1 thread (la vettorizzazione agisce dentro un
  core), `OMP_SCHEDULE=static`, k=5, euclidea, 3 run + 1 warm-up. Training 64 righe
  (in L2) × 10000 test e 60000 righe (DRAM) × 100 test.
- **Comando:** `cmake -S . -B build-novec -DKNN_NO_VEC=ON && cmake --build build-novec -j && bash scripts/novec.sh`
- **File risultati:** `results/raw/novec_20261115_181101.csv`
- **Assembly** (istruzioni float nel corpo parallelo, `objdump`):
  - omp_naive `build`: `vsubps`, `vmulps` + catena di `vaddss` (fold-left reduction)
  - omp_naive `build-novec`: solo scalare, `vsubss` + `vfmadd231ss` (`s += t*t` fusa)
  - omp_simd: identica nelle due build (`vfmadd231ps`) → conferma (a)
- **Risultato** (mediana, 1 thread):

  | training     | variante  | vec      | novec    | vec/novec |
  |--------------|-----------|----------|----------|-----------|
  | 64 (L2)      | omp_naive | 0.520 s  | 0.529 s  | 1.02×     |
  | 64 (L2)      | omp_simd  | 0.072 s  | 0.073 s  | 1.01×     |
  | 60000 (DRAM) | omp_naive | 5.546 s  | 5.761 s  | 1.04×     |
  | 60000 (DRAM) | omp_simd  | 1.972 s  | 1.990 s  | 1.01×     |

  0 mismatch ovunque. NB: la cache del riferimento (`.cache/`) non distingue la build,
  quindi anche la novec è stata validata contro il riferimento della build normale:
  nonostante FMA scalare vs `vmulps`+`vaddss` (arrotondamenti diversi) le predizioni
  coincidono.
- **Interpretazione:** entrambe le ipotesi confermate. L'auto-vettorizzazione di GCC
  dà solo 2-4% su omp_naive: vettorizza la parte "facile" (sub/mul) ma non può
  riassociare la riduzione senza `-ffast-math`, quindi resta latency-bound. Il guadagno
  di omp_simd (7.2× in L2) viene tutto dal permesso esplicito di riordinare le somme,
  non dalla vettorizzazione automatica. Messaggio per il report: "vettorizzato" nel
  report di GCC ≠ veloce; bisogna guardare *come* è fatta la riduzione.
- **Prossimo passo:** variante blocking (roadmap 4); in alternativa più accumulatori
  per superare il limite di latenza dell'FMA in L2.

---

## 2026-11-15 — omp_blocked: cache blocking query × training
- **Ipotesi:** omp_simd su MNIST completo è memory-bound (ogni query rilegge 188 MB).
  Con blocchi di QB=16 query × tile di TB=32 righe di training (100 KB, sta in L2 anche
  con 2 hyperthread per core) ogni tile viene letto dalla DRAM una volta e riusato 16
  volte → traffico DRAM ÷16. Mi aspetto di tornare al regime "training in L2" misurato
  nell'esperimento cache-fit: ~21 GFLOP/s con 1 thread, ~100 con 6.
- **Correttezza:** ogni query vede ancora j crescente → stessa sequenza di `push` in
  TopK, stessi pareggi. ctest passato (0 mismatch) con i default e anche con
  `KNN_TB/KNN_QB` = 1/1, 7/5, 13/97, 100000/1000 (resti di blocco e tile).
- **Configurazione:** commit `15b112c` + omp_blocked, `OMP_SCHEDULE=static`, MNIST 60000
  train × 192 test (12 blocchi di query → 2 per thread a 6 thread), k=5, euclidea,
  3 run + 1 warm-up, TB=32, QB=16.
- **Comando:** `OMP_SCHEDULE=static ./build/knn --data data/mnist --n-test 192 --variant {omp_simd,omp_blocked} --threads {1,6} --runs 3 --warmup 1`
- **File risultati:** nessuno (prova rapida, solo stdout).
- **Risultato** (mediana; 27.1 GFLOP totali):

  | variante    | thr | tempo   | GFLOP/s | blocked/simd |
  |-------------|-----|---------|---------|--------------|
  | omp_simd    | 1   | 3.857 s | 7.0     |              |
  | omp_blocked | 1   | 1.267 s | 21.4    | **3.0×**     |
  | omp_simd    | 6   | 1.717 s | 15.8 (*)|              |
  | omp_blocked | 6   | 0.263 s | 103     | 6.5× (*)     |

  (*) omp_simd a 6 thread è anomalo: std 0.30 s e solo 15.8 GFLOP/s contro i 38.7
  misurati con 100 query. Possibile spiegazione: con 100 query i thread `static`
  avanzano quasi in sincronia e si condividono le righe in L3; oppure rumore
  termico/frequenza. Il rapporto 6.5× va confermato con misure controllate.
  0 mismatch in tutti i casi.
- **Interpretazione:** ipotesi confermata. omp_blocked raggiunge le stesse prestazioni
  del caso "training in L2" (21.4 vs 20.8 GFLOP/s a 1 thread, 103 vs 109 a 6): il
  blocking elimina il collo di bottiglia della memoria. Traffico DRAM stimato a 6 thread:
  12 blocchi × 188 MB / 0.263 s ≈ 8.6 GB/s, ben sotto la banda disponibile → ora il
  limite torna a essere il calcolo (latenza della FMA con un solo accumulatore).
  Scalabilità 1→6 thread di omp_blocked: 4.8×.
- **Prossimo passo:** (1) sweep di TB e QB (`KNN_TB`, `KNN_QB`) per vedere il ginocchio
  quando il tile esce da L2; (2) misure controllate (portatile in carica) di
  strong scaling 1-12 thread per naive/simd/blocked; (3) più accumulatori nel kernel.
- **Nota sul budget di cache [aggiunta dopo revisione]:** il commento originale diceva
  che il tile da 100 KB sta in L2 "anche con 2 hyperthread". Ma ogni thread tiene anche
  le sue 16 query (50 KB): 150 KB per thread, 2 × 150 = 300 KB > 256 KB con 2 thread per
  core. Probabilmente le query vengono servite da L3 oltre i 6 thread: lo sweep TB × QB
  a 6 e 12 thread lo mostra. Commento nel codice corretto.
- **Nota sulla granularità [aggiunta dopo revisione]:** con 192 query e QB=16 i blocchi
  sono 12: a 8 thread 4 thread fanno 2 blocchi e 4 ne fanno 1 → efficienza massima
  teorica 12/(8·2) = 75%, un artefatto. Da qui in poi `--n-test 384` = 24 blocchi,
  divisibile per 1, 2, 4, 6, 8, 12.

---

## 2026-11-15 — Revisione della metodologia (prima della campagna ufficiale)
- **Metriche (plot_results.py):** l'efficienza era `T_seq / T_p / p`. Per le varianti
  SIMD lo speedup rispetto a seq è già > 1 con un thread (omp_blocked ≈ 4×), quindi
  quella "efficienza" superava 1 e mescolava guadagno SIMD e guadagno dei thread. Ora:
  - **speedup** = T_seq / T_p (definizione della traccia: guadagno totale);
  - **scalabilità** = T_variante(1) / T_variante(p) (solo il guadagno dei thread);
  - **efficienza** = scalabilità / p.
- **Tempo CPU:** il driver misura anche `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)` attorno
  alla chiamata (colonna `cpu_s`). CPU / wall = numero medio di core occupati. NB: un core
  in stallo sulla memoria conta come occupato, e i thread che aspettano a una barriera
  OpenMP fanno spin (consumano CPU): il tempo CPU non distingue calcolo e attesa.
- **Run order e binding:** `run_benchmarks.py` ruota l'ordine delle varianti a ogni
  punto (la deriva termica non penalizza sempre la stessa) e usa sempre
  `OMP_PROC_BIND=close OMP_PLACES=cores`. Bug trovato: con >6 thread usava
  `OMP_PLACES=threads`; su questa macchina le CPU 0-1, 2-3, … sono hyperthread dello
  stesso core (`lscpu -e`), quindi 8 thread `close` finivano su 4 core soltanto.
- **RAM:** un solo modulo DDR4 16 GB a 2933 MT/s (slot ChannelA vuoto) → single channel,
  picco 23.5 GB/s. Condizioni di misura verificate: alimentazione da rete (batteria
  100%), piano energetico "Prestazioni elevate".

## 2026-11-15 — Varianza di omp_simd a 6 thread
- **Ipotesi:** la varianza (0.35-1.10 s) dipende dal posizionamento dei thread (senza
  binding due thread possono finire sullo stesso core).
- **Comando:** `bash scripts/variance_probe.sh` (384 query, 10 run, 5 configurazioni)
- **File risultati:** `results/raw/variance_20261115_184437.csv`
- **Risultato:** mediane 1.33-1.38 s in *tutte* le configurazioni (unbound, close/cores,
  spread/threads, close/threads). `close/threads` mette 6 thread su **3 core** (CPU 0-5)
  eppure è veloce uguale. Un run isolato di close/cores ha impiegato 4.30 s (mediana 1.38).
- **Interpretazione:** ipotesi smentita. (1) omp_simd è così memory-bound che 3 o 6 core
  danno lo stesso tempo: conta solo la banda condivisa. (2) La varianza viene da outlier
  sporadici (attività dell'host Windows / WSL), non dal binding. Con 384 query: 39.9
  GFLOP/s, coerente con i 38.7 misurati con 100 query; l'anomalia del run a 192 query
  era transitoria. Banda apparente 384 × 188 MB / 1.35 s ≈ 53 GB/s = 2.3× il picco
  DRAM → righe condivise in L3. → Usare la **mediana** (robusta agli outlier) e
  riportare min/max.

## 2026-11-15 — AoS vs SoA vs AoSoA (prove preliminari, 3 run)
- **Ipotesi:** SoA (feature-major) vettorizza su 64 campioni di training con 8
  accumulatori indipendenti, senza riduzione orizzontale → kernel più veloce di AoS.
- **Assembly** (omp_soa): 8 `vfmadd231ps` per feature su 8 registri accumulatori
  diversi (ymm1, ymm3-9), nessuno spill: kernel ideale, nessuna catena di dipendenza.
- **Risultato** (1 thread, mediana; la trasposta è inclusa nel tempo):

  | training     | AoS (simd) | SoA     | AoSoA   |
  |--------------|------------|---------|---------|
  | 64 (L2)      | 0.069 s    | 0.029 s | 0.031 s |
  | 2000 (L3)    | 0.59 s     | 0.47 s  | –       |
  | 60000 (DRAM) | 1.75 s     | 2.17 s  | 1.82 s  |

  Trasposta di 188 MB: 0.11-0.14 s (1 thread), 0.07-0.11 s (6 thread). 0 mismatch: la
  somma per campione è in ordine di feature come nello scalare (con FMA).
- **Interpretazione:** in L2 SoA è **2.3× più veloce** di AoS (≈ 50 GFLOP/s a 1
  thread): ipotesi confermata sul kernel. Su MNIST completo SoA puro *perde*: per un
  blocco di 64 campioni legge 784 righe distanti 240 KB → 784 flussi concorrenti, una
  pagina nuova a ogni passo: il prefetcher hardware non li segue e il TLB manca.
  AoSoA (blocchi contigui [784][64] = 200 KB) recupera il pattern sequenziale e torna
  alla pari con AoS: in DRAM tutte e due sono memory-bound, il layout conta poco.
  Conclusione: il vantaggio del layout SoA si vede solo se i dati vengono riusati dalla
  cache → **AoSoA + blocking** è il prossimo passo naturale.

---

## 2026-11-15 — Campagna ufficiale (`run_all.sh`): ATTENZIONE metà non valida
- **Commit:** `c33260f`. Log: `results/raw/run_all_20261115.log`.
- **Problema:** il caricatore è stato staccato durante la campagna (100% in carica
  all'avvio, 71% in scarica alla fine). Lo stesso `seq` k=5 euclidea passa da 19.1 s
  (18:47) a 28.8 s (19:30). Validi: `threads` e `schedule` (18:47-19:04, tempi coerenti
  con le prove del pomeriggio). **Non validi:** `tiles`, `cachefit`, `vec`, `kmetric`
  (spostati in `results/raw/invalid_battery/`, da rifare in carica). Restano valide le
  accuratezze (`accuracy_full_*.csv`), che non dipendono dalla velocità.
- **Lezione di metodologia:** controllare l'alimentazione anche *durante* la campagna
  (es. `system_info.sh` all'inizio e alla fine di ogni esperimento) e ripetere un run
  di riferimento (seq) in fondo per accorgersi della deriva.

## 2026-11-15 — Strong scaling 1-12 thread (valido)
- **Configurazione:** MNIST 60000 × 384 query, k=5, euclidea, `static`,
  `OMP_PROC_BIND=close OMP_PLACES=cores`, 5 run + 1 warm-up, ordine varianti ruotato.
- **Comando:** `python3 scripts/run_benchmarks.py threads --variants omp_naive,omp_simd,omp_soa,omp_aosoa,omp_blocked,omp_blocked4`
- **File risultati:** `results/raw/threads_20261115_184752.csv`; grafico
  `results/plots/speedup_5_euclidean_mnist_60000_384_784.png` (rigenerabile).
- **Risultato** (mediana; seq = 19.08 s = 2.84 GFLOP/s; 54.2 GFLOP totali):

  | variante     | 1 thr  | 6 thr  | 8 thr  | 12 thr | speedup max vs seq | GFLOP/s max |
  |--------------|--------|--------|--------|--------|--------------------|-------------|
  | omp_naive    | 19.03 s| 3.10 s | 2.29 s | 2.74 s | 8.3× (8 thr)       | 23.7        |
  | omp_simd     | 6.62 s | 0.91 s | 0.73 s | 2.88 s | 26.3× (8 thr)      | 74.7        |
  | omp_soa      | 5.61 s | 1.80 s | 1.43 s | 2.90 s | 13.4× (8 thr)      | 38.0        |
  | omp_aosoa    | 4.89 s | 0.95 s | 0.75 s | 3.26 s | 25.3× (8 thr)      | 71.9        |
  | omp_blocked  | 2.05 s | 0.45 s | 0.36 s | 0.27 s | 70.1× (12 thr)     | 199         |
  | omp_blocked4 | 0.89 s | 0.28 s | 0.24 s | 0.17 s | **111× (12 thr)**  | **317**     |

  0 mismatch ovunque. CPU / wall ≈ numero di thread in tutti i casi (anche i core in
  stallo sulla memoria contano come occupati).
- **Interpretazione:**
  1. **Crollo a 12 thread delle varianti senza blocking.** Banda "apparente"
     (384 × 188 MB = 72.2 GB letti) a 12 thread: naive 26.4, simd 25.1, soa 24.9, aosoa
     22.2 GB/s → tutte convergono al **picco DRAM single channel (23.5 GB/s)**. A 8
     thread omp_simd ha banda apparente 99.6 GB/s = 4.2× il picco: i thread scorrono il
     training quasi in sincronia e si condividono le righe in L3. Ipotesi: con 12 thread
     (2 per core) la sincronia si rompe e/o la L3 per thread si dimezza (12 MB / 12) →
     ogni thread rilegge dalla DRAM. Conferma indiretta dallo sweep di scheduling: omp_simd
     con `dynamic,64` (solo 6 chunk → 6 thread attivi) impiega 0.90 s contro 2.75 s di
     `static` a 12 thread: **meno thread = più veloce**.
  2. **Efficienza > 1 di omp_simd (1.2) e omp_naive (1.09).** Per omp_simd: con 1 thread
     la banda è limitata da un singolo core (72.2 GB / 6.62 s = 10.9 GB/s); con più thread
     le righe condivise in L3 aumentano la banda effettiva per thread → superlineare.
     Per omp_naive (compute-bound) la causa non è chiara: domanda aperta.
  3. **Hyperthreading:** aiuta le varianti *latency-bound* in cache: omp_blocked
     6 → 12 thread = 1.65× (un solo accumulatore FMA per thread: il secondo thread del
     core riempie gli slot vuoti della pipeline). Danneggia le varianti memory-bound
     (punto 1). Risposta alla domanda aperta del CLAUDE.md: dipende dal collo di bottiglia.
  4. **omp_blocked4: plateau 4-6 thread, poi di nuovo memory-bound.** Con 1 thread fa
     60.8 GFLOP/s (2.3× omp_blocked, 21× seq): il micro-kernel 4×1 rompe il limite di
     latenza. Ma il traffico DRAM è 24 blocchi × 188 MB = 4.5 GB: a 6 thread 16.4 GB/s
     (70% del picco), a 12 thread 26.4 GB/s (≈ picco). **Roofline:** con QB=16 ogni
     float letto dalla DRAM (4 B) serve 16 query × 3 flop = 12 flop/B → tetto
     23.5 GB/s × 12 = 282 GFLOP/s; misurato 317 a 12 thread (un po' sopra grazie alla
     L3). omp_blocked (199 GFLOP/s) resta sotto il tetto: è compute-bound.
     → per blocked4 serve più riuso (QB maggiore), ma meno blocchi = peggior
     bilanciamento: compromesso granularità/riuso da misurare con lo sweep tiles.

## 2026-11-15 — Scheduling a 12 thread (valido)
- **Comando:** `python3 scripts/run_benchmarks.py schedule --variants omp_naive,omp_simd,omp_blocked,omp_blocked4 --threads 12 --no-seq`
- **File risultati:** `results/raw/schedule_20261115_185923.csv`
- **Risultato** (mediana, s):

  | schedule   | naive | simd  | blocked | blocked4 |
  |------------|-------|-------|---------|----------|
  | static     | 2.66  | 2.75  | 0.279   | 0.170    |
  | static,1   | 2.82  | 2.83  | 0.277   | 0.171    |
  | static,16  | 2.93  | 2.78  | 1.405   | 0.609    |
  | dynamic,1  | 2.70  | 3.09  | 0.279   | 0.172    |
  | dynamic,16 | 2.90  | 2.92  | 1.393   | 0.620    |
  | dynamic,64 | 3.04  | 0.90  | 2.055   | 0.870    |
  | guided     | 2.62  | 2.85  | 0.270   | 0.164    |

- **Interpretazione:** il carico per query è uniforme (stessa quantità di lavoro per
  ogni query), quindi `static` è già bilanciato: per naive e simd tutti gli schedule
  sono entro il rumore (std 0.1-0.37 s), tranne il caso simd/dynamic,64 spiegato sopra.
  Per le varianti a blocchi il chunk conta in **blocchi da 16 query**, non in query:
  con 24 blocchi, chunk 16 → 2 chunk → 2 thread al lavoro (CPU/wall = 1.5, tempo ×5);
  chunk 64 → 1 chunk → 1 thread (tempo = quello a 1 thread). Lezione: la granularità
  dello schedule va ragionata sull'unità di lavoro del loop parallelo.

## 2026-11-15 — Accuratezza sull'intero test set (10000 query, valido)
- **Comando:** fine di `run_all.sh` (omp_blocked4, 12 thread, 1 run, senza validazione:
  la variante ha dato 0 mismatch in tutte le configurazioni validate).
- **File:** `results/raw/accuracy_full_20261115_193504.csv` (tempi NON validi: batteria).

  | k  | euclidea | coseno |
  |----|----------|--------|
  | 1  | 96.91%   | 97.23% |
  | 5  | 96.93%   | 97.37% |
  | 10 | 96.83%   | 97.24% |

  Il coseno è ~0.4 punti migliore: normalizza l'intensità complessiva dell'immagine
  (tratti più spessi/sottili della stessa cifra). k=5 è il migliore per entrambe.

- **Da rifare in carica:** tiles (6 e 12 thread), cachefit, vec, kmetric (tempi).

---

## 2026-11-16 — Chiusura della fase A: rilettura dei soli dati validi
Nessuna misura nuova: solo controllo di integrità, grafici e rilettura critica di
`threads_20261115_184752.csv` e `schedule_20261115_185923.csv` (commit dei binari:
`c33260f`; da allora i sorgenti non sono cambiati, verificato con `git diff`).
I numeri di `PREVISIONI_run_rest.md` sez. 3 **non** compaiono qui: sono previsioni.

- **Integrità (verificata):** campagna terminata (`### done` nel log, nessun exit
  code ≠ 0); 5 run + 1 warm-up per ognuna delle 37 + 28 configurazioni; 0 mismatch
  ovunque; `_system.txt` presente per entrambi i CSV; 384 query = 24 blocchi da 16
  (divisibile per 1, 2, 4, 6, 8, 12); `seq` = 19.08 s (≥ 10 s). `ctest` verde su HEAD.
  **Mancano** (oltre agli esperimenti a batteria da rifare): `sizes` (curve di speedup
  per più dimensioni dell'input: richiesto dalla traccia, non è né in `run_all.sh` né
  in `run_rest.sh`), il confronto top-k e il weak scaling.
- **Grafici:** `plot_results.py` sistemato (solo presentazione): asse log per lo
  speedup (le varianti coprono 1×–110×), barre d'errore = run più veloce / più lento
  attorno alla mediana (prima: ± std, gonfiata da un solo outlier), scheduling in un
  pannello per variante con gli schedule nello stesso ordine (prima le etichette si
  sovrapponevano ed erano illeggibili). I due CSV vanno passati **separatamente**
  allo script: insieme, le righe `static` a 12 thread dei due file verrebbero fuse in
  un'unica configurazione da 10 run. Riepiloghi: `results/plots/summary_threads.csv`,
  `summary_schedule.csv` (rigenerabili, non committati).

### Osservazioni (dati validi)
Riferimenti: 54.2 GFLOP per run; training 188.2 MB; se ogni query rilegge tutto il
training sono 72.3 GB per run.

1. **Due regimi per le varianti senza blocking: "convoglio" e "flussi indipendenti".**
   Velocità di scorrimento *per thread* di `omp_simd` (GB di training letti da ogni
   thread al secondo = 72.3 GB / (p · T_p)):

   | thread | 1 | 2 | 4 | 6 | 8 | 12 |
   |---|---|---|---|---|---|---|
   | GB/s per thread | 10.9 | 13.6 | 13.3 | 13.2 | 12.5 | **2.1** |
   | GB/s apparenti totali | 10.9 | 27.2 | 53.2 | 79.0 | **99.6** | 25.1 |

   Da 2 a 8 thread ogni thread scorre il training alla *stessa* velocità (~13 GB/s)
   qualunque sia il numero di thread, e il totale apparente arriva a 4.2× il picco
   teorico della DRAM (23.5 GB/s, single channel): i thread non leggono 8 copie
   dalla DRAM, leggono **la stessa** riga quasi nello stesso momento — il primo la
   porta in L3, gli altri la trovano lì. A 12 thread il totale di *tutte e quattro*
   le varianti (naive 26.3, simd 25.1, soa 24.9, aosoa 22.1 GB/s) coincide col picco
   DRAM: ogni thread legge per conto suo. Conseguenza per il report: lo "scaling
   quasi ideale" di `omp_simd` fino a 8 thread **non** dice che la banda di memoria
   scala; dice che con `schedule(static)` e query che scorrono lo stesso training
   nello stesso ordine il traffico DRAM non cresce col numero di thread (un blocking
   "nel tempo" involontario). *Ipotesi da verificare* sul perché si rompa a 12:
   vedi `report/review.md` §1.
2. **Controprova nello sweep di scheduling.** `omp_simd`, 12 thread, `dynamic,64`:
   0.90 s contro 2.75–3.09 s di tutti gli altri schedule. 384 query / 64 = 6 chunk →
   lavorano solo 6 thread (CPU/wall = 6.0) e il tempo torna quello a 6 thread
   (0.914 s). Stessa cosa per `omp_naive` (`dynamic,64`: CPU/wall = 5.9, 3.04 s ≈
   3.10 s a 6 thread). Meno thread attivi = stesso tempo o migliore.
3. **Granularità delle varianti a blocchi.** Il loop parallelo è su 24 blocchi da 16
   query, quindi il chunk si conta in blocchi: `static,16`/`dynamic,16` → 2 chunk →
   2 thread al lavoro (CPU/wall = 1.5; `omp_blocked` 1.40 s contro 1.08 s a 2 thread
   veri: i 2 chunk sono sbilanciati, 16 + 8 blocchi); `dynamic,64` → 1 chunk → 1
   thread (2.055 s = tempo a 1 thread, 2.046 s). Non è un effetto dello scheduling ma
   della granularità: per queste varianti lo sweep sensato è con chunk 1–4.
4. **`omp_blocked4`: dopo 4 thread smette di scalare.** Mediane 0.270 s (4 thr) e
   0.275 s (6 thr); ma le 5 misure sono rumorose (4 thr: 0.259–0.351; 6 thr:
   0.231–0.290), quindi "plateau" è corretto solo entro ±15%. Traffico DRAM se ogni
   blocco rilegge il training: 24 × 188 MB = 4.52 GB → 16.7 GB/s a 4 thread, 16.4 a
   6 (≈ 70% del picco teorico). Curioso: `omp_blocked` a 12 thread fa 16.6 GB/s, lo
   stesso valore. *Ipotesi:* ~16.5 GB/s è la banda DRAM realmente sostenibile su
   questa macchina (va misurata con un test STREAM-like); allora `omp_blocked4` è
   memory-bound da 4 thread **e `omp_blocked` a 12 thread potrebbe esserlo anche
   lui** (il diario lo dava per compute-bound). A 12 thread `omp_blocked4` fa
   26.4 GB/s apparenti (sopra il picco): di nuovo righe condivise in L3 (regime 1).
   Lo sweep `tiles` con QB=32 (traffico DRAM dimezzato) discrimina.
5. **Tetto di calcolo.** sub + FMA = 2 µop vettoriali per 8 elementi; con 2 porte FP
   (p0, p1) sono 8 elementi × 3 flop per ciclo = 24 flop/ciclo/core. Con l'ipotesi
   (non misurata!) di ~3.8 GHz sotto carico su 6 core: ≈ 550 GFLOP/s.
   `omp_blocked4` a 12 thread fa 317 GFLOP/s ≈ 58% di questo tetto. Assembly
   controllato (`objdump`, build `c33260f`): `omp_blocked4` ha 4 accumulatori
   `ymm8–11`, nessuno spill, 5 load + 4 `vsubps` + 4 `vfmadd231ps` per iterazione;
   `omp_simd`/`omp_blocked` 1 accumulatore; `omp_soa`/`omp_aosoa` 8 accumulatori.
6. **Accuratezza.** 1-NN euclidea sul test set completo = 96.91%, cioè errore 3.09%:
   coincide con il valore riportato per k-NN L2 su MNIST nella tabella di LeCun.
   Coseno sempre migliore di 0.3–0.4 punti (k=5: 97.37% contro 96.93%).
7. **Rumore.** `omp_simd` a 1 thread: 6.71, 6.77, 6.62, 5.77, 5.24 s nello stesso
   processo — non rumore casuale ma una deriva verso il veloce (−22%). La mediana
   (6.62 s) è il valore "lento": l'efficienza > 1 (1.2) di `omp_simd` dipende da quale
   T(1) si prende (con 5.24 s sarebbe 0.96). `seq`: primo run 20.85 s, poi
   18.9–19.1 (un solo warm-up non basta al primo esperimento della campagna).

- **Prossimo passo:** `report/review.md` (fase B) elenca per ogni conclusione del
  diario le spiegazioni alternative e l'esperimento breve che le discrimina; domani
  `run_rest.sh` + `report/predictions_vs_measured.md`.

## 2026-11-16 — Errata alle voci precedenti (dalla revisione, `report/review.md`)
Le voci sopra restano com'erano scritte; queste correzioni prevalgono.

1. **Voce `omp_blocked`, nota (*) su `omp_simd` a 6 thread con 192 query
   (15.8 GFLOP/s "anomalo").** Non è anomalo: il tetto memory-bound di `omp_simd` è
   23.5 GB/s ÷ 4 B × 3 flop = 17.6 GFLOP/s, quindi 15.8 è il 90% della banda DRAM
   (regime "flussi indipendenti"). Eccezionale è la misura con 100 query
   (38.7 GFLOP/s = 2.2× il tetto): lì i thread si condividono le righe in L3
   (regime "convoglio"). Il rapporto blocked/simd "6.5×" di quella tabella confronta
   quindi `omp_blocked` con `omp_simd` nel suo regime lento: nella campagna valida,
   a 6 thread, il rapporto è 0.914 / 0.449 = 2.0×.
2. **Voce cache-fit: "DRAM ≈ 9.6 GB/s" a 1 thread.** È il limite di *un core*, non
   della DRAM: ~10–12 miss in volo (line fill buffer) × 64 B / ~70 ns ≈ 9–11 GB/s
   (legge di Little). In convoglio ogni thread arriva a ~13 GB/s.
3. **Voce variance probe: "39.9 GFLOP/s, coerente…" e "3 o 6 core danno lo stesso
   tempo".** La campagna ufficiale, 4 minuti dopo, misura la stessa configurazione a
   0.914 s (59.3 GFLOP/s) contro 1.35 s del probe. Differenza non spiegata: la
   conclusione sul binding è **sospesa** finché il probe non viene rifatto nel
   blocco di misure.
4. **Voce `omp_blocked`: "il limite torna a essere il calcolo"; voce strong
   scaling, punto 4: "`omp_blocked` resta sotto il tetto: è compute-bound".** Il
   blocking riduce il traffico DRAM di QB volte, non lo elimina: a 1 thread la
   lettura dalla DRAM vale ≈ 20% del tempo di `omp_blocked` e ≈ 50% di quello di
   `omp_blocked4` (4.52 GB a ~10–11 GB/s ≈ 0.41–0.47 s su 2.05 e 0.89 s). E il
   "tetto" di 282 GFLOP/s è stato superato (317): assumeva che ogni blocco leggesse
   la sua copia dalla DRAM, ma anche le tile vengono condivise in L3.
5. **Voce accuratezza: "k=5 è il migliore".** Le differenze tra k (2–10 immagini su
   10000) sono dentro l'incertezza statistica (±0.17 punti): i tre k sono
   indistinguibili. Il vantaggio del coseno (30–44 immagini) è più solido; la sua
   spiegazione resta un'ipotesi.
6. **Modello a latenza (voce cache-fit):** le frequenze usate (4.5 GHz) non sono mai
   state misurate e i dati a 1 thread non sono coerenti tra loro (`omp_naive` ⇒
   3.8 GHz, `omp_blocked` ⇒ ≥ 4.4 GHz con lo stesso modello). Da rifare con una
   sonda di frequenza (review §1-E).

## 2026-11-16 — Modifiche dopo la revisione (nessuna misura)
Concordate con lo studente: tutte le proposte di `review.md` §6 tranne la 9
(`-ffp-contract=off` sul riferimento: non si fa, si dichiara nel report).

- **Varianti nuove** (una idea ciascuna):
  - `omp_blocked4_pn` — `omp_blocked4` con le norme del coseno calcolate da un
    `parallel for` sulle righe. Idea: togliere la frazione seriale (Amdahl). Ogni
    riga è ancora sommata da un solo thread nello stesso ordine.
  - `omp_simd_stagger` — **diagnostica**, non un'ottimizzazione: `omp_simd` in cui il
    thread t parte con t × `KNN_STAGGER_MS` (default 20) ms di ritardo, per rompere
    di proposito la condivisione delle righe in L3. Sempre `static`. Il ritardo è
    dentro il tempo misurato ((p−1) × 20 ms).
- **Varianti esistenti**, senza cambiarne l'idea: `default(none)` con la lista
  esplicita delle variabili condivise in tutte le regioni parallele;
  `omp_soa`/`omp_aosoa` scrivono il tempo di allocazione + trasposta in
  `knn_prep_seconds` → nuova colonna **`prep_s`** in fondo al CSV (`time_s` continua
  a includerla). `omp_blocked4` è ora un involucro di `blocked4_impl(..., false)`.
- **Il codice macchina dei kernel non è cambiato** (controllo fatto perché i dati
  validi di ieri devono restare confrontabili con le misure di domani): confronto
  `objdump` tra il binario della campagna e quello nuovo, per i corpi paralleli di
  tutte e 6 le varianti. Stesso numero di istruzioni (aosoa: +3 nel prologo), stessa
  identica sequenza di istruzioni `ymm`; le uniche differenze sono gli offset con
  cui si leggono i campi della struttura delle variabili condivise (l'ordine della
  clausola `shared` ne cambia la disposizione) — codice fuori dai loop.
  `knn_sequential` è identico byte per byte.
- **Strumenti:** `tools/hwprobe.cpp` (+ `scripts/hwprobe.sh`): frequenza reale sotto
  carico (catena di `imul` da 3 cicli e di `vfmadd231ps` da 4 cicli, in assembly) e
  banda DRAM in lettura (somma di 1 GB con 4 accumulatori), 1–12 thread.
- **Script:** `run_benchmarks.py` con `--thread-list`, `--schedules`, `--n-trains`,
  `--dim`; `run_rest.sh` con controllo di deriva anche **all'inizio**;
  `run_extra.sh` = gli esperimenti della revisione (≈ 25 min + 15 di `sizes`);
  `system_info.sh` registra alimentazione, piano energetico di Windows, THP, flag e
  commit; `plot_roofline.py`.
- **Test:** `smoke_test.sh` prova anche TB/QB dispari per tutte le varianti a
  blocchi (prima solo a mano e solo per `omp_blocked`) e `k > n_train`. Passa con 0
  mismatch (9 varianti). Validazione sull'intero test set: vedi `review.md` §4.1.
- **Modello:** `report/perf_model.md` (fase C) con le previsioni P-1…P-6 registrate
  prima delle misure di `run_extra.sh`.

---

## 2026-11-16 — Requisiti scoperti della traccia: top-k, weak scaling, sincronizzazione
Sessione a batteria: **codice e validazione soltanto, nessun tempo**. Ogni voce
registra l ipotesi e una previsione numerica; le misure sono in `scripts/run_new.sh`.

### Top-k: heap contro insertion sort (`omp_blocked4_heap`)
- **Cosa:** nuova politica di selezione `TopKHeap` (`include/topk_heap.hpp`): heap
  binario di massimo sui k migliori. `blocked4_impl` è ora un template sulla
  politica: `omp_blocked4` = `blocked4_impl<TopK>`, `omp_blocked4_heap` =
  `blocked4_impl<TopKHeap>`. Una idea: la struttura dati della selezione.
- **Perché un max-heap per i k più piccoli:** serve solo sapere chi è il peggiore
  dei k tenuti, ed è la radice. Rifiuto di un candidato: 1 confronto in entrambe
  le politiche. Inserimento: O(k) spostamenti (insertion sort) contro O(log k).
- **Stesso risultato, pareggi compresi:** l heap ordina per (distanza, ordine di
  arrivo); un nuovo arrivato è l ultimo, quindi entra solo se la distanza è
  *strettamente* minore della radice (stesso test `d < dist[k-1]` di `TopK`).
  Il voto riordina i k elementi e vota come `TopK`.
- **Validazione:** 0 mismatch in `ctest` (ora anche con k = 32 e 128 per tutte
  le varianti, su dati con molti quasi-pareggi) e su MNIST 2000×64.
- **Il codice di `omp_blocked4` non è cambiato:** stessa sequenza di 6049
  istruzioni nel corpo parallelo prima e dopo il template (confronto `objdump`
  dei mnemonici: differiscono solo gli indirizzi). I dati validi restano
  confrontabili.
- **Ipotesi (T1): risultato nullo.** Con il training in ordine casuale il j-esimo
  campione entra tra i k migliori con probabilità k/j: gli inserimenti attesi per
  query sono k·(1 + ln(n/k)) — 47 per k = 5, **915 per k = 128** su 60000
  distanze. Anche a k = 128, 915 inserimenti × ~64 spostamenti = 6·10⁴ operazioni
  contro 4.7·10⁷ elementi del calcolo delle distanze: 0.1%. **Previsione: heap e
  insertion sort indistinguibili (entro il rumore) per ogni k da 1 a 128.**
  L heap vincerebbe solo con dati ordinati per distanza *decrescente* (ogni
  candidato entra): caso avverso che qui non esiste. È la risposta alla richiesta
  della traccia: il confronto va fatto, e il risultato atteso è che la scelta non
  conta — *perché* il top-k è una frazione trascurabile del lavoro.
- **Misura:** `run_benchmarks.py topk` (k = 1, 5, 10, 32, 64, 128).

### Weak scaling (`run_benchmarks.py weak`, `plot_weak.py`)
- **Cosa:** il problema cresce con i thread, lavoro per thread costante. Due modi:
  `--weak-mode test` (default): 96·p query sul training completo; `train`:
  query fisse, training replicato p volte (cresce anche la memoria: p × 188 MB).
  Metriche: efficienza debole T(1)/T(p) (ideale 1), speedup scalato p·T(1)/T(p)
  (ideale p), con la retta di Gustafson S(p) = p − s(p − 1) adattata su 2–6 thread.
- **Ipotesi (W1):** nel modo `test` il traffico DRAM *totale* cresce con p (ogni
  thread rilegge il training per i propri blocchi), quindi le varianti a blocchi
  restano efficienti finché p·(banda per thread) < banda DRAM (~16.5 GB/s [I]).
  **Previsione:** `omp_blocked4` efficienza debole ≥ 0.9 fino a 2 thread, in calo
  da 4 (satura la DRAM, come nello strong scaling); `omp_blocked` ≥ 0.9 fino a 6
  (kernel più lento, banda richiesta più bassa); `omp_simd` ≈ 1 fino a 8 *se* il
  convoglio regge (i thread scorrono lo stesso training in sincronia), crollo a 12.
  Nel modo `train` il convoglio non può aiutare di più (ogni thread scorre comunque
  tutto il training, p volte più lungo): efficienza attesa simile, tempi p volte
  maggiori — è il caso in cui Gustafson e strong scaling raccontano la stessa
  storia di memoria.

### Strategie di sincronizzazione (`tools/sync_strategies.cpp`)
- **Perché un esperimento a parte:** nel k-NN non c è nessuna sincronizzazione
  (ogni query scrive il suo `out[i]`). L unico stato condiviso che ha senso è la
  **matrice di confusione** della valutazione, un istogramma 10×10: troppo
  piccola per pesare nel k-NN, quindi la si isola e la si calcola su 2·10⁷ coppie
  (etichetta, predizione) sintetiche con il 97% sulla diagonale.
- **Sei strategie**, tutte verificate uguali al calcolo seriale (test `ctest`
  `sync_strategies`): `serial`; `critical` (un lock globale); `atomic`
  (`lock add`, niente lock ma la linea di cache della cella rimbalza: condivisione
  *vera*); `reduction(+: m[:100])` (copia privata per thread, somma finale);
  `packed` (contatori privati ma interlacciati `[cella][thread]`: nessuna race,
  ma thread diversi scrivono la stessa linea = **false sharing**); `padded`
  (una matrice per thread allineata a 64 B con `alignas`).
  `packed` e `padded` eseguono le stesse istruzioni: cambia solo il layout.
- **Previsioni (prima della misura), in milioni di aggiornamenti al secondo:**
  - S1: `critical` è la più lenta e **peggiora** con i thread (contesa sul lock):
    < 20 M/s a 1 thread, < 5 M/s a 6;
  - S2: `atomic` a 1 thread costa ~20 cicli per aggiornamento (~200 M/s) e
    **non scala**: con 10 celle calde ogni incremento è un trasferimento di linea
    tra core; a 6 thread è più lenta che a 1;
  - S3: `reduction` ≈ `padded` e scalano quasi linearmente fino a 6 thread
    (carico: 2 byte letti per aggiornamento, niente scritture condivise);
  - S4: `packed` a 1 thread = `padded`; a 6 thread è **5–20 volte più lenta** di
    `padded`, e più lenta di `packed` a 1 thread: è il costo del false sharing.
- **Misura:** `bash scripts/sync_strategies.sh` (1, 2, 4, 6, 12 thread, 5 run).

## 2026-11-16 — Varianti avanzate: intrinsics AVX2, micro-kernel MR×NR, early abandoning
Sessione a batteria: codice, validazione, assembly e **contatori**; nessun tempo.
Tutte le varianti sono in `src/knn_avx2.cpp` (compilato con `-mavx2 -mfma`), tutte a
**0 mismatch** in `ctest` (17 varianti; TB/QB dispari; k fino a 128) e su MNIST
60000 × 64 query con entrambe le metriche. Misure: `scripts/run_new.sh` N4–N7.
Parametri del modello: f ≈ 4.2 GHz **[I]**, 2 porte FP, FMA con latenza 4 cicli.

Proprietà comune (risolve il punto sottile di `review.md` §4.3): tutti i kernel di
questo file calcolano la distanza di una coppia con la **stessa sequenza di
operazioni float** (un accumulatore a 8 corsie aggiornato con FMA esplicite, somma
orizzontale `hsum256` a ordine fisso, coda scalare con `std::fma`). Quindi una
coppia riceve gli stessi bit da ogni kernel MR×NR e dal kernel con early
abandoning, qualunque siano TB, QB e i resti. Fa eccezione, per costruzione,
`omp_blocked_avx2u4` (32 somme parziali).

### 1. Intrinsics scritti a mano contro `omp simd` (`omp_blocked_avx2`)
- **Idea:** stesso algoritmo di `omp_blocked`; cambia *chi* vettorizza. Collega la
  lezione sugli intrinsics SSE: `_mm256_loadu_ps` / `_mm256_sub_ps` /
  `_mm256_fmadd_ps` sono la versione a 256 bit (8 float) di `_mm_*_ps` (4 float).
- **Assembly (verificato con `tools/asm_loops.py`):** il loop di `omp_blocked`
  (`omp simd`) e quello scritto a mano sono lo stesso: 6 istruzioni, 1 `vsubps`,
  1 `vfmadd231ps` su 1 accumulatore, 2 operandi in memoria. Idem `omp_blocked4`
  e `omp_mk4x1`: 16 istruzioni, 4 `vsubps`, 4 `vfmadd231ps`, 5 operandi in memoria.
- **Previsione A:** `omp_blocked_avx2` = `omp_blocked` e `omp_mk4x1` =
  `omp_blocked4` entro ±3%. Messaggio: per questo kernel il compilatore con la
  direttiva giusta fa quanto gli intrinsics; gli intrinsics servono dove la
  direttiva non arriva (punti 2, 3, 4).
- **Cosa `omp simd` non sa fare (scoperto provando):** la generalizzazione naturale
  di `omp_blocked4`, `#pragma omp simd reduction(+: s)` con `s[MR*NR]` array, viene
  vettorizzata malissimo da GCC 15: accumulatori sullo stack, `vextractf128` /
  `vinsertf128` a ogni iterazione. Per questo il micro-kernel generico usa gli
  intrinsics.

### 2. Quattro accumulatori sulla stessa distanza (`omp_blocked_avx2u4`)
- **Idea:** spezzare la catena di dipendenza *senza* cambiare il flusso dei dati:
  le feature di una coppia vanno a turno su 4 accumulatori. `omp simd` dà un solo
  accumulatore vettoriale: il loop resta legato alla latenza della FMA.
- **Assembly:** 16 istruzioni, 4 FMA su 4 accumulatori, **8** operandi in memoria
  (2 load per FMA; `omp_blocked4` ne ha 1.25: lì la riga di training è riusata).
- **Previsione B (L2, 1 thread):** 8 µop FP ogni 32 elementi = 4 cicli sulle 2
  porte, e 8 load ogni 4 cicli = le 2 porte di load **sature**: 2.5–3.5× più
  veloce di `omp_blocked_avx2` (non 4×: il kernel a 1 accumulatore sovrappone già
  distanze consecutive grazie all esecuzione fuori ordine, review §1-E), e
  0.8–1.0× `omp_blocked4`. Su MNIST da 4 thread: uguale a `omp_blocked4` (DRAM).
  Insegna che latenza nascosta e riuso dei dati sono due cose distinte:
  `omp_blocked4` le fa entrambe, questa variante solo la prima.

### 3. Micro-kernel a registri MR×NR (`omp_mk4x1`, `omp_mk4x2`, `omp_mk6x2`, `omp_mk2x4`)
- **Idea (fase D1 del brief):** MR query × NR righe di training per passata,
  MR·NR accumulatori indipendenti; ogni vettore di training riusato MR volte, ogni
  vettore di query NR volte. MR e NR sono parametri template di `kernel_mrnr` e
  `blocked_mrnr` (blocking di cache esterno uguale a `omp_blocked`). Query di
  resto: kernel 2×NR, poi 1×NR; righe di resto della tile: kernel MR×1.
- **Budget dei registri (16 ymm):** MR·NR accumulatori + NR vettori di training
  + 1–2 temporanei. 4×2: 8 + 2 + 2 = 12; 2×4: 8 + 4 + 2 = 14; 6×2: 12 + 2 + 2 = 16.
- **Assembly:**

  | kernel | istruzioni | FMA / accumulatori | load per FMA | accessi allo stack |
  |---|---|---|---|---|
  | 4×1 | 16 | 4 / 4 | 1.25 | 0 |
  | 4×2 | 25 | 8 / 8 | 0.75 | 0 |
  | 2×4 | 25 | 8 / 8 | 0.75 | 0 |
  | 6×2 | 36–37 | 12 / 12 | 0.67 | **2–4 per iterazione** |

  (Il numero esatto dipende dal contesto di compilazione: nel binario finale il
  kernel euclideo ha 4 accessi allo stack per iterazione = 2 accumulatori in
  memoria, quello del prodotto scalare 2 = 1 accumulatore. Al limite dei
  registri l allocazione è fragile: un altro motivo per preferire il 4×2.)
  Nel 6×2 **un accumulatore vive sullo stack**: `vfmadd213ps 0x200(%rsp)` seguito
  da `vmovaps %ymm3, 0x200(%rsp)`. Il conto dei registri lo prevedeva (16 su 16:
  non resta nulla per il temporaneo). Quella catena passa dalla memoria a ogni
  iterazione: FMA (4 cicli) + store-to-load forwarding (~5 cicli) ≈ 9–10 cicli.
- **Previsione C (euclidea): nessun guadagno oltre il 4×1.** Per elemento servono
  2 µop FP (`vsubps` + FMA) e le porte FP sono 2: 1 coppia-vettore per ciclo =
  24 flop/ciclo *qualunque* siano MR e NR. Il 4×1 è già bilanciato (4 catene ×
  latenza 4 = 4 cicli = 8 µop / 2 porte): più accumulatori non aggiungono porte.
  Attesi 4×2, 2×4, 6×2 entro ±10% dal 4×1 in L2; identici su MNIST (limite DRAM).
- **Previsione D (coseno): qui il guadagno c è.** Il prodotto scalare ha 1 µop FP
  per elemento (FMA con operando in memoria): il 4×1 ha 4 FMA per iterazione ed è
  limitato dalla *latenza* (4 cicli → 1 coppia-vettore/ciclo, metà delle porte
  inutilizzate). Con 8 accumulatori (4×2, 2×4) le 8 FMA riempiono le 2 porte in
  4 cicli: **fino a 2× il 4×1 in L2**. Il 6×2 ha 12 FMA = 6 cicli sulle porte, ma
  la catena sullo stack ne chiede 9–10: **più lento del 4×2** (0.6–0.7×), più
  veloce del 4×1 (~1.25×). Sarebbe la prova che lo spill conta: stesso sorgente,
  due registri in meno.
- **L1 (review §3):** il kernel tiene in L1 (MR + NR) righe × 3136 B: 4×2 = 18.8 KB,
  6×2 = 25 KB su 32 KB; con 2 hyperthread per core 6×2 esce dalla L1 (50 KB).
- **Legame col prossimo passo:** il micro-kernel del prodotto scalare è già il
  cuore di un GEMM: la formulazione |a|² + |b|² − 2a·b userebbe proprio il kernel
  che la previsione D indica come il più veloce (1 µop per elemento invece di 2).

### 4. Early abandoning esatto (`omp_blocked_ea`)
- **Idea:** per l euclidea, interrompere il calcolo di una distanza appena la
  somma parziale raggiunge la distanza del k-esimo vicino corrente della query:
  quel candidato non può più entrare nel top-k. Controllo ogni `KNN_EA_STEP`
  feature (default 64 = 8 iterazioni AVX2): un controllo per elemento ucciderebbe
  la vettorizzazione. Base: `omp_blocked_avx2` (stesso kernel 1×1, stesso blocking).
- **Dimostrazione che è esatto (non approssimato).** Tesi: se il kernel abbandona,
  `TopK` avrebbe rifiutato la distanza completa D (rifiuta quando `!(D < bound)`).
  1. *Ogni corsia dell accumulatore è non decrescente in float.* L aggiornamento è
     acc′ = fl(e·e + acc) (una FMA: e·e calcolato esatto, un solo arrotondamento).
     e·e ≥ 0, quindi il valore reale e·e + acc ≥ acc. L arrotondamento IEEE è
     monotono (x ≥ y ⇒ fl(x) ≥ fl(y)) e acc è rappresentabile (fl(acc) = acc):
     fl(e·e + acc) ≥ acc.
  2. *`hsum256` è monotona in ogni corsia.* È una sequenza fissa di somme a due
     operandi di valori ≥ 0; per la stessa monotonia dell arrotondamento,
     aumentare un operando non può diminuire il risultato di una somma, e per
     induzione nemmeno il risultato finale. Quindi hsum(acc ora) ≤ hsum(acc alla fine).
  3. *La coda scalare* (d non multiplo di 8) aggiunge termini ≥ 0 con la stessa FMA.
  Dunque parziale ≤ D. Se parziale ≥ bound allora D ≥ bound, cioè `!(D < bound)`:
  rifiutato. ∎ Con la sottrazione al posto della somma (prodotto scalare, termini
  di segno qualsiasi) il passo 1 cade: per questo il coseno non viene abbandonato.
  Nota: la monotonia vale **in float**, non solo in aritmetica esatta — è questo
  che rende il taglio sicuro senza margini di tolleranza.
- **Conseguenza verificabile:** quando non abbandona, il kernel restituisce gli
  stessi bit di `omp_blocked_avx2`; quindi le due varianti devono dare le stesse
  predizioni per *ogni* valore del passo. Verificato: 0 mismatch con
  `KNN_EA_STEP` = 8, 16, 50 (arrotondato a 48), 64, 4096 (mai un controllo utile)
  in `ctest`, e su MNIST 60000 × 64 per k = 1, 5, 10 × passo 16…256.
- **Contatore (non è un tempo): frazione delle feature effettivamente valutate**,
  MNIST 60000 × 64 query (`KNN_VERBOSE=1`). Stima a priori, scritta prima di
  guardare: 50–65% (le prime ~100 feature di MNIST sono bordi neri e non
  contribuiscono). Misurato:

  | passo | 16 | 32 | 64 | 128 | 256 |
  |---|---|---|---|---|---|
  | k = 1 | 33.7% | 34.7% | 36.7% | 41.1% | 48.5% |
  | k = 5 | 37.9% | 38.9% | **40.9%** | 45.3% | 54.3% |
  | k = 10 | 40.0% | 41.0% | 43.0% | 47.4% | 56.8% |

  Meglio della stima: si salta circa il 60% del lavoro. Cresce con k (soglia più
  larga) e col passo (si esce più tardi).
- **Legame con la branch prediction.** Il loop interno ha due salti. Quello di
  fine loop è perfettamente prevedibile. Quello di abbandono dipende dai dati: per
  ogni candidato è "non preso" per alcuni controlli e poi "preso" una volta, in un
  punto che il predittore non può conoscere → **circa un salto sbagliato per
  candidato** (~15–20 cicli per svuotare la pipeline, più il lavoro speculativo
  buttato: il processore ha già iniziato le iterazioni successive della stessa
  riga). In più il confronto dipende dalla catena delle FMA, quindi il salto si
  risolve tardi. Conto: una distanza completa costa ~370 cicli (89 ns: da
  `omp_blocked` a 1 thread, 26.5 GFLOP/s); con il 41% diventano ~150 + ~20 di
  salto sbagliato + ~15 di controlli (5 × `hsum256`). Il salto sbagliato vale
  ~10% del tempo risparmiato: l early abandoning conviene perché ogni salto
  sbagliato fa risparmiare ~58 iterazioni, non perché sia gratis. Con un passo
  piccolo i controlli aumentano e i salti restano uno per candidato; con una
  distanza corta (poche feature) il bilancio si rovescerebbe. In WSL2 non ci sono
  contatori hardware (`perf`): i salti sbagliati restano dedotti dal modello.
- **Previsione E:** a 1 thread su MNIST `omp_blocked_ea` è **1.7–2.2× più veloce
  di `omp_blocked_avx2`** e circa pari a `omp_blocked4` (che fa tutto il lavoro,
  ma 2.3× più in fretta); passo ottimo 32–64; passo 16 più lento di 64 nonostante
  valuti meno feature (38% contro 41%: i controlli raddoppiano due volte); passo
  4096 = `omp_blocked_avx2` + qualche %. A 12 thread guadagno ≥ 1.3×.
- **Perché non si compone col micro-kernel:** l abbandono è una decisione *per
  coppia*; il micro-kernel 4×1 fa avanzare 4 coppie insieme. Due modi di andare
  più veloci — fare meno lavoro (algoritmo) o fare lo stesso lavoro più in fretta
  (architettura) — che qui si escludono. È una buona domanda d esame.
- **Limiti:** l efficacia dipende dai dati (su dati sintetici isotropi le somme
  parziali crescono tutte uguali e si abbandona tardi) e dall ordine del training
  (la soglia si stringe man mano che arrivano vicini buoni).

## 2026-11-16 — Formulazione matriciale |q|² + |t|² − 2 q·t e analisi numerica
Fase D2 del brief (opzionale nella traccia). Codice, validazione e **contatori**;
nessun tempo (misure: `run_new.sh` N8).

- **Varianti** (in `src/knn_avx2.cpp`, sul micro-kernel 4×2 del prodotto scalare):
  - `omp_gemm` — la formula in float così com è. Norme al quadrato precalcolate
    (un `parallel for` sulle righe, stesso kernel del prodotto), lavoro per coppia
    = un prodotto scalare = un blocco del prodotto di matrici Q·Tᵀ.
  - `omp_gemm_exact` — la formula come **filtro**: un candidato viene ricalcolato
    con il kernel diretto 1×1 (`omp_blocked_avx2`) solo se
    `approx − margine < distanza del k-esimo vicino`; gli altri sono rifiutati.
    Il `TopK` riceve solo distanze *esatte*.
- **Perché dovrebbe essere più veloce (previsione F):** l euclidea diretta costa 2
  µop FP per elemento (`vsubps` + FMA) → al massimo 8 elementi/ciclo con 2 porte.
  Il prodotto scalare ne costa 1 → **16 elementi/ciclo**: in L2 a 1 thread
  `omp_gemm` dovrebbe arrivare fino a **2× `omp_mk4x1`/`omp_blocked4`** (stessa
  previsione D del micro-kernel col coseno, senza la divisione finale). Su MNIST
  da 4 thread in su: uguale agli altri (limite DRAM). `omp_gemm_exact` = `omp_gemm`
  + ~0.1% di ricalcoli da ~370 cicli = +0.3 cicli su ~6 per coppia: **≤ 10% più
  lento di `omp_gemm`**.

### Il problema numerico: cancellazione
D = |q|² + |t|² − 2q·t sottrae numeri grandi e quasi uguali quando q e t sono
vicini. Ognuno dei tre termini ha un errore *relativo* piccolo (≈ n·u, con
n ≈ d/8 passi di somma e u = 2⁻²⁴), ma il risultato eredita un errore **assoluto**
proporzionale alla *scala* S = |q|² + |t|², non a D:

    |D̃ − D| ≲ c · S        contro        |D_diretta − D| ≲ c · D

L errore relativo sulla distanza è amplificato di S/D: enorme per i vicini (D
piccola), cioè proprio per i candidati che contano.

**Misure (contatori; MNIST 60000 × 64 query, k = 5; `KNN_GEMM_STATS=1` confronta
ogni coppia con il kernel diretto; riproducibile con `scripts/gemm_numerics.sh`).**
Per variare S a parità di distanze si trasla tutto il dataset di una costante
(`--offset X`): le distanze euclidee non cambiano in aritmetica esatta — la
formula diretta è invariante per traslazione, quella matriciale no.

| offset | errore assoluto max | errore max / S | distanze negative | mismatch `omp_gemm` | mismatch `omp_gemm_exact` | ricalcolate da `_exact` |
|---|---|---|---|---|---|---|
| 0 (MNIST) | 1.2·10⁻⁴ | 3.3·10⁻⁷ | 0 | 0 | 0 | 0.09% |
| 1 | 7.3·10⁻⁴ | 3.3·10⁻⁷ | 0 | 0 | 0 | 0.09% |
| 4 | 0.012 | 4.5·10⁻⁷ | 0 | 0 | 0 | 0.10% |
| 16 | 0.19 | 4.5·10⁻⁷ | 0 | 0 | 0 | 0.53% |
| 64 | 3.0 | 4.7·10⁻⁷ | 0 | 0 | 0 | 99.9% |
| 256 | 28.8 | 2.8·10⁻⁷ | 13 | **2 su 64** | 0 | 100% |
| 1024 | 740 | 4.5·10⁻⁷ | **1 058 561 (28%)** | **49 su 64** (accuratezza 22%) | 0 | 100% |

Lettura:
1. **errore/S è costante** (3–5·10⁻⁷ ≈ 5–8 u) mentre l errore assoluto cresce di
   6 ordini di grandezza: è la firma della cancellazione. Il modello "errore ∝ S"
   regge su tutto l intervallo.
2. **Su MNIST così com è la formula è innocua**: 0 mismatch su 384 query (k = 1, 5,
   10) e nessuna distanza negativa. I pixel stanno in [0, 1], S ≈ 200, errore
   ~10⁻⁴ contro distanze dei vicini ~10 e differenze tra candidati ≫ 10⁻⁴.
   Da confermare sull intero test set (in coda: non si può oggi).
3. **Non è una proprietà della formula, è una proprietà dei dati**: basta una
   traslazione (dati non centrati, feature con una media grande) e il risultato
   degenera — distanze negative (impossibili per un quadrato), poi vicini
   scambiati, poi rumore puro (22% di accuratezza). La versione diretta sugli
   stessi dati traslati resta a 0 mismatch.
4. **Rimedi, dal più debole al più forte:**
   - *clamp a 0*: toglie i negativi (serve se poi si fa la radice) ma non
     ripristina l ordine dei vicini: cosmetico per il k-NN;
   - *centrare i dati* (sottrarre la media delle feature): riduce S, e la distanza
     non cambia. È il rimedio "gratis": sposta il problema da offset 1024 a 0;
   - *norme e prodotti in double*: riduce c di 2²⁹, ma dimezza gli elementi per
     registro (4 double invece di 8 float): si perde il vantaggio di velocità;
   - **ricalcolo esatto dei candidati** (`omp_gemm_exact`): la formula veloce
     decide solo chi *scartare*, con un margine di sicurezza; chi può entrare nel
     top-k viene ricalcolato col kernel diretto. Esatto per costruzione finché il
     margine è una vera maggiorazione dell errore: margine = 4·n·u·S = 2.57·10⁻⁵·S
     (n = d/8 + 10), cioè **77 volte** l errore massimo osservato (3.3·10⁻⁷·S).
     Il margine è argomentato (analisi dell errore delle somme) e verificato
     empiricamente, non dimostrato all ultima costante: dichiararlo.
     Costo: 0 mismatch in tutte le righe della tabella; e quando i dati sono
     ostili (offset ≥ 64) ricalcola tutto, cioè **degrada alla variante esatta
     invece di sbagliare**.
5. **Controprova sul top-k:** `omp_gemm_exact` ricalcola 12.3 candidati per query
   con k = 1, 53.6 con k = 5, 98.1 con k = 10 (384 query). La stima analitica degli
   inserimenti nel top-k, k·(1 + ln(n/k)), dà 12.0, 52.0, 97.0. Il contatore
   conferma il conto usato per prevedere che heap e insertion sort sono
   indistinguibili (previsione T1): il top-k riceve ~50 inserimenti su 60000
   candidati.
- **Legame col corso:** stesso schema dell early abandoning — un test economico che
  scarta quasi tutto, con una garanzia che rende esatto il risultato (lì la
  monotonia delle somme parziali, qui una maggiorazione dell errore). E stesso
  micro-kernel del GEMM: la formulazione matriciale è il ponte verso BLAS.
