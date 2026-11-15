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
