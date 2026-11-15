# Revisione critica "da esaminatore" — 2026-11-16

Stato esaminato: commit `430e072` (sorgenti identici a `c33260f`, quello della
campagna). Fonti: tutto `src/`, `include/`, `scripts/`, `tests/`, `report/notes.md`,
i CSV **validi** (`threads_20261115_184752`, `schedule_20261115_185923`,
`accuracy_full_20261115_193504`, `variance_20261115_184437`) e i preliminari.
I dati a batteria e i numeri di `PREVISIONI_run_rest.md` §3 non sono usati come prove.

Verifiche fatte per questa revisione (nessuna è una misura di prestazioni):
`ctest` (verde), `objdump` dei kernel di tutte le varianti, build con
ThreadSanitizer, validazione sull'intero test set (`scripts/validate_full.sh`).

Legenda: **[S]** supportata dai dati · **[P]** plausibile ma con alternative non
escluse · **[N]** non supportata / contraddetta · ⏱ durata stimata dell'esperimento
che discrimina (tutti da fare nel blocco di misure, in carica).

---

## 0. In una pagina: le sei cose che un esaminatore attaccherebbe

1. **Lo scaling "quasi ideale" di `omp_simd`/`omp_naive` fino a 8 thread è in gran
   parte un effetto di condivisione in L3**, non di banda di memoria che scala
   (§1-A). Va presentato così, altrimenti il crollo a 12 thread sembra inspiegabile.
2. **Il perché del crollo a 12 thread non è dimostrato**: il diario propone una
   causa, ce ne sono almeno tre compatibili con i dati (§1-B).
3. **La stessa configurazione misurata due volte a 4 minuti di distanza dà 1.35 s e
   0.91 s** (`omp_simd`, 6 thread, 384 query): la conclusione tratta dalla prima
   misura ("3 o 6 core è uguale") non regge finché non si capisce la differenza (§1-C).
4. **Nessuna frequenza di clock è mai stata misurata**: tutti i confronti
   "modello a latenza vs misurato" e il tetto di 550 GFLOP/s poggiano su 3.8–4.5 GHz
   ipotizzati; i dati a 1 thread danno indicazioni *incoerenti* tra loro (§1-E).
5. **`omp_blocked4` è per metà memory-bound già a 1 thread** (§1-F): il diario lo
   descrive come variante "tornata compute-bound" che ridiventa memory-bound solo da
   4 thread.
6. **Mancano pezzi richiesti dalla traccia**: curve per più dimensioni dell'input
   (`sizes`), tempi di k × metrica, vec/novec ufficiale, confronto con Amdahl (§5).

Le cose solide, da valorizzare: validazione sistematica contro `seq`, metriche
speedup/scalabilità/efficienza ben distinte, binding corretto, l'incidente della
batteria documentato invece che nascosto, previsioni registrate prima della misura.

---

## 1. Affermazioni del diario: sono supportate?

### A. "I thread si condividono le righe in L3" — **[S]**, ma la conseguenza va tratta
Aritmetica inattaccabile: a 8 thread `omp_simd` consuma 72.3 GB di training in
0.725 s = 99.6 GB/s, 4.2× il picco teorico della DRAM (23.5 GB/s). Non può venire
dalla DRAM. Dato nuovo (da `notes.md`, voce del 16/11): la velocità di scorrimento
*per thread* è costante da 2 a 8 thread (13.6 / 13.3 / 13.2 / 12.5 GB/s).

Conseguenza che nel diario manca: **la curva di strong scaling di `omp_simd` non
misura la scalabilità della memoria**. Misura il fatto che p thread leggono un solo
flusso. È un risultato interessante (un cache blocking "nel tempo" ottenuto per caso
da `schedule(static)`), ma va dichiarato: "efficienza 1.2" detta senza questa
spiegazione è il primo punto su cui un esaminatore si ferma.

### B. "A 12 thread la sincronia si rompe e/o la L3 per thread si dimezza" — **[P]**
I dati dicono *che cosa* succede (a 12 thread tutte e quattro le varianti senza
blocking convergono a 22–26 GB/s = picco DRAM: flussi indipendenti), non *perché*.
Ipotesi compatibili:

| | ipotesi | cosa prevede |
|---|---|---|
| H1 | **Convoglio metastabile.** Un thread rimasto indietro legge da L3 (più veloce della DRAM) e raggiunge il gruppo: il convoglio si auto-stabilizza. Ma se un thread perde più della "finestra" della L3 (12 MB ≈ 3800 righe ≈ 1 ms a 13 GB/s) non rientra più, apre un secondo flusso DRAM e rallenta tutti. Con 12 thread non resta nessun thread hardware libero per Windows/Hyper-V: ogni interruzione espelle un thread. | la rottura è casuale e più frequente al crescere dei thread: 9, 10, 11 thread → tempi bimodali, non un gradino netto |
| H2 | **Nessun core "solista".** A 8 thread 4 core hanno un solo thread (fanno da apripista con il prefetcher pulito); a 12 ogni core ha due thread che leggono le stesse pagine a offset diversi. | gradino netto tra 10 thread (2 solisti) e 12 (nessuno) |
| H3 | **Capacità della L3.** 12 flussi + 12 query non stanno più nella finestra. | dipende da p in modo regolare; insensibile a perturbazioni |

A favore di H1 c'è già un indizio nei dati: gli outlier lenti delle prove a 6 thread
hanno *tutti* la banda del regime "flussi indipendenti" — 4.30 s nel variance probe
(16.8 GB/s), 1.10 s con 100 query (17 GB/s), 1.717 s con 192 query (21 GB/s). Non
sono rumore qualsiasi: sono run in cui il convoglio si è rotto.

- **Esperimento 1** ⏱ ~3 min: `omp_simd` a 7, 9, 10, 11 thread (oggi `THREADS` salta
  da 8 a 12). Gradino netto → H2; degrado progressivo e bimodale → H1.
- **Esperimento 2** ⏱ ~1 min: variante diagnostica `omp_simd_stagger` (identica a
  `omp_simd`, ma il thread t aspetta t·20 ms prima di iniziare: 20 ms × 13 GB/s =
  260 MB ≫ finestra L3). Stessi risultati numerici per costruzione. Se a 6 thread il
  tempo passa da 0.91 s a ~3.5–4 s, la condivisione è *dimostrata* con un intervento
  diretto invece che dedotta. È l'esperimento che renderebbe il punto A inattaccabile.

### C. "omp_simd è così memory-bound che 3 o 6 core danno lo stesso tempo" — **[N] per ora**
Viene dal variance probe (18:44): mediane 1.32–1.37 s in tutte e 5 le configurazioni
di binding. Ma la campagna ufficiale, 4 minuti dopo, per la *stessa* configurazione
(`close/cores`, 6 thread, 384 query, `static`) dà **0.914 s** (5 run: 0.90–1.07).
Il probe è il 48% più lento, in modo sistematico (50 run). Il diario dice "39.9
GFLOP/s, coerente con i 38.7 misurati con 100 query": è coerente con la prova
*preliminare*, ma è incoerente con la misura ufficiale (59.3 GFLOP/s).

Spiegazioni possibili: qualcosa girava in parallelo (la cartella `build-novec` ha
data 18:47: una compilazione durante il probe?); oppure il probe era in un regime di
convoglio parziale. Finché non si sa, **non si può concludere nulla sul binding**.
- **Esperimento 3** ⏱ ~4 min: rifare `variance_probe.sh` dentro il blocco di misure,
  subito dopo un punto di controllo `omp_simd` 6 thread via `run_benchmarks.py`.

### D. "omp_naive è compute-bound (GFLOP/s costanti)" — **[S] fino a 8 thread, [N] a 12**
Per thread: 2.85 / 3.11 / 2.99 / 2.91 / 2.96 GFLOP/s da 1 a 8 thread, poi **1.65** a
12. A 12 thread `omp_naive` legge 26.3 GB/s: è memory-bound come le altre. E a 8
thread ne legge 31.6 (sopra il picco): anche `omp_naive` sta in piedi grazie alla
condivisione in L3. Nel report: "compute-bound finché i thread condividono il flusso".

L'efficienza > 1 (1.09 a 2 thread) resta **senza spiegazione**, come il diario
ammette. Candidati: (a) con 1 thread ogni riga arriva dalla DRAM e il prefetcher si
ferma a ogni confine di pagina; con 2 thread metà delle righe è già in L3; (b) il
thread 0 gira sul core che riceve le interruzioni della VM; (c) deriva termica (il
punto a 1 thread è il primo della campagna, dopo 2 minuti di `seq`).
- **Esperimento 4** ⏱ compreso in `cachefit` se si aggiunge il punto a 2 thread:
  `omp_naive` con training in L2 (64 righe) a 1 e 2 thread. Efficienza ≤ 1 in L2 e
  > 1 in DRAM → (a). Nota: l'effetto è del 2–9%, la deviazione standard di `seq` è
  del 4%: potrebbe non essere nemmeno significativo.

### E. Modello a latenza e tetti di calcolo — **[P]**, manca la frequenza
Il diario confronta 3.4 GFLOP/s previsti (4.5 GHz) con 2.9 misurati per `omp_naive`,
e 27 con 20.8 per `omp_simd` in L2; le previsioni usano 3.8 GHz per il tetto a 6
core. Nessuna di queste frequenze è misurata (in WSL2 `/proc/cpuinfo` riporta sempre
2592 MHz). Peggio: i dati a 1 thread non si accordano tra loro.

- Assembly (verificato): `omp_naive`/`seq` = `vmulps` + catena di 8 `vaddss` per 8
  elementi → 4 cicli/elemento → 0.75 flop/ciclo. 2.85 GFLOP/s ⇒ **3.8 GHz**.
- `omp_blocked` = 1 accumulatore `vfmadd231ps` → 4 cicli per 8 elementi → 6
  flop/ciclo. 26.5 GFLOP/s a 1 thread ⇒ **≥ 4.4 GHz** (e la stima è per difetto:
  una parte del tempo è lettura dalla DRAM, vedi F).

Lo stesso core non può andare a 3.8 e a 4.4+ GHz nella stessa ora. Uno dei due
modelli è sbagliato. Candidato: **il loop è lungo solo 98 iterazioni** (784/8) e
ogni distanza riparte da un accumulatore azzerato, quindi la catena di una distanza
è *indipendente* dalla successiva. Il processore out-of-order (ROB da 224 µop ≈ 45
iterazioni da 5 µop) inizia la distanza successiva mentre la coda della precedente
è ancora in esecuzione: per una frazione del tempo ci sono **due catene in volo**.
"1 accumulatore ⇒ limite di 4 cicli per iterazione" vale per un loop lungo, non per
98 iterazioni. Se è così: il "limite di latenza" di `omp_simd` non è un tetto rigido
e parte del guadagno attribuito al micro-kernel 4×1 va ridimensionato.

Inoltre `omp_blocked` nella campagna (26.5 GFLOP/s, con la DRAM in mezzo) è il 27%
più veloce di `omp_simd` "in L2" nei preliminari (20.8): le due serie non sono
confrontabili (condizioni diverse), e il diario le confronta.

- **Esperimento 5** ⏱ ~1 min: *sonda di frequenza* — un loop con una sola catena di
  dipendenza di latenza nota (es. 2·10⁹ `imul` dipendenti = 3 cicli l'uno), eseguito
  da 1, 6 e 12 thread. Dà i GHz reali sotto carico: serve a tutto il modello (fase C)
  e all'analisi di Amdahl (se la frequenza cala da 1 a 6 core, l'efficienza ideale
  non è 1).
- **Esperimento 6** ⏱ ~2 min: dati sintetici in L2, stesso numero di flop, `--dim`
  784 contro 7840 (con `n_train` ÷ 10), 1 thread, `omp_simd`. Se la sovrapposizione
  out-of-order conta, con righe 10 volte più lunghe i GFLOP/s **scendono** verso
  6 flop/ciclo × f; se restano uguali l'ipotesi è falsa.

### F. "Il blocking elimina il collo di bottiglia della memoria" — **[P]**: lo riduce
Con QB=16 ogni blocco di query rilegge comunque il training dalla DRAM una volta:
24 × 188 MB = 4.52 GB per run. A 1 thread, alla banda di un core (~10–11 GB/s,
misurata con `omp_simd`), sono **≈ 0.41–0.47 s**. Confronto con i tempi a 1 thread:
`omp_blocked` 2.05 s (≈ 20%), `omp_blocked4` **0.89 s (≈ 50%)**.

Modello a due termini per `omp_blocked4` (4 gruppi di query per tile: il primo
gruppo trova le righe in DRAM, gli altri tre in L2): T ≈ 4.52 GB / B₁ + ¾ · 54.2
GFLOP / R_cache. Con B₁ = 10–11 GB/s e R_cache = 24 flop/ciclo × 4.4 GHz ≈ 106
GFLOP/s: T ≈ 0.41–0.45 + 0.38 = **0.79–0.83 s** contro 0.89 misurati. Quindi
`omp_blocked4` non "ridiventa memory-bound da 4 thread": la lettura dalla DRAM è
metà del suo tempo fin da 1 thread, e da 4 thread satura la banda condivisa.
Messaggio per il report (più forte di quello attuale): *più il kernel è veloce, più
riuso serve; QB=16 era dimensionato per un kernel 2.3 volte più lento*.

- **Esperimento 7** ⏱ ~8 min: sweep `tiles` anche a **1 thread** (oggi solo 6 e 12).
  Il modello prevede T(QB) = a + b/QB con b = 72.3 GB / B₁: una retta in 1/QB la cui
  pendenza è una *misura* della banda di un core. Previsto vs misurato pronto per il
  report.

### G. "omp_blocked a 12 thread resta sotto il tetto: è compute-bound" — **[P]**
Il "tetto" (23.5 GB/s × 12 flop/B = 282 GFLOP/s) è superato da `omp_blocked4`
(317): un tetto superato non è un tetto. Assume che ogni blocco legga la sua copia
dalla DRAM, ma i thread a blocchi scorrono le tile in sincronia e le condividono in
L3 come in A. Inoltre `omp_blocked` a 12 thread muove 16.6 GB/s, **lo stesso valore**
a cui `omp_blocked4` si ferma a 4 e 6 thread (16.7 / 16.4): se ~16.5 GB/s è la banda
DRAM reale, `omp_blocked` a 12 thread è al limite di memoria, non di calcolo.
- **Esperimento 8** ⏱ ~1 min: banda reale con un test STREAM-like (somma di un
  array da 1 GB) a 1, 2, 4, 6, 12 thread. Già previsto dal brief (fase C).
- Lo sweep `tiles` di `run_rest.sh` (QB=32 dimezza il traffico) discrimina: se
  `omp_blocked` a 12 thread accelera con QB=32, era memory-bound.

### H. Hyperthreading: "aiuta le varianti latency-bound (6 → 12 = 1.65×)" — **[P]**
Coerente, ma a 12 thread cambiano tre cose insieme: 2 thread per core, working set
2 × 150 KB > L2 (§3), e condivisione delle tile in L3. Il guadagno 1.65× è la somma
dei tre effetti. Il confronto pulito è in L2 (cachefit a 6 e 12 thread, training da
64 righe): lì resta solo l'effetto SMT. ⏱ aggiungere `--threads 12` a `cachefit`:
~2 min.

### I. SoA: "784 flussi: il prefetcher non li segue e il TLB manca" — **[P]**, non testata
È un'ipotesi ragionevole presentata come spiegazione. Punti deboli:
- TLB: 784 pagine attive stanno nel TLB di secondo livello (1536 voci), non nel
  primo (64): ogni accesso costa qualche ciclo in più, non un page walk. Potrebbe
  non bastare a spiegare un fattore 2.
- la curva di scaling di `omp_soa` ha un'anomalia non commentata: 2 thread 3.05 s,
  **4 thread 2.71 s** (efficienza da 0.92 a 0.52), poi riprende (6: 1.80, 8: 1.43);
- "in L2 il kernel SoA è ideale": fa ≈ 50 GFLOP/s, ma con 8 accumulatori il limite
  delle porte è 24 flop/ciclo ≈ 100 GFLOP/s. La metà mancante non è spiegata.
  Candidato: il blocco [784][64] è 200 KB, sta in L2 ma non in L1, e il kernel
  chiede 256 B nuovi ogni ~8 cicli: è limitato dal trasferimento L2 → L1.
- **Esperimento 9** (brief, fase C.3) variante `omp_soa_hp` con
  `madvise(MADV_HUGEPAGE)`: THP è in modalità `madvise` su questa macchina
  (verificato), quindi oggi il training è su pagine da 4 KB e il test è fattibile.
  Se recupera → TLB; se no → prefetcher.
- **Esperimento 10** ⏱ ~1 min: `omp_soa`/`omp_aosoa` su dati sintetici con
  `--dim 100` (blocco da 25 KB, sta in L1): se i GFLOP/s salgono verso 100, il
  limite in "L2" era la banda L2 → L1.

Nota di metodo: `omp_soa`/`omp_aosoa` cronometrano anche l'allocazione di 188 MB e
la trasposta (0.07–0.14 s secondo il diario). A 8 thread `omp_aosoa` impiega 0.75 s:
**il 10–15% è conversione di layout**, un termine quasi seriale (limitato dalla
memoria) che la curva di scaling mescola al kernel. Scelta legittima, ma va
riportata separando i due tempi (oggi `KNN_VERBOSE=1` li stampa solo su stderr).

### J. Scheduling: "tutti gli schedule sono entro il rumore" — **[S]**, ma nel punto sbagliato
Vero. Però lo sweep è fatto solo a 12 thread, cioè — per `omp_naive` e `omp_simd` —
nel regime crollato, dove nessuno schedule può fare differenza. Il punto informativo
è 6 (o 8) thread. Inoltre con carico uniforme `dynamic` non *può* vincere: va detto
esplicitamente (e se si vuole mostrare a cosa serve `dynamic`, serve un carico non
uniforme — non è richiesto). Per le varianti a blocchi i chunk 16 e 64 misurano la
granularità, non lo scheduling (già in `notes.md`): servono chunk 1, 2, 4.
⏱ `schedule --threads 6` per naive/simd: ~5 min.

### K. Accuratezza: "k=5 è il migliore", "il coseno normalizza l'intensità" — **[P]**
Tra k=1, 5, 10 le differenze sono 0.02–0.10 punti = 2–10 immagini su 10000;
l'incertezza binomiale su un'accuratezza del 97% è ±0.17 punti. "k=5 è il migliore"
non è statisticamente sostenibile: "indistinguibili" è la parola giusta. Coseno
contro euclidea (30–44 immagini, stesso test set) è più solido, ma la spiegazione
(spessore del tratto) è un'ipotesi non verificata: presentarla come tale.

### L. "0 mismatch nonostante il riordino delle somme" — **[S] ora sull'intero test set**
Fino a ieri era verificato su ≤ 384 query e sui dati sintetici; le accuratezze sul
test set completo erano prese con `--no-validate`. Vedi §4.1 per l'esito della
validazione completa lanciata per questa revisione.

### M. Metodologia: punti minori
- `seq`: primo run misurato 20.85 s, gli altri 18.9–19.1 (un warm-up non basta
  quando è il primo processo della campagna). La mediana regge; la std (4%) no.
- `omp_simd` a 1 thread: 6.71, 6.77, 6.62, 5.77, 5.24 s — deriva monotona del 22%
  dentro lo stesso processo. T(1) è il denominatore di tutta la curva di
  scalabilità: con 5 run non è stabile. Per i punti a 1 thread delle varianti
  memory-bound servono più run (10) e va riportato l'intervallo.
- `main.cpp` calcola la mediana come `sorted[n/2]` (con n pari è l'elemento
  superiore, non la mediana); `plot_results.py` usa la mediana vera. Con 5 run
  coincidono, con i 10 del variance probe no.
- `system_info.sh`: "Mem 7.7 Gi" è il limite della VM WSL2, non la RAM fisica
  (16 GB, single channel): nel report va scritto a mano. Non registra né lo stato
  dell'alimentazione né il piano energetico di Windows — proprio le due cose che
  hanno invalidato metà campagna (da WSL si leggono: `/sys/class/power_supply` e
  `powercfg.exe /getactivescheme`).
- `run_rest.sh` fa il controllo di deriva solo **alla fine**: se le condizioni sono
  sbagliate lo si scopre dopo 30 minuti. Serve anche all'inizio.

---

## 2. Correzioni al diario (applicate come *errata* in fondo a `notes.md`)

Le voci originali restano com'erano (sono la storia degli esperimenti); la voce
"Errata" del 16/11 le corregge.

1. **Picco DRAM 23.5 GB/s, non 47** — già corretto il 15/11 nella prima voce.
2. **La misura "anomala" di `omp_simd` a 6 thread con 192 query (15.8 GFLOP/s) non è
   anomala.** Tetto memory-bound: 23.5 GB/s ÷ 4 B × 3 flop = 17.6 GFLOP/s → è al 90%
   della banda DRAM, cioè il regime "flussi indipendenti". Eccezionale è invece la
   misura con 100 query (38.7 GFLOP/s, 2.2× il tetto): regime "convoglio".
   Precisazione rispetto al brief: l'ipotesi "con più passate per thread i thread si
   desincronizzano" **non è confermata dai dati validi** — nella campagna a 6 thread
   ogni thread fa 64 passate e il convoglio regge (0.90–0.92 s, un run a 1.07).
   Meglio H1 del §1-B: il convoglio regge finché una perturbazione non espelle un
   thread dalla finestra della L3.
3. **~9.6–10.9 GB/s a 1 thread è un limite del singolo core, non della DRAM**:
   legge di Little, ~10–12 miss L1 in volo (line fill buffer) × 64 B / ~70 ns ≈
   9–11 GB/s. Si vede anche dai dati: in convoglio ogni thread arriva a 13 GB/s.
4. **"il limite torna a essere il calcolo" (voce `omp_blocked`)**: vedi §1-F e §1-G.
5. **"39.9 GFLOP/s, coerente…" (voce variance probe)**: vedi §1-C.

---

## 3. Budget di cache del blocking

Verificato: il commento in `knn_omp_blocked` (righe 114–119) è **già corretto**
(tile 32 × 3136 B = 100 KB + 16 query = 50 KB = 150 KB per thread; con 2
hyperthread 300 KB > 256 KB). Il conto del brief, 2 × (TB + QB) × 3136 B =
2 × 48 × 3136 = 301 KB, coincide. Da aggiungere:

- **L1 per `omp_blocked4`:** il micro-kernel tiene in L1 4 query + 1 riga di
  training = 5 × 3136 B = 15.7 KB (su 32 KB): bene con 1 thread per core. Con 2
  hyperthread sono 31.4 KB: **al limite della L1d**. Conta per la fase D: un kernel
  6 × 2 terrebbe 8 righe = 25 KB per thread, 50 KB con SMT → oltre la L1.
- I `TopK` del blocco (16 × 648 B = 10 KB) non sono nel conto: trascurabili in L2.
- Su questo processore (Comet Lake client) la L3 è **inclusiva**: ogni riga che sta
  in una L2 occupa anche spazio in L3. Con
  12 thread: 12 × 150 KB = 1.8 MB su 12 MB, nessun problema; con TB=256 e QB=32:
  12 × 903 KB = 10.8 MB ≈ tutta la L3 (coerente con la previsione 3.3).
- Dove sta il "ginocchio" lo dirà lo sweep `tiles` (da rifare). Attenzione a come
  leggerlo: con il convoglio (§1-A) le tile sono condivise tra thread, quindi il
  ginocchio a 12 thread può stare più in là del conto statico qui sopra.

---

## 4. Revisione del codice

### 4.1 Correttezza
- **Validazione sull'intero test set** (10000 query × k ∈ {1, 5, 10} × 2 metriche ×
  6 varianti, 8 thread): *esito in fondo a questo paragrafo* —
  `results/raw/validation_full_20261116_002255.log`.
  **Esito: ancora in corso al momento del commit** (lanciata alle 00:22, ~35 min);
  questa riga verrà sostituita con il numero di mismatch per variante.
- **Race condition: nessuna trovata.** Per ogni regione parallela: `train`, `test`,
  `params`, `train_norm`, `test_norm`, `xt`/`xb` sono condivisi in sola lettura;
  `TopK top`, `acc[]`, `tops` sono dichiarati dentro la regione (privati); ogni
  `out[i]` è scritto da un solo thread. ThreadSanitizer (build `KNN_TSAN`, 4 thread,
  12 configurazioni): 124 segnalazioni, **tutte** tra un worker e il thread main
  *fuori* dalla regione (lettura di `pred` dopo la chiamata, distruttori, riuso
  dello stack) — falsi positivi noti perché libgomp non è strumentata; **nessuna**
  tra due worker. Limite: TSan con libgomp non strumentata non è una prova formale.
- Manca `default(none)`: non è un bug, ma elencare esplicitamente shared/private
  nelle direttive è ciò che l'esaminatore vuole sentire all'orale (e documenta il
  ragionamento qui sopra nel codice).
- **False sharing: trascurabile, e si può quantificare.** `out` è 384 byte = 6 linee
  di cache; viene scritto una volta per query (o per blocco): al massimo 384
  scritture in 0.17 s contro ~10¹⁰ operazioni. Con `static` i thread scrivono zone
  contigue, quindi si toccano solo le linee di confine. I `tops` sono un'allocazione
  per thread. Risposta alla domanda aperta di `CLAUDE.md`: c'è (in teoria), costa
  nulla (la frequenza di scrittura è 8 ordini di grandezza sotto quella del kernel).

### 4.2 Prestazioni
- **P1 — `row_norms()` sequenziale dentro il tempo, in tutte le varianti** (coseno).
  47 M elementi con una catena scalare (verificato nell'assembly: `vmulps` xmm +
  `vaddss`): decine di ms, invisibili su 19 s di `seq`, potenzialmente il 20–25% di
  `omp_blocked4` a 12 thread. È la frazione seriale di Amdahl del progetto. Il
  commento `// TODO: parallelise too (small cost)` non è più vero per le varianti
  veloci. Parallelizzare il loop *sulle righe* lascia ogni norma bit-identica
  (nessun rischio per la validazione).
- **P2 — trasposta e allocazione dentro il tempo in `omp_soa`/`omp_aosoa`** (§1-I):
  188 MB allocati, toccati per la prima volta (page fault, costosi in una VM) e
  liberati a ogni chiamata, warm-up compreso.
- **P3 — riduzione orizzontale di `omp simd`**: GCC salva l'accumulatore sullo stack
  e somma gli 8 float con 8 `vaddss` da memoria (catena da 32 cicli per distanza).
  Non è sul percorso critico (la distanza successiva non ne dipende), ma spiega
  perché ordine e valore delle somme non coincidono con `seq`.
- **P4 — `tops[i - i0] = TopK(params.k)`** copia 648 B *non inizializzati* per query
  e per blocco: costo nullo, ma è una lettura di memoria indeterminata (un
  sanitizer di memoria la segnalerebbe). Basterebbe azzerare `size`.
- Nessuna allocazione nei loop caldi; `acc[64]` azzerato per blocco (256 B): ok.

### 4.3 Resti e casi limite
| caso | esito |
|---|---|
| `d` non multiplo di 8 | epilogo scalare generato da GCC; coperto da ctest (`--dim 50`) |
| `n_test` non multiplo di QB / di 4 | `min()` + loop dei resti; coperto (97 query) solo con QB di default. Il diario riporta prove manuali con `KNN_TB/QB` = 1/1, 7/5, 13/97 per `omp_blocked`, **non per `omp_blocked4`** → da aggiungere a `smoke_test.sh` |
| `n_train` non multiplo di TB / di 64 | coperto (1500 = 23 × 64 + 28; padding SoA saltato) |
| `k > n_train` | funziona: `TopK.size < k`, il voto usa `size` elementi. Non testato |
| `k > 128` | rifiutato da `main.cpp`; `TopK` da solo non controlla |
| coseno con norma nulla | `cosine_distance` → 1.0, uguale in tutte le varianti |
| NaN nei dati | un NaN entrato in `TopK` quando `size < k` resta in ultima posizione e blocca gli inserimenti successivi (`d < NaN` è falso). Irrilevante per MNIST, da sapere |
| `test.d != train.d` | non controllato (non raggiungibile dal driver) |

Un punto sottile per l'orale: in `omp_blocked4` le query "di resto" passano da un
kernel diverso (`sq_euclidean_simd`) da quello 4×1. I due fanno la stessa somma per
corsia, ma niente garantisce che GCC generi la stessa riduzione finale: la distanza
di una query può in teoria cambiare di 1 ulp a seconda di QB e di `n_test`. Non è
mai successo nei test; è il motivo per cui si valida *ogni* configurazione.

### 4.4 Il riferimento non è definito in modo unico
`seq` è compilato con `-ffp-contract=fast` (default di GCC). Verificato
nell'assembly: il corpo vettorizzato usa `vmulps` + `vaddss` (**senza** FMA),
l'epilogo scalare per i `d mod 8` elementi finali usa `vfmadd231ss` (**con** FMA),
e la build `novec` usa FMA ovunque. Quindi "il risultato sequenziale" dipende da
come il compilatore decide di generare il loop. Per MNIST (784 = 98 × 8) l'epilogo
non esiste e le predizioni coincidono anche con `novec` (diario), ma la cache dei
riferimenti (`.cache/`) non distingue la build. Da dire nel report in una frase;
l'alternativa (compilare `knn_seq.cpp` con `-ffp-contract=off`) cambia il
riferimento → **decisione dello studente** (regola invariante).

---

## 5. Copertura della traccia

✅ fatto e misurato (dati validi) · 🟡 fatto, misura mancante o non valida · ❌ mancante

| requisito | stato | nota |
|---|---|---|
| Sequenziale + parallela, validate contro il sequenziale | ✅ | 0 mismatch su 384 query in ogni configurazione; intero test set: §4.1 |
| Euclidea **e** coseno, k = 1, 5, 10, accuratezza e tempi | 🟡 | accuratezze ✅ (10000 query); **tempi** solo a batteria → `kmetric` in `run_rest.sh` |
| Alta dimensionalità, sequenziale ≥ 10 s | ✅ | MNIST 784; `seq` 19.08 s con 384 query |
| Parallel loop sulle query | ✅ | `omp_naive` |
| Distanze vettorizzate; AoS vs SoA | ✅ / 🟡 | strong scaling ✅; confronto in L2/L3/DRAM (`cachefit`) solo preliminare o a batteria |
| Top-k per thread; insertion sort vs heap al crescere di k | ❌ | top-k privato per thread ✅, confronto mai fatto. Proposta: argomento analitico (inserimenti attesi ≈ k·ln(n/k) ≈ 47 su 60000 per k=5 → il top-k pesa < 0.1%) + un contatore, invece di una variante che darebbe un risultato nullo |
| Blocking sul training | ✅ / 🟡 | `omp_blocked` misurato; sweep TB × QB solo a batteria |
| Thread 1, 2, 4, 6, 8, 12; numero ottimale | ✅ | ottimo: 8 senza blocking, 12 con blocking |
| Scheduling static/dynamic/guided + chunk | ✅ | solo a 12 thread (§1-J) |
| Con/senza vettorizzazione; report di vettorizzazione | 🟡 | preliminare (3 run) o a batteria → `vec` in `run_rest.sh`; report di GCC citato nel diario, non salvato in un file |
| Wall-clock **e** CPU; ≥ 5 run; media, std, min, max | ✅ | tutto in `summary_*.csv` |
| Speedup per **più dimensioni dell'input** | ❌ | `sizes` esiste in `run_benchmarks.py` ma non è né in `run_all.sh` né in `run_rest.sh` |
| Strong scaling | ✅ | |
| Weak scaling (extra) | ❌ | |
| Colli di bottiglia; Amdahl/Gustafson | 🟡 | analisi qualitativa ricca; nessun fit di Amdahl, nessuna frazione seriale stimata (P1 è l'esempio pronto) |
| Hardware/software documentati | 🟡 | `_system.txt` ✅; mancano RAM fisica/single channel, flag di compilazione, alimentazione (§1-M) |
| Opzionale: |a|² + |b|² − 2a·b | ❌ | fase D2 |
| Report e slide | ❌ | solo `outline.md` |

---

## 6. 🛑 Decisioni da prendere insieme (prima di toccare il codice)

Proposte in ordine di valore per il voto; ognuna è piccola e indipendente.

| # | proposta | tipo | tocca |
|---|---|---|---|
| 1 | Norme parallele (P1): nuovo helper `row_norms_parallel` usato dalle varianti OpenMP — **oppure** nuova variante `omp_blocked4_pn` per tenere "prima/dopo" come esempio di Amdahl nel report | codice | `knn_omp.cpp` |
| 2 | Variante diagnostica `omp_simd_stagger` (esperimento 2) | codice | nuova variante |
| 3 | Strumenti di misura: sonda di frequenza + test di banda STREAM-like (esperimenti 5, 8), eseguibile separato in `tools/` | codice | nuovo file |
| 4 | `run_benchmarks.py`: thread 7, 9, 10, 11 opzionali; `tiles` a 1 thread; `cachefit` a 2 e 12 thread; `schedule` a 6 thread e chunk 1/2/4 per le varianti a blocchi | script | nessuna variante |
| 5 | `run_rest.sh`: controllo di deriva anche all'inizio; aggiungere `sizes`; `system_info.sh` con alimentazione e piano energetico | script | |
| 6 | `smoke_test.sh`: combinazioni dispari di `KNN_TB`/`KNN_QB` per le varianti a blocchi, `k > n_train` | test | |
| 7 | `default(none)` nelle direttive (nessun cambiamento di comportamento) | codice | varianti esistenti |
| 8 | Tempo della trasposta separato nel CSV per `omp_soa`/`omp_aosoa` | codice | driver + 2 varianti |
| 9 | `-ffp-contract=off` per `knn_seq.cpp` (§4.4) — io **non** lo farei: basta dichiararlo | flag | riferimento |

Domande aperte per te:
- **(1)** preferisci correggere le varianti esistenti o tenere il "prima/dopo"?
  La regola "una variante = una idea" suggerisce la variante nuova.
- **(7), (8)** modificano varianti esistenti senza cambiarne l'idea: va bene?
- Quanto tempo c'è domani per le misure? Gli esperimenti 1–10 aggiungono ~30 min ai
  ~30 di `run_rest.sh`; se è troppo, i più importanti sono 2, 5, 7, 8.
