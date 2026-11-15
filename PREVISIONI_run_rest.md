# Previsioni per `run_rest.sh` — NON SONO MISURE

> ⚠️ **Questo file contiene stime, non dati.** Nessun numero qui sotto va usato
> nei grafici, nelle tabelle dei risultati o nel report come se fosse misurato.
> Servono a due cose:
> 1. permettere di lavorare (modello di prestazioni, revisione, progetto CUDA)
>    prima che le misure mancanti vengano rifatte;
> 2. fare da **previsioni registrate prima della misura**: quando `run_rest.sh`
>    sarà eseguito, la tabella *previsto vs misurato* diventa essa stessa
>    contenuto del report (metodologia: "abbiamo previsto X perché Y, misurato Z").
>
> Scritte il 2026-11-16 alle 00:10, a partire dal commit `98806d6`, prima di qualunque
> misura di `run_rest.sh`.

## 1. Da dove vengono le stime

Dati **validi** (alimentazione collegata, 5 run, 384 query, 60000 righe di training):
`threads_20261115_184752.csv`, `schedule_20261115_185923.csv`, e le accuratezze in
`accuracy_full_20261115_193504.csv` e nei CSV di kmetric (l'accuratezza non dipende
dalla velocità). Dati **preliminari** (alimentazione collegata, 3 run):
`preliminary/cachefit_*`, `preliminary/novec_*`.

Dati **a batteria** (`invalid_battery/`): i valori assoluti non sono affidabili, ma
i *rapporti tra configurazioni misurate nella stessa finestra* lo sono in parte.
Il fattore di rallentamento stimato dalle configurazioni presenti in entrambi i set:

| finestra (15/11) | esperimento | fattore batteria stimato |
|---|---|---|
| 19:04–19:10 | tiles, 6 thread | ≈ 1.1–1.2× |
| 19:06–19:11 | tiles, 12 thread | ≈ 1.8–2.0× |
| 19:11–19:14 | cachefit | ≈ 1.5–1.9× (1 e 6 thread) |
| 19:14–19:17 | vec | ≈ 1.3–1.8×, molto rumoroso |
| 19:17–19:35 | kmetric | seq ≈ 1.25–1.9×, blocked4 12 thr ≈ 2× |

A 12 thread il limite di potenza a batteria pesa di più: tutti i core sono attivi
e la frequenza scende di più.

## 2. Osservazioni sui dati validi (da usare subito, non sono previsioni)

Riferimento per i conti: 384 × 60000 × 784 × 3 flop = **54.2 GFLOP** per run,
training = 188.2 MB, quindi una passata completa sul training per ogni query
corrisponde a 72.3 GB letti.

1. **Le varianti non a blocchi crollano a 12 thread.**

   | variante | 8 thr | 12 thr | banda apparente a 12 thr |
   |---|---|---|---|
   | `omp_simd` | 0.725 s | 2.88 s | 25 GB/s |
   | `omp_naive` | 2.29 s | 2.74 s | 26 GB/s |
   | `omp_soa` | 1.42 s | 2.90 s | 25 GB/s |
   | `omp_aosoa` | 0.754 s | 3.26 s | 22 GB/s |

   A 12 thread tutte convergono al picco teorico della DRAM in single channel
   (23.5 GB/s): i dati arrivano davvero dalla memoria. A 8 thread `omp_simd` legge
   invece 72.3 GB in 0.725 s, cioè **≈ 100 GB/s apparenti, 4 volte il picco**:
   i thread si condividono le righe in L3. Interpretazione: fino a 8 flussi
   la L3 da 12 MB tiene le righe che servono a più thread, con 12 flussi
   desincronizzati il riuso salta.
2. **Prova a favore, nello sweep di scheduling:** `omp_simd` con `dynamic,64` a
   12 thread impiega 0.90 s contro ~2.8 s di tutti gli altri schedule. Con 384
   query e chunk da 64 ci sono solo **6 chunk, quindi lavorano solo 6 thread**, e
   il tempo torna quello misurato a 6 thread (0.91 s). Meno flussi contemporanei
   danno di più. È un risultato da mettere nel report.
3. **Artefatto di granularità nelle varianti a blocchi.** Il loop parallelo è sui
   blocchi (24 blocchi da 16 query), quindi il chunk size si conta in blocchi:
   - `static,16` e `dynamic,16` danno 2 chunk, cioè 2 thread attivi
     (`omp_blocked` 1.39 s ≈ tempo a 2 thread, 1.08 s);
   - `dynamic,64` dà 1 chunk, cioè 1 thread (2.05 s = tempo a 1 thread).

   Non è un problema di scheduling, è un problema di granularità. Va spiegato così
   (oppure lo sweep di scheduling per queste varianti va fatto con chunk 1–4).
4. **Plateau di `omp_blocked4` tra 4 e 6 thread** (0.270 → 0.275 s). Con QB=16 il
   training viene riletto dalla DRAM 24 volte: 24 × 188 MB = 4.5 GB. In 0.27 s
   fanno **16.7 GB/s ≈ 71% del picco**, vicino alla banda sostenibile in single
   channel. Quindi `omp_blocked4` con QB=16 torna **memory-bound già da 4 thread**:
   il micro-kernel è diventato così veloce che il riuso di 16 volte non basta più.
   A 12 thread (0.171 s, 26 GB/s apparenti, sopra il picco) c'è di nuovo un po' di
   condivisione in L3 tra thread che percorrono le stesse tile.
5. **Tetto di calcolo realistico** per questo kernel: sub + FMA sono 3 flop in 2 µop
   vettoriali, quindi con 2 porte FP fanno 8 elementi × 3 flop per ciclo e per core
   = 24 flop/ciclo/core. Con 6 core × ~3.8 GHz si ottengono ≈ **550 GFLOP/s**.
   `omp_blocked4` a 12 thread fa 317 GFLOP/s ≈ 58%. La frequenza reale sotto carico
   va stimata o misurata e dichiarata: 3.8 GHz è un'ipotesi.
6. **Accuratezza (valida):** 1-NN euclidea sul test set completo = **96.91%**. È
   esattamente il valore di riferimento noto per MNIST (errore del 3.09% nella
   tabella di LeCun). Ottima prova di correttezza da citare. Il coseno è sempre
   migliore di circa 0.3–0.4 punti (k=5: 97.37% contro 96.93%).

## 3. Previsioni per esperimento

Confidenza: **A** = alta (stessa configurazione già misurata alla corrente, o
argomento solido), **B** = media (riscalata da dati a batteria), **C** = bassa
(soprattutto a 12 thread).

### 3.1 Controllo di deriva (`threads` per `omp_blocked4`) — confidenza A
- `seq`: **19.1 s ± 3%** (riferimento 19.08 s).
- `omp_blocked4`: 1 thr ≈ 0.89 s, 4 thr ≈ 0.27 s, 6 thr ≈ 0.25–0.28 s, 12 thr
  ≈ **0.171 s ± 5%**.
- Il plateau tra 4 e 6 thread deve ripresentarsi (punto 2.4).
- Di notte la stanza è più fresca, quindi un 1–3% più veloce non sarebbe strano.
- **Falsificazione:** uno scarto > 5% su `seq` vuol dire che le condizioni sono
  cambiate (piano energetico, batteria, carico in background), e le due metà della
  campagna non si possono confrontare direttamente.

### 3.2 `kmetric`: k ∈ {1, 5, 10} × {euclidea, coseno}, 384 query — confidenza A/B
- `seq`: **≈ 19.1 s per tutte e sei le configurazioni** (±3%). Euclidea e coseno
  hanno la stessa catena di dipendenza (una `vaddss` dipendente per elemento,
  riduzione fold-left), e `k` costa quasi zero: su 60000 candidati quasi tutti i
  `push` vengono scartati dal primo confronto con il k-esimo.
- `omp_blocked4` a 12 thread, euclidea: **0.17–0.18 s** per k = 1, 5, 10.
- `omp_blocked4` a 12 thread, coseno: **≈ 0.21–0.23 s**, cioè circa +0.04–0.06 s.
  Causa (verificata nel codice al commit `98806d6`): le norme del training vengono
  calcolate da `row_norms()` **in sequenziale dentro la regione cronometrata**, in
  tutte le varianti. Fa 47 M elementi con una
  catena scalare di add (~4 cicli l'una) ≈ 40–50 ms. È un **esempio di Amdahl da
  manuale**: 1 thread su 0.17 s, invisibile a 1 thread e pari al 20% del tempo a
  12. Anche i dati a batteria mostrano il coseno ~10% più lento. Se l'ipotesi è
  confermata, una variante con le norme parallelizzate o precalcolate fuori dal
  tempo (va dichiarato) recupera la differenza.
- Speedup previsto rispetto a `seq`: euclidea ≈ **105–115×**, coseno ≈ **85–90×**
  (≈ 105× se le norme vengono parallelizzate).
- L'accuratezza su 384 query è già nei CSV a batteria ed è valida: non va prevista.

### 3.3 `tiles`: TB ∈ {8…256} × QB ∈ {4, 8, 16, 32}

Regola per leggere i risultati: il traffico DRAM vale circa
(384 / QB) × 188 MB (**QB comanda il riuso**), mentre l'ingombro in cache di un
thread vale circa (TB + QB) × 3136 B (**TB comanda dove sta la tile**). A 12
thread, due hyperthread si dividono la L2 da 256 KB e 12 thread si dividono la L3
da 12 MB.

**6 thread — confidenza B**

| | QB=4 | QB=8 | QB=16 | QB=32 |
|---|---|---|---|---|
| `omp_blocked` | 0.50–0.70 s | 0.45–0.60 | 0.44–0.57 | **0.41–0.45** (TB 8–32) |
| `omp_blocked4` | 0.34–0.41 | 0.29–0.39 | 0.23–0.32 | **0.20–0.23** |

- **Il miglior QB è 32 per entrambe le varianti.** Previsione netta: per
  `omp_blocked4` con QB=32 il plateau tra 4 e 6 thread si attenua o scompare,
  perché il traffico DRAM si dimezza (punto 2.4).
- A 6 thread TB conta poco, anche a 256: 6 tile da ~900 KB stanno comunque in L3, e
  il kernel con riuso in registri regge la banda della L3.

**12 thread — confidenza C** (dati a batteria molto rumorosi)

| | QB=4 | QB=8 | QB=16 | QB=32 |
|---|---|---|---|---|
| `omp_blocked` | 0.35–1.4 s | 0.31–0.85 | **0.27–0.30** (TB 8–32) | 0.25–0.40 |
| `omp_blocked4` | 0.27–0.75 | 0.25–0.55 | **0.16–0.18** (TB 16–32) | 0.17–0.19 (TB 16–64) |

- **TB = 128 e 256 peggiorano nettamente a 12 thread** (≈ 1.5–2× rispetto all'ottimo):
  12 × (256 + 32) × 3136 B ≈ 10.8 MB, quanto tutta la L3. Le tile non stanno più
  in nessuna cache condivisa.
- **QB = 4 è il caso peggiore** e il più rumoroso: il traffico DRAM è 4 volte quello
  con QB=16, quindi si torna memory-bound.
- QB=32 dà 12 blocchi, cioè 1 per thread a 12 thread: in questo esperimento non ci
  sono problemi di granularità (lo sweep è fatto solo a 6 e 12 thread). A 8 thread
  invece sì.

### 3.4 `cachefit` a 1 e 6 thread — confidenza A per naive e simd, B per soa e aosoa

Tempi previsti (mediana). Naive e simd vengono dalle misure preliminari alla
corrente, soa e aosoa sono riscalati dai rapporti misurati a batteria.

| training (query) | thr | naive | simd | soa | aosoa |
|---|---|---|---|---|---|
| 64 (10000), in L2 | 1 | 0.52 s | 0.072 | **0.035–0.040** | **0.030–0.035** |
| 2000 (2000), in L3 | 1 | 3.67 | 0.72 | 0.30–0.40 | 0.25–0.32 |
| 60000 (100), DRAM | 1 | 5.5 | 1.7–2.0 | 1.5–2.0 | 1.3–1.6 |
| 64, L2 | 6 | 0.086 | 0.014 | ≈ 0.007–0.008 | ≈ 0.007–0.008 |
| 2000, L3 | 6 | 0.565 | 0.116 | 0.11–0.13 | 0.08–0.09 |
| 60000, DRAM | 6 | 0.92 | 0.35–0.50 | **0.70–0.90** | 0.35–0.50 |

Lettura prevista:
- in L2, soa e aosoa sono ≈ **2× più veloci di simd**: 8 accumulatori contro uno
  solo, e nessuna riduzione orizzontale;
- su DRAM a 1 thread soa ≈ simd e aosoa leggermente migliore;
- su DRAM a 6 thread **soa è ≈ 2× più lenta** di simd e aosoa: con 6 thread ci
  sono 6 × 784 flussi distanti 240 KB, che mandano in crisi TLB e prefetcher;
- aosoa recupera tutto;
- naive ha GFLOP/s quasi costanti (2.5–2.9 per core) in tutti e tre i regimi
  (compute-bound).
- Attenzione: con 6 thread su 64 righe i tempi sono di 7–14 ms, quindi l'overhead
  della regione parallela e il rumore pesano molto. Valutare più run oppure più
  query.

### 3.5 `vec` (con e senza `-fno-tree-vectorize`, 1 thread) — confidenza A
- `omp_naive`: vec/novec ≈ **1.02–1.05×** (in L2 e su DRAM).
- `omp_simd`: vec/novec ≈ **1.00×**: `#pragma omp simd` forza comunque la
  vettorizzazione e l'assembly è identico.
- Valori assoluti attesi come nelle misure preliminari: L2 naive ≈ 0.52 s,
  simd ≈ 0.072 s; DRAM naive ≈ 5.5–5.8 s, simd ≈ 1.95–2.0 s.
- I dati a batteria (naive novec/vec = 1.68 in L2) sono **rumore**, non un effetto.
- Prerequisito: `build-novec` deve essere ricompilata al commit attuale.

## 4. Da verificare nel codice prima di fidarsi delle previsioni
- [x] `row_norms()` sequenziale e dentro il tempo in **tutte** le varianti (incluso
      `omp_blocked4`, `src/knn_omp.cpp` ~riga 363) quando si usa il coseno:
      **verificato al commit `98806d6`**, è la base della previsione 3.2.
- [ ] Il `tiles` di `run_benchmarks.py` usa `--n-test 384` e i 16 valori di TB × QB
      del CSV a batteria.
- [ ] `run_rest.sh` **non** ricompila `build-novec`: la presuppone già aggiornata
      al commit attuale. Ricompilarla prima di lanciare le misure.
- [ ] Ordine delle varianti ruotato anche in `cachefit` e `vec` (per la deriva
      termica).

## 5. Istruzioni per chi usa questo file (Fable o altro modello)
- Usa questi numeri **solo** per pianificare, per il modello di prestazioni (fase C
  del brief) e per decidere quali esperimenti contano. Non generare grafici o tabelle
  di risultati da questi numeri.
- Quando `run_rest.sh` è stato eseguito: crea `report/predictions_vs_measured.md`
  con una tabella *previsto / misurato / dentro l'intervallo? / spiegazione dello
  scarto*, e aggiorna `report/notes.md`. Previsioni sbagliate vanno tenute e
  spiegate, non cancellate: sono la parte più istruttiva.
- Questo file resta nel repository, invariato, come documentazione delle previsioni
  fatte prima della misura.
