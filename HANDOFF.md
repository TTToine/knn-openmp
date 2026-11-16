# HANDOFF — knn-openmp

Passaggio di consegne per chi riprende il lavoro (studente o altro modello).
Aggiornato alla fine di ogni punto di lavoro. Leggere insieme a `CLAUDE.md`
(regole), `report/notes.md` (diario), `report/review.md` (revisione critica),
`report/perf_model.md` (modello e previsioni).

## Stato (2026-11-16, sessione a batteria: nessuna misura valida oggi)

- **Punto 3 (16/11, a batteria): fatto, non misurato.** Variante
  `omp_blocked4_heap` (top-k con heap, 0 mismatch anche a k = 128), weak scaling
  negli script, esperimento `tools/sync_strategies` (6 strategie, tutte uguali al
  seriale). `ctest` ha ora 2 test.
- **Punto 4 (16/11, a batteria): fatto, non misurato.** `src/knn_avx2.cpp`:
  `omp_blocked_avx2`, `omp_blocked_avx2u4`, `omp_mk4x1/4x2/6x2/2x4`,
  `omp_blocked_ea`. 17 varianti, tutte a 0 mismatch. Assembly controllato
  (`tools/asm_loops.py`). Previsioni A–E e dimostrazione di esattezza nel diario.
- Fasi A, B, C del brief concluse (commit `64fc89f`): campagna valida a metà
  (`threads`, `schedule`), revisione da esaminatore, modello di prestazioni con
  previsioni P-1…P-6 registrate, validazione sull'intero test set a 0 mismatch.
- **Tutte le misure di oggi sono vietate**: portatile a batteria. Oggi si scrive e
  si valida codice (test brevi, input piccoli); ogni cosa nuova aggiunge il suo
  comando alla coda qui sotto.

## Coda delle misure (da lanciare in UN blocco, in carica)

Condizioni: caricatore collegato, piano "Prestazioni elevate", nessuna app pesante,
niente compilazioni durante le misure. `run_benchmarks.py` si ferma da solo se il
portatile è a batteria. Ordine obbligato: prima compilare tutto, poi misurare.

```bash
cd ~/knn-openmp
# 0. compilare PRIMA (mai durante le misure)
cmake -S . -B build && cmake --build build -j
cmake -S . -B build-novec -DKNN_NO_VEC=ON && cmake --build build-novec -j   # era obsoleta
ctest --test-dir build --output-on-failure                                   # 0 mismatch

# 1. seconda metà della campagna (~30 min): tiles, cachefit, vec, kmetric + controlli di deriva
bash scripts/run_rest.sh  2>&1 | tee results/raw/run_rest_$(date +%Y%m%d_%H%M%S).log

# 2. esperimenti della revisione (~25 min + 15 di sizes): comincia con hwprobe
bash scripts/run_extra.sh 2>&1 | tee results/raw/run_extra_$(date +%Y%m%d_%H%M%S).log

# 3. esperimenti aggiunti in questa sessione (elenco più sotto)
bash scripts/run_new.sh   2>&1 | tee results/raw/run_new_$(date +%Y%m%d_%H%M%S).log
```

Dopo le misure:
1. `python3 scripts/plot_results.py <csv>` (un CSV alla volta) e
   `python3 scripts/plot_roofline.py results/raw/threads_*.csv --hwprobe results/raw/hwprobe_*.csv`;
2. scrivere `report/predictions_vs_measured.md`: per ogni previsione di
   `report/perf_model.md` §6 (P-1…P-6), di `PREVISIONI_run_rest.md` e delle voci
   "previsione" del diario: previsto, misurato, scarto, spiegazione. Sostituire i
   parametri **[I]** del modello (f, B_max) con i valori di `hwprobe`;
3. una voce in `report/notes.md` per esperimento.

## Misure aggiunte in questa sessione

Tutte in `scripts/run_new.sh`; previsioni nel diario (voci del 16/11).

| # | esperimento | comando | previsione da verificare |
|---|---|---|---|
| N1 | top-k: insertion sort vs heap, k = 1…128 | `run_benchmarks.py topk --variants omp_blocked4,omp_blocked4_heap --threads 6` | T1: nessuna differenza per ogni k |
| N2 | weak scaling (query ∝ thread; poi training ∝ thread) | `run_benchmarks.py weak ...` + `plot_weak.py` | W1: `omp_blocked4` cala da 4 thread (DRAM) |
| N4 | kernel in L2 (64 righe), 1 e 6 thread, euclidea e coseno | `run_benchmarks.py threads --synthetic --n-train 64 ...` | A: intrinsics = `omp simd`; B: u4 2.5–3.5× avx2; C: MR×NR euclidea ≈ 4×1; D: coseno 4×2 fino a 2× il 4×1, 6×2 più lento del 4×2 (spill) |
| N5 | stessi kernel su MNIST, strong scaling (+ coseno) | `run_benchmarks.py threads --variants ...` | da 4 thread tutti sullo stesso limite DRAM |
| N6 | tile sweep per `omp_mk4x1`, `omp_mk4x2` a 6 thread | `run_benchmarks.py tiles ...` | QB = 32 aiuta il kernel più veloce |
| N7 | early abandoning: passo del controllo, 1 e 6 thread; strong scaling | `run_benchmarks.py eastep --threads 6` | E: 1.7–2.2× `omp_blocked_avx2` a 1 thread; passo ottimo 32–64 |
| N3 | sincronizzazione: critical / atomic / reduction / packed / padded | `bash scripts/sync_strategies.sh` | S1–S4: critical peggiora coi thread; packed 5–20× più lenta di padded |

## Cosa resta

Vedi la roadmap in `CLAUDE.md`. In breve: misure in coda, report e slide (la prosa
è dello studente), simulazione d'esame (fase F del brief).

## Rischi / cose da sapere

- I tempi presi a batteria non valgono nulla (15%–2× più lenti): metà della prima
  campagna è stata buttata per questo (`results/raw/invalid_battery/`).
- `FABLE_BRIEF.md` non è nel repository (non tracciato): resta solo sul disco.
- La cache dei riferimenti (`.cache/`) non distingue la build (`build` /
  `build-novec`): vedi `review.md` §4.4.
