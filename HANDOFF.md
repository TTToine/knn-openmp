# HANDOFF — knn-openmp

Passaggio di consegne per chi riprende il lavoro (studente o altro modello).
Aggiornato alla fine di ogni punto di lavoro. Leggere insieme a `CLAUDE.md`
(regole), `report/notes.md` (diario), `report/review.md` (revisione critica),
`report/perf_model.md` (modello e previsioni).

## Stato (2026-11-16, sessione a batteria: nessuna misura valida oggi)

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

# 3. esperimenti aggiunti in questa sessione: vedi le voci "MISURA" più sotto
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

(vuoto: aggiornato a ogni punto)

## Cosa resta

Vedi la roadmap in `CLAUDE.md`. In breve: misure in coda, report e slide (la prosa
è dello studente), simulazione d'esame (fase F del brief).

## Rischi / cose da sapere

- I tempi presi a batteria non valgono nulla (15%–2× più lenti): metà della prima
  campagna è stata buttata per questo (`results/raw/invalid_battery/`).
- `FABLE_BRIEF.md` non è nel repository (non tracciato): resta solo sul disco.
- La cache dei riferimenti (`.cache/`) non distingue la build (`build` /
  `build-novec`): vedi `review.md` §4.4.
