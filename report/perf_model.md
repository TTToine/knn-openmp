# Modello di prestazioni — k-NN OpenMP + SIMD

Scopo: passare da "abbiamo misurato" a "abbiamo previsto, misurato e capito lo
scarto". Tre regole:

1. ogni numero qui dentro è marcato **[M]** misurato (dati validi), **[C]** calibrato
   (ricavato da una misura dello stesso esperimento: *non* è una previsione),
   **[I]** ipotesi dichiarata, **[P]** previsione (da confrontare con una misura
   futura);
2. i numeri **[I]** vanno sostituiti con quelli di `scripts/hwprobe.sh` appena
   eseguito (frequenza reale, banda DRAM reale);
3. niente numeri da `PREVISIONI_run_rest.md` né dai dati a batteria.

Stato: scritto il 2026-11-16 sui dati di `threads_20261115_184752.csv`.

## 1. Il problema in numeri

| grandezza | valore |
|---|---|
| lavoro per run (384 query × 60000 × 784 × 3 flop) | W = **54.2 GFLOP** |
| flop per elemento | 3 (sub, mul, add; mul + add fusi in una FMA) |
| training | 188.2 MB (riga = 3136 B = 49 linee di cache) |
| byte letti se ogni query rilegge il training | 72.3 GB |
| byte letti se ogni blocco di QB=16 query lo rilegge una volta | D = **4.52 GB** |

## 2. La macchina: i parametri del modello

| parametro | valore | tipo | da dove |
|---|---|---|---|
| f, frequenza sotto carico | **4.2 GHz** ± 10% | **[I]** | tra i 3.8 GHz impliciti in `omp_naive` e i ≥ 4.4 impliciti in `omp_blocked` (review §1-E). Turbo di targa: 5.0 (1 core), 4.3 (6 core) |
| latenza `vaddss` / `vfmadd231ps` | 4 cicli | noto | Skylake / Comet Lake |
| porte FP | 2 (p0, p1) → 2 FMA per ciclo | noto | |
| ROB | 224 µop | noto | finestra out-of-order |
| B_peak, banda DRAM teorica | 23.5 GB/s | noto | 1 × DDR4-2933, single channel |
| B_max, banda DRAM reale | ~16.5 GB/s | **[I]** | valore a cui si fermano `omp_blocked4` (4, 6 thr) e `omp_blocked` (12 thr) |
| B₁, banda DRAM di un solo core | 10.9 GB/s (9.4–14.8) | **[M]** | `omp_simd` 1 thr: 72.3 GB / 6.62 s. Dipende dal pattern: `omp_soa` 12.9, `omp_aosoa` 14.8 |
| velocità in convoglio, per thread | 13 GB/s | **[M]** | `omp_simd`, 2–8 thread |
| L2 → L1 sostenuta | ~16 B/ciclo | **[I]** | dedotta dal kernel SoA "in L2" (preliminare) |

**Perché B₁ è molto sotto B_peak.** Un core tiene in volo al massimo ~10–12 cache
miss (line fill buffer). Per la legge di Little, banda = richieste in volo ×
dimensione / latenza = 12 × 64 B / 70 ns ≈ 11 GB/s. Più core = più miss in volo,
finché satura il canale di memoria.

## 3. I tre tetti

**Tetto di calcolo** = (catene di dipendenza indipendenti) × (elementi per
istruzione) × 3 flop ÷ (latenza in cicli) × f, con il massimo imposto dalle porte.

| kernel | catene | flop/ciclo | GFLOP/s per core a 4.2 GHz |
|---|---|---|---|
| scalare in ordine (`seq`, `omp_naive`): 8 `vaddss` in fila per 8 elementi | 1 × 1 elemento / 4 cicli | 0.75 | 3.15 |
| 1 accumulatore ymm (`omp_simd`, `omp_blocked`) | 1 × 8 / 4 | 6 | 25.2 |
| 4 accumulatori (`omp_blocked4`) | 4 × 8 / 4 = 8 elem/ciclo | 24 | 100.8 |
| 8 accumulatori (`omp_soa`, `omp_aosoa`) | limitato dalle porte | 24 | 100.8 |
| **picco**: sub + FMA = 2 µop per 8 elementi, 2 porte | — | **24** | **100.8** (605 su 6 core) |

Con 4 accumulatori latenza e porte danno lo stesso limite (8 µop FP per iterazione
su 2 porte = 4 cicli = latenza della FMA): `omp_blocked4` è *bilanciato*, più
accumulatori da soli non darebbero nulla — servirebbe meno µop per elemento
(riuso anche delle righe di training: kernel MR × NR, fase D).

*Avvertenza sul tetto a 1 accumulatore.* Il loop sulle feature dura solo 98
iterazioni e ogni distanza riparte da zero: la finestra out-of-order (224 µop ≈ 45
iterazioni) fa partire la distanza successiva mentre la precedente sta finendo.
Il limite "4 cicli per iterazione" è quindi un limite **morbido** per `omp_simd` e
`omp_blocked`. Verifica: esperimento con `--dim 7840` (run_extra.sh, parte 9).

**Tetto di memoria** = banda del livello da cui arrivano i dati × intensità
aritmetica (flop per byte letto da quel livello).

| variante | byte da DRAM per elemento | intensità (flop/B, DRAM) |
|---|---|---|
| `omp_naive`, `omp_simd`, `omp_soa`, `omp_aosoa` | 4 (ogni query rilegge tutto) | 0.75 |
| `omp_blocked`, `omp_blocked4` (QB query per tile) | 4 / QB = 0.25 | 0.75 × QB = **12** |

A 0.75 flop/B il tetto DRAM è 8.2 GFLOP/s con un core (B₁) e 12–18 con tutti
(B_max–B_peak). A 12 flop/B è 131 (un core) e 198–282 (tutti).

**Tetto "convoglio"** (non c'è nei libri, c'è nei dati): con `schedule(static)`
tutti i thread scorrono le stesse righe quasi nello stesso istante; una riga viene
letta dalla DRAM una volta e poi servita dalla L3 agli altri. Il traffico DRAM non
cresce con i thread, quindi il tetto di memoria si moltiplica per p. Vale finché i
thread restano entro i 12 MB della L3 l'uno dall'altro (review §1-B).

## 4. Modello a due termini per le varianti a blocchi

Una tile viene percorsa QB/R volte (R = query per passata: 1 per `omp_blocked`, 4
per `omp_blocked4`). La **prima** passata trova le righe in DRAM e va alla velocità
della memoria; le altre le trovano in L2 e vanno alla velocità del calcolo:

    T(p) ≈ D(QB) / B(p)  +  (1 − R/QB) · W / (p · R_core)

con D(QB) = 72.3 GB / QB, B(1) = B₁, B(p) = min(p · B₁, B_max), R_core = tetto di
calcolo per core. Se D/B domina, aggiungere thread non serve: serve più riuso (QB).

## 5. Previsto e misurato (dati validi, 384 query, `static`)

GFLOP/s; scarto = (misurato − previsto) / previsto.

| variante | thr | limite previsto | previsto | misurato | scarto | commento |
|---|---|---|---|---|---|---|
| `omp_naive` | 1 | calcolo (catena scalare) | 3.15 [I] | 2.85 [M] | −10% | dentro l'incertezza su f; coerente con f ≈ 3.8 GHz |
| `omp_naive` | 6 | calcolo, *se* in convoglio | 18.9 | 17.5 | −7% | senza convoglio servirebbero 23.3 GB/s ≈ B_peak: sarebbe al limite della memoria |
| `omp_naive` | 12 | memoria, flussi indipendenti | 12.4–17.6 | 19.8 | +12% sul massimo | convoglio rotto; un po' di condivisione residua |
| `omp_simd` | 1 | memoria (B₁) | 8.2 [C] | 8.2 | — | è la misura che definisce B₁: calibrazione, non previsione |
| `omp_simd` | 6 | convoglio: 6 × 13 GB/s × 0.75 | 58.5 [C] | 59.3 | +1% | idem. Senza convoglio: 12–18 (ed è il valore dei run "anomali": 15.8) |
| `omp_simd` | 12 | memoria, flussi indipendenti | 12.4–17.6 | 18.8 | +7% sul massimo | |
| `omp_soa` | 1 | memoria (B₁) | 8.2 | 9.65 | **+18%** | 784 flussi paralleli: più miss in volo di un flusso solo? B₁ dipende dal pattern |
| `omp_aosoa` | 1 | memoria (B₁) | 8.2 | 11.1 | **+35%** | come sopra; e il tempo include ~0.12 s di trasposta |
| `omp_blocked` | 1 | due termini, R_core = 25.2 | 22.3 | 26.5 | **+19%** | il tetto a 1 accumulatore è superato: finestra out-of-order (§3) e/o f > 4.2 |
| `omp_blocked` | 6 | due termini | 89–103 | 120.7 | **+17–36%** | stesso motivo, più condivisione delle tile in L3 |
| `omp_blocked4` | 1 | due termini, R_core = 100.8 | 66.3 | 60.8 | −8% | metà del tempo è D/B₁ = 0.41 s: memory-bound per metà già a 1 thread |
| `omp_blocked4` | 6 | due termini, B = B_max–B_peak | 159–209 | 196.9 | dentro | D/B = 0.19–0.27 s su 0.275 misurati: satura la DRAM |
| `omp_blocked4` | 12 | come a 6 (SMT non aggiunge porte) | 159–209 | 316.5 | **+51%** | impossibile senza condivisione delle tile in L3 (26 GB/s apparenti > B_peak) |

Scarti > 20% e cosa li spiegherebbe:
- **`omp_blocked` (+19…36%)**: o la frequenza a 1 thread è ≥ 4.4 GHz *e* il limite a
  1 accumulatore non è rigido, o entrambe. Decide la sonda di frequenza +
  l'esperimento `--dim 7840`.
- **`omp_aosoa`/`omp_soa` a 1 thread (+18…35%)**: B₁ non è una costante della
  macchina; dipende da quante richieste indipendenti il kernel riesce a tenere in
  volo. Nota: nei dati validi a 1 thread SoA **non** perde contro AoS (5.61 s contro
  6.62 s), come invece dicevano le prove preliminari (2.17 contro 1.75 s): perde da
  4 thread in su.
- **`omp_blocked4` a 12 thread (+51%)**: il modello non contiene il convoglio per
  le tile. È il limite principale del modello, da dichiarare.

## 6. Previsioni registrate prima della misura (run_extra.sh)

Parametri: f = 4.2 GHz [I], B₁ = 10.9 GB/s [M]. Da confrontare in
`report/predictions_vs_measured.md` insieme a quelle di `PREVISIONI_run_rest.md`.

**P-1. Sweep di QB a 1 thread** (TB=32). `omp_blocked4`: tutto in DRAM con QB=4.

| QB | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| D/B₁ [s] | 1.66 | 0.83 | 0.41 | 0.21 |
| `omp_blocked4` T [s] | **1.66** | **1.10** | 0.82 (mis. 0.89) | **0.68** |
| `omp_blocked` T [s], R_core = 31 [C] | **2.97** | **2.36** | 2.05 [C] | **1.90** |

Test del modello: T contro 1/QB deve essere una retta; la pendenza è 72.3 GB / B₁.

**P-2. Avvio sfalsato** (`omp_simd_stagger`, 6 thread): senza condivisione il tempo
passa da 0.91 s a **3.1–4.4 s** (72.3 GB a B_max–B_peak, + 0.1 s di attese).
Se resta sotto 1.5 s l'ipotesi del convoglio è **falsa**.

**P-3. Norme parallele** (`omp_blocked4_pn`, coseno, 12 thread): la parte seriale è
47 M elementi a 4 cicli = 0.045 s a 4.2 GHz. Guadagno atteso ≈ **0.04 s** su
~0.21 s (Amdahl: frazione seriale ≈ 20% a 12 thread, < 0.3% a 1 thread).

**P-4. Banda DRAM** (`hwprobe membw`): 1 thread **11–15 GB/s**; saturazione a
**16–19 GB/s** entro 4 thread. Se satura sopra 21 GB/s, B_max [I] è sbagliato e
`omp_blocked4` a 4–6 thread non era al limite della memoria.

**P-5. Frequenza** (`hwprobe freq`): 1 thread **4.3–4.8 GHz**, 6 thread
**3.8–4.2 GHz**; con AVX2 uguale o fino a 0.2 GHz in meno.

**P-6. Righe 10 volte più lunghe** (`--dim 7840`, in L2, 1 thread, `omp_simd`):
GFLOP/s **più bassi** che con 784 (meno sovrapposizione tra distanze), verso
6 flop/ciclo × f ≈ 25–27.

## 7. Roofline

`python3 scripts/plot_roofline.py results/raw/threads_*.csv [--hwprobe results/raw/hwprobe_*.csv]`
→ `results/plots/roofline.png`. Asse x: intensità aritmetica rispetto alla DRAM
(0.75 per le varianti che rileggono tutto, 12 per quelle a blocchi con QB=16).
Lettura attesa: `omp_naive` sotto il suo tetto di calcolo basso, lontano dalla
diagonale di memoria (compute-bound); `omp_simd` a 1 thread *sulla* diagonale B₁ e a
6 thread **sopra** la diagonale B_peak (possibile solo col convoglio: è il punto da
commentare a voce); `omp_blocked` e `omp_blocked4` a destra, sotto i tetti
orizzontali di calcolo, con `omp_blocked4` a 6 thread che tocca la diagonale.

## 8. Limiti del modello (da dire prima che li dica l'esaminatore)

- f non è misurata; in una VM WSL2 non si leggono né frequenza né contatori
  hardware (niente `perf`): tutte le affermazioni su cache miss e TLB sono dedotte
  dai tempi, non osservate.
- Il convoglio è descritto, non modellato: non sappiamo prevedere quando si rompe.
- B₁ varia del ±20% con il pattern di accesso e nel tempo (deriva di `omp_simd` a
  1 thread: 6.7 → 5.2 s nello stesso processo).
- Top-k, voto e overhead OpenMP sono trascurati (inserimenti attesi ≈ k·ln(n/k) ≈
  47 su 60000 distanze per k=5).
