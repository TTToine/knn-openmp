Implementa una nuova variante k-NN: $ARGUMENTS

Procedura:
1. Rileggi la roadmap in CLAUDE.md e il codice di `src/knn_omp.cpp` per partire
   dall'ultima variante valida. Prima di scrivere codice, esponi in breve il piano:
   quale singola idea di ottimizzazione introduce la variante e quale effetto ti
   aspetti (su tempo, cache, vettorizzazione), collegandolo ai concetti del corso.
2. Dichiara la funzione in `include/knn.hpp`, implementala in `src/knn_omp.cpp`,
   registrala in `src/variants.cpp`. Non toccare `knn_seq.cpp`.
3. Compila e lancia `ctest --test-dir build --output-on-failure`. Se ci sono mismatch,
   indaga la causa prima di qualsiasi tolleranza.
4. Esegui un confronto rapido (meno di 1 minuto) con la variante precedente su
   `--n-test 100` e riporta i tempi.
5. Aggiungi una voce in `report/notes.md` con ipotesi, comando e risultato.
6. Spiegami il codice nuovo riga per riga nelle parti OpenMP/SIMD, e proponi il
   messaggio di commit.
