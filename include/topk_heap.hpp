#pragma once
// Top-k selection policy #2: binary MAX-HEAP (policy #1 is the sorted array
// with insertion sort, `TopK` in knn_common.hpp). Same interface: push(), vote().
//
// Why a max-heap for the k SMALLEST distances: the only element we ever need
// to look at or remove is the WORST of the k kept, and a max-heap keeps it at
// the root. A candidate is compared with the root (O(1), like the insertion
// sort's test against the last element); if it enters, it replaces the root
// and is sifted down in O(log k) instead of the O(k) shifts of insertion sort.
//
// Exactly the same result as `TopK`, including ties:
//   * TopK keeps the k smallest by (distance, arrival order): on equal
//     distance the earlier training sample wins (strict '<').
//   * Here every entry stores its arrival number `order`, and the heap is
//     ordered by the pair (distance, order). A newcomer is the latest arrival,
//     so it is smaller than the root only if its distance is strictly smaller:
//     the same acceptance test.
//   * vote() first sorts the k entries by (distance, order), which rebuilds the
//     very array TopK would hold, then votes in the same way.
#include <cstdint>

#include "knn.hpp"

struct TopKHeap {
    int k;
    int size = 0;
    std::uint32_t arrivals = 0;
    float dist[kMaxK];
    std::uint32_t order[kMaxK];
    std::uint8_t label[kMaxK];

    explicit TopKHeap(int k_) : k(k_) {}

    // true if entry a must be closer to the root than entry b (a is "worse")
    inline bool worse(int a, int b) const {
        return dist[a] > dist[b] || (dist[a] == dist[b] && order[a] > order[b]);
    }
    inline void swap_entries(int a, int b) {
        const float d = dist[a]; dist[a] = dist[b]; dist[b] = d;
        const std::uint32_t o = order[a]; order[a] = order[b]; order[b] = o;
        const std::uint8_t l = label[a]; label[a] = label[b]; label[b] = l;
    }

    inline void push(float d, std::uint8_t lab) {
        const std::uint32_t o = arrivals++;
        if (size == k) {
            if (!(d < dist[0])) return;          // not better than the worst kept: O(1) reject
            dist[0] = d; order[0] = o; label[0] = lab;   // replace the root...
            int i = 0;
            for (;;) {                           // ...and sift it down: O(log k)
                const int l = 2 * i + 1, r = l + 1;
                int w = i;
                if (l < size && worse(l, w)) w = l;
                if (r < size && worse(r, w)) w = r;
                if (w == i) break;
                swap_entries(i, w);
                i = w;
            }
        } else {                                 // heap not full yet: append and sift up
            int i = size++;
            dist[i] = d; order[i] = o; label[i] = lab;
            while (i > 0) {
                const int parent = (i - 1) / 2;
                if (!worse(i, parent)) break;
                swap_entries(i, parent);
                i = parent;
            }
        }
    }

    // Majority vote; ties broken in favour of the class of the nearest neighbour
    // (same rule as TopK::vote). Done once per query: an O(k^2) sort is fine.
    inline std::uint8_t vote() const {
        int idx[kMaxK];
        for (int j = 0; j < size; ++j) {         // insertion sort of the indices, ascending
            int pos = j;
            while (pos > 0 && worse(idx[pos - 1], j)) { idx[pos] = idx[pos - 1]; --pos; }
            idx[pos] = j;
        }
        int counts[kMaxClasses] = {0};
        int best = 0;
        for (int j = 0; j < size; ++j) {
            const int c = ++counts[label[idx[j]]];
            if (c > best) best = c;
        }
        for (int j = 0; j < size; ++j)
            if (counts[label[idx[j]]] == best) return label[idx[j]];
        return 0;
    }
};
