#pragma once
// std::vector with 64-byte aligned storage (cache-line and AVX friendly).
#include <cstddef>
#include <new>
#include <vector>

template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    template <typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    T* allocate(std::size_t n) {
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(Alignment)));
    }
    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t(Alignment));
    }
    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;
