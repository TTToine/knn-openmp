#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "aligned_vector.hpp"

// Dense labelled dataset. Features are stored row-major: sample i occupies
// x[i*d .. i*d + d). This is the "AoS-like" layout (one sample = one contiguous row).
struct Dataset {
    std::size_t n = 0;   // number of samples
    std::size_t d = 0;   // number of features (784 for MNIST)
    int num_classes = 0;
    AlignedVector<float> x;          // n * d values
    std::vector<std::uint8_t> y;     // n labels

    const float* row(std::size_t i) const { return x.data() + i * d; }
};

// Load an MNIST-format IDX pair (uncompressed). Pixels are scaled to [0, 1].
// max_samples = 0 means "load everything".
Dataset load_idx(const std::string& images_path, const std::string& labels_path,
                 std::size_t max_samples = 0);

// Gaussian blobs: one random centre per class, samples = centre + noise.
// centre_seed fixes the classes (use the same value for train and test),
// sample_seed fixes the samples (use different values for train and test).
Dataset make_synthetic(std::size_t n, std::size_t d, int num_classes,
                       std::uint32_t centre_seed, std::uint32_t sample_seed);

// Concatenate the dataset with itself `factor` times (to reach >= 10 s sequential).
Dataset replicate(const Dataset& ds, std::size_t factor);
