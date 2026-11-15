#include "dataset.hpp"

#include <algorithm>
#include <fstream>
#include <random>
#include <stdexcept>

namespace {

std::uint32_t read_be_u32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f) throw std::runtime_error("unexpected end of IDX file");
    return (std::uint32_t(b[0]) << 24) | (std::uint32_t(b[1]) << 16) |
           (std::uint32_t(b[2]) << 8) | std::uint32_t(b[3]);
}

}  // namespace

Dataset load_idx(const std::string& images_path, const std::string& labels_path,
                 std::size_t max_samples) {
    std::ifstream fi(images_path, std::ios::binary);
    std::ifstream fl(labels_path, std::ios::binary);
    if (!fi) throw std::runtime_error("cannot open " + images_path);
    if (!fl) throw std::runtime_error("cannot open " + labels_path);

    if (read_be_u32(fi) != 2051) throw std::runtime_error("bad magic in " + images_path);
    if (read_be_u32(fl) != 2049) throw std::runtime_error("bad magic in " + labels_path);

    const std::size_t n_img = read_be_u32(fi);
    const std::size_t rows = read_be_u32(fi);
    const std::size_t cols = read_be_u32(fi);
    const std::size_t n_lab = read_be_u32(fl);
    if (n_img != n_lab) throw std::runtime_error("image/label count mismatch");

    Dataset ds;
    ds.n = (max_samples > 0) ? std::min(max_samples, n_img) : n_img;
    ds.d = rows * cols;
    ds.x.resize(ds.n * ds.d);
    ds.y.resize(ds.n);

    std::vector<unsigned char> buf(ds.n * ds.d);
    fi.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    fl.read(reinterpret_cast<char*>(ds.y.data()), static_cast<std::streamsize>(ds.n));
    if (!fi || !fl) throw std::runtime_error("truncated IDX file");

    for (std::size_t i = 0; i < buf.size(); ++i) ds.x[i] = buf[i] / 255.0f;
    ds.num_classes = 1 + *std::max_element(ds.y.begin(), ds.y.end());
    return ds;
}

Dataset make_synthetic(std::size_t n, std::size_t d, int num_classes,
                       std::uint32_t centre_seed, std::uint32_t sample_seed) {
    std::mt19937 crng(centre_seed);
    std::uniform_real_distribution<float> centre_dist(0.0f, 0.5f);
    std::vector<float> centres(static_cast<std::size_t>(num_classes) * d);
    for (auto& c : centres) c = centre_dist(crng);

    // Noise is large on purpose: classes overlap, so accuracy is < 100% and k matters.
    std::mt19937 rng(sample_seed);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);

    Dataset ds;
    ds.n = n;
    ds.d = d;
    ds.num_classes = num_classes;
    ds.x.resize(n * d);
    ds.y.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const int c = class_dist(rng);
        ds.y[i] = static_cast<std::uint8_t>(c);
        const float* centre = centres.data() + static_cast<std::size_t>(c) * d;
        for (std::size_t j = 0; j < d; ++j) ds.x[i * d + j] = centre[j] + noise(rng);
    }
    return ds;
}

Dataset replicate(const Dataset& ds, std::size_t factor) {
    if (factor <= 1) return ds;
    Dataset out;
    out.n = ds.n * factor;
    out.d = ds.d;
    out.num_classes = ds.num_classes;
    out.x.reserve(out.n * out.d);
    out.y.reserve(out.n);
    for (std::size_t r = 0; r < factor; ++r) {
        out.x.insert(out.x.end(), ds.x.begin(), ds.x.end());
        out.y.insert(out.y.end(), ds.y.begin(), ds.y.end());
    }
    return out;
}
