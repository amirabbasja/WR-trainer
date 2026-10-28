#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>
#include "blake3.cuh"
namespace {
constexpr uint32_t kRank = 128;
__device__ __forceinline__ uint32_t uniform_word(uint32_t random) {
    const uint32_t masked = random & 0x3f3f3f3fu;
    return (masked & 0x1f1f1f1fu) | ((~masked & 0x20202020u) * 7u);
}
__global__ void noise_factors_kernel(
    uint32_t a_draws, uint32_t b_draws, uint32_t permutation_draws,
    uint32_t k, const uint32_t* key_a, const uint32_t* key_b,
    int8_t* al, int8_t* brt, uint32_t* a_indices, uint32_t* b_indices
) {
    const uint32_t job = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t uniform_draws = a_draws + b_draws;
    if (job >= uniform_draws + 2u * permutation_draws) return;
    bool is_b;
    bool permutation;
    uint32_t draw;
    if (job < a_draws) {
        is_b = false;
        permutation = false;
        draw = job;
    } else if (job < uniform_draws) {
        is_b = true;
        permutation = false;
        draw = job - a_draws;
    } else if (job < uniform_draws + permutation_draws) {
        is_b = false;
        permutation = true;
        draw = job - uniform_draws;
    } else {
        is_b = true;
        permutation = true;
        draw = job - uniform_draws - permutation_draws;
    }
    uint32_t message[16] = {};
    message[permutation ? 1 : 0] = draw + 1u;
    message[8] = is_b ? 0x65745f42u : 0x65745f41u;
    message[9] = 0x726f736eu;
    uint32_t digest[8];
    LRAEP_blake3_keyed64(message, is_b ? key_b : key_a, digest);
    if (!permutation) {
        uint32_t* output = reinterpret_cast<uint32_t*>(is_b ? brt : al) + draw * 8u;
        #pragma unroll
        for (uint32_t word = 0; word < 8; ++word) {
            output[word] = uniform_word(digest[word]);
        }
    } else {
        uint2* indices = reinterpret_cast<uint2*>(is_b ? b_indices : a_indices);
        #pragma unroll
        for (uint32_t word = 0; word < 8; ++word) {
            const uint32_t column = draw * 8u + word;
            if (column < k) {
                const uint32_t value = digest[word];
                const uint32_t first = value & (kRank - 1u);
                const uint32_t second = first ^ (1u + __umulhi(kRank - 1u, value));
                indices[column] = make_uint2(first, second);
            }
        }
    }
}
__global__ void apply_noise_kernel(
    const int8_t* a, const int8_t* bt, uint32_t a_size, uint32_t total,
    uint32_t k, const int8_t* al, const int8_t* brt,
    const uint32_t* a_indices, const uint32_t* b_indices,
    int8_t* out_a, int8_t* out_bt
) {
    const uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += stride) {
        const bool is_b = index >= a_size;
        const uint32_t offset = is_b ? index - a_size : index;
        const uint32_t row = offset / k;
        const uint32_t common = offset - row * k;
        const uint32_t* indices = (is_b ? b_indices : a_indices) + 2u * common;
        const int8_t* uniform = (is_b ? brt : al) + row * kRank;
        const int8_t* signal = is_b ? bt : a;
        const int value = static_cast<int>(signal[offset])
                        + static_cast<int>(uniform[indices[0]])
                        - static_cast<int>(uniform[indices[1]]);
        uint8_t* output = reinterpret_cast<uint8_t*>(is_b ? out_bt : out_a);
        output[offset] = static_cast<uint8_t>(value);
    }
}
}
extern "C" int LRAEP_noise_factors(
    int m, int n, int k, const uint32_t* key_a, const uint32_t* key_b,
    int8_t* al, int8_t* brt, uint32_t* a_indices, uint32_t* b_indices,
    void* stream
) {
    if (m <= 0 || n <= 0 || k <= 0 || !key_a || !key_b || !al || !brt
        || !a_indices || !b_indices) return static_cast<int>(cudaErrorInvalidValue);
    const uint64_t a_draws = 4ull * static_cast<uint64_t>(m);
    const uint64_t b_draws = 4ull * static_cast<uint64_t>(n);
    const uint64_t permutation_draws = (static_cast<uint64_t>(k) + 7ull) / 8ull;
    const uint64_t jobs = a_draws + b_draws + 2ull * permutation_draws;
    if (jobs > INT_MAX) return static_cast<int>(cudaErrorInvalidValue);
    constexpr uint32_t threads = 128;
    const uint32_t blocks = static_cast<uint32_t>((jobs + threads - 1) / threads);
    noise_factors_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint32_t>(a_draws), static_cast<uint32_t>(b_draws),
        static_cast<uint32_t>(permutation_draws), static_cast<uint32_t>(k),
        key_a, key_b, al, brt, a_indices, b_indices
    );
    return static_cast<int>(cudaGetLastError());
}
extern "C" int LRAEP_apply_noise(
    const int8_t* a, const int8_t* bt, int m, int n, int k,
    const int8_t* al, const int8_t* brt,
    const uint32_t* a_indices, const uint32_t* b_indices,
    int8_t* out_a, int8_t* out_bt, void* stream
) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !bt || !al || !brt
        || !a_indices || !b_indices || !out_a || !out_bt)
        return static_cast<int>(cudaErrorInvalidValue);
    const uint64_t a_size = static_cast<uint64_t>(m) * static_cast<uint64_t>(k);
    const uint64_t total = (static_cast<uint64_t>(m) + static_cast<uint64_t>(n))
                         * static_cast<uint64_t>(k);
    if (total > INT_MAX) return static_cast<int>(cudaErrorInvalidValue);
    constexpr uint32_t threads = 256;
    const uint64_t needed = (total + threads - 1) / threads;
    const uint32_t blocks = static_cast<uint32_t>(needed < 65535 ? needed : 65535);
    apply_noise_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        a, bt, static_cast<uint32_t>(a_size), static_cast<uint32_t>(total),
        static_cast<uint32_t>(k), al, brt, a_indices, b_indices, out_a, out_bt
    );
    return static_cast<int>(cudaGetLastError());
}
