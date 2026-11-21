#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>
#include "digest_core.cuh"
namespace {
constexpr uint32_t kRank = 128;
__device__ __forceinline__ uint32_t uniform_word(uint32_t random) {
    const uint32_t masked = random & 0x3f3f3f3fu;
    return (masked & 0x1f1f1f1fu) | ((~masked & 0x20202020u) * 7u);
}
__device__ __forceinline__ void noise_factor(bool is_b, bool permutation,
                                            uint32_t draw, uint32_t k,
                                            const uint32_t* key, int8_t* uniform,
                                            uint32_t* indices) {
    uint32_t message[16] = {};
    message[permutation ? 1 : 0] = draw + 1u;
    message[8] = is_b ? 0x65745f42u : 0x65745f41u;
    message[9] = 0x726f736eu;
    uint32_t digest[8];
    wr_digest_keyed64(message, key, digest);
    if (!permutation) {
        uint32_t* output = reinterpret_cast<uint32_t*>(uniform) + draw * 8u;
        #pragma unroll
        for (uint32_t word = 0; word < 8; ++word) {
            output[word] = uniform_word(digest[word]);
        }
    } else {
        uint2* output = reinterpret_cast<uint2*>(indices);
        #pragma unroll
        for (uint32_t word = 0; word < 8; ++word) {
            const uint32_t column = draw * 8u + word;
            if (column < k) {
                const uint32_t value = digest[word];
                const uint32_t first = value & (kRank - 1u);
                const uint32_t second = first ^ (1u + __umulhi(kRank - 1u, value));
                output[column] = make_uint2(first, second);
            }
        }
    }
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
    noise_factor(is_b, permutation, draw, k, is_b ? key_b : key_a,
                 is_b ? brt : al, is_b ? b_indices : a_indices);
}
__global__ void single_noise_factors_kernel(
    uint32_t uniform_draws, uint32_t jobs, uint32_t k, bool is_b, const uint32_t* key,
    int8_t* uniform, uint32_t* indices
) {
    const uint32_t job = blockIdx.x * blockDim.x + threadIdx.x;
    if (job >= jobs) return;
    const bool permutation = job >= uniform_draws;
    noise_factor(is_b, permutation, permutation ? job - uniform_draws : job,
                 k, key, uniform, indices);
}
__global__ void apply_noise_kernel(
    const int8_t* a, const int8_t* bt, uint32_t a_size, uint32_t total,
    uint32_t k, const int8_t* al, const int8_t* brt,
    const uint32_t* a_indices, const uint32_t* b_indices,
    int8_t* out_a, int8_t* out_bt
) {
    const uint32_t stride = blockDim.x * gridDim.x;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void apply_constant_noise_kernel(
    uint32_t total, uint32_t k, int value, const int8_t* uniform,
    const uint2* indices, uint8_t* output
) {
    const uint32_t stride = blockDim.x * gridDim.x;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += stride) {
        const uint32_t offset = static_cast<uint32_t>(index);
        const uint32_t row = offset / k;
        const uint2 pair = indices[offset - row * k];
        const int8_t* row_uniform = uniform + row * kRank;
        output[offset] = static_cast<uint8_t>(value + row_uniform[pair.x] - row_uniform[pair.y]);
    }
}
}
extern "C" int wr_noise_factors(
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
extern "C" int wr_apply_noise(
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
    if (total > UINT32_MAX) return static_cast<int>(cudaErrorInvalidValue);
    constexpr uint32_t threads = 256;
    const uint64_t needed = (total + threads - 1) / threads;
    const uint32_t blocks = static_cast<uint32_t>(needed < 65535 ? needed : 65535);
    apply_noise_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        a, bt, static_cast<uint32_t>(a_size), static_cast<uint32_t>(total),
        static_cast<uint32_t>(k), al, brt, a_indices, b_indices, out_a, out_bt
    );
    return static_cast<int>(cudaGetLastError());
}
extern "C" int wr_noise_constant_operand(
    int rows, int k, int value, int is_b, const uint32_t* key, int8_t* uniform,
    uint32_t* indices, uint8_t* output, void* stream
) {
    if (rows <= 0 || k <= 0 || !key || !uniform || !indices || !output)
        return static_cast<int>(cudaErrorInvalidValue);
    const uint64_t jobs = 4ull * rows + (static_cast<uint64_t>(k) + 7ull) / 8ull;
    const uint64_t total = static_cast<uint64_t>(rows) * k;
    if (jobs > INT_MAX || total > UINT32_MAX) return static_cast<int>(cudaErrorInvalidValue);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    constexpr uint32_t factor_threads = 128;
    single_noise_factors_kernel<<<(jobs + factor_threads - 1) / factor_threads,
                                 factor_threads, 0, cuda_stream>>>(
        4u * rows, static_cast<uint32_t>(jobs), static_cast<uint32_t>(k), is_b != 0,
        key, uniform, indices);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    constexpr uint32_t apply_threads = 256;
    const uint64_t needed = (total + apply_threads - 1) / apply_threads;
    const uint32_t blocks = static_cast<uint32_t>(needed < 65535 ? needed : 65535);
    apply_constant_noise_kernel<<<blocks, apply_threads, 0, cuda_stream>>>(
        static_cast<uint32_t>(total), static_cast<uint32_t>(k), value, uniform,
        reinterpret_cast<const uint2*>(indices), output);
    return static_cast<int>(cudaGetLastError());
}
