#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include "blake3.cuh"
namespace {
__global__ void hash_candidates_kernel(
    const unsigned int* transcripts, int count, const unsigned int* key,
    const unsigned int* target, unsigned int* match_indices,
    unsigned char* match_digests, unsigned int* match_count, int capacity
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(count)) {
        return;
    }
    uint32_t digest[8];
    LRAEP_blake3_keyed64(
        transcripts + static_cast<size_t>(index) * 16, key, digest
    );
    #pragma unroll
    for (int word = 7; word >= 0; --word) {
        if (digest[word] > target[word]) {
            return;
        }
        if (digest[word] < target[word]) {
            break;
        }
    }
    const unsigned int slot = atomicAdd(match_count, 1u);
    if (slot >= static_cast<unsigned int>(capacity)) {
        return;
    }
    match_indices[slot] = index;
    unsigned char* out = match_digests + static_cast<size_t>(slot) * 32;
    #pragma unroll
    for (int word = 0; word < 8; ++word) {
        const uint32_t value = digest[word];
        out[word * 4] = static_cast<unsigned char>(value);
        out[word * 4 + 1] = static_cast<unsigned char>(value >> 8);
        out[word * 4 + 2] = static_cast<unsigned char>(value >> 16);
        out[word * 4 + 3] = static_cast<unsigned char>(value >> 24);
    }
}
}
extern "C" int LRAEP_hash_candidates(
    const unsigned int* transcripts, int count, const unsigned int* key,
    const unsigned int* target, unsigned int* match_indices,
    unsigned char* match_digests, unsigned int* match_count, int capacity,
    void* stream
) {
    if (count < 0 || capacity < 0 || match_count == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (count > 0 && (transcripts == nullptr || key == nullptr || target == nullptr
        || (capacity > 0 && (match_indices == nullptr || match_digests == nullptr)))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaStream_t selected_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t error = cudaMemsetAsync(match_count, 0, sizeof(unsigned int), selected_stream);
    if (error != cudaSuccess || count == 0) {
        return static_cast<int>(error);
    }
    constexpr unsigned int threads = 128;
    const unsigned int blocks =
        (static_cast<unsigned int>(count) + threads - 1) / threads;
    hash_candidates_kernel<<<blocks, threads, 0, selected_stream>>>(
        transcripts, count, key, target, match_indices, match_digests,
        match_count, capacity
    );
    return static_cast<int>(cudaGetLastError());
}
