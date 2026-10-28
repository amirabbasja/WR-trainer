#include <cuda_runtime.h>
#include <stdint.h>
#include "blake3.cuh"
__global__ void LRAEP_merkle_leaf_kernel(const unsigned char* data, size_t bytes,
                                        const uint32_t* key, uint32_t* output,
                                        int leaves) {
    int leaf = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf >= leaves) return;
    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = key[i];
    for (int block = 0; block < 16; ++block) {
        uint32_t message[16];
        #pragma unroll
        for (int word = 0; word < 16; ++word) {
            size_t offset = size_t(leaf) * 1024 + block * 64 + word * 4;
            uint32_t value = 0;
            if (offset + 4 <= bytes && (reinterpret_cast<uintptr_t>(data + offset) & 3) == 0) {
                value = *reinterpret_cast<const uint32_t*>(data + offset);
            } else {
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte)
                    if (offset + byte < bytes)
                        value |= uint32_t(data[offset + byte]) << (8 * byte);
            }
            message[word] = value;
        }
        uint32_t flags = 16 | (block == 0 ? 1 : 0) | (block == 15 ? 2 : 0);
        if (leaves == 1 && block == 15) flags |= 8;
        uint32_t next[8];
        LRAEP_blake3_compress(message, cv, uint64_t(leaf), 64, flags, next);
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = next[i];
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) output[leaf * 8 + i] = cv[i];
}
__global__ void LRAEP_merkle_parent_kernel(const uint32_t* input, int count,
                                          const uint32_t* key, uint32_t* output,
                                          int root) {
    int parent = blockIdx.x * blockDim.x + threadIdx.x;
    if (parent >= (count + 1) / 2) return;
    if (parent * 2 + 1 == count) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) output[parent * 8 + i] = input[parent * 16 + i];
        return;
    }
    uint32_t message[16], digest[8];
    #pragma unroll
    for (int i = 0; i < 16; ++i) message[i] = input[parent * 16 + i];
    LRAEP_blake3_compress(message, key, 0, 64, 16 | 4 | (root ? 8 : 0), digest);
    #pragma unroll
    for (int i = 0; i < 8; ++i) output[parent * 8 + i] = digest[i];
}
extern "C" int LRAEP_merkle_leaves(const unsigned char* data, size_t bytes,
                                   const uint32_t* key, uint32_t* output,
                                   void* stream) {
    if (!data || !key || !output || !bytes) return int(cudaErrorInvalidValue);
    int leaves = int((bytes + 1023) / 1024);
    LRAEP_merkle_leaf_kernel<<<(leaves + 127) / 128, 128, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(data, bytes, key, output, leaves);
    return int(cudaGetLastError());
}
extern "C" int LRAEP_merkle_parents(const uint32_t* input, int count,
                                    const uint32_t* key, uint32_t* output,
                                    int root, void* stream) {
    if (!input || !key || !output || count < 2 || (root && count != 2))
        return int(cudaErrorInvalidValue);
    int parents = (count + 1) / 2;
    LRAEP_merkle_parent_kernel<<<(parents + 127) / 128, 128, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(input, count, key, output, root);
    return int(cudaGetLastError());
}
