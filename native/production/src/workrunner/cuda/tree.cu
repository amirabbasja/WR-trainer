#include <cuda_runtime.h>
#include <stdint.h>
#include "digest_core.cuh"
template <bool Constant>
__global__ void wr_tree_leaf_kernel(const unsigned char* data, size_t bytes, uint32_t fill,
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
            if constexpr (Constant) {
                if (offset + 4 <= bytes) {
                    value = fill;
                } else if (offset < bytes) {
                    value = fill & (0xffffffffu >> (8 * (4 - (bytes - offset))));
                }
            } else if (offset + 4 <= bytes && (reinterpret_cast<uintptr_t>(data + offset) & 3) == 0) {
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
        wr_digest_compress(message, cv, uint64_t(leaf), 64, flags, next);
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = next[i];
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) output[leaf * 8 + i] = cv[i];
}
__global__ void wr_tree_parent_kernel(const uint32_t* input, int count,
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
    wr_digest_compress(message, key, 0, 64, 16 | 4 | (root ? 8 : 0), digest);
    #pragma unroll
    for (int i = 0; i < 8; ++i) output[parent * 8 + i] = digest[i];
}
extern "C" int wr_tree_leaves(const unsigned char* data, size_t bytes,
                                   const uint32_t* key, uint32_t* output,
                                   void* stream) {
    if (!data || !key || !output || !bytes) return int(cudaErrorInvalidValue);
    int leaves = int((bytes + 1023) / 1024);
    wr_tree_leaf_kernel<false><<<(leaves + 127) / 128, 128, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(data, bytes, 0, key, output, leaves);
    return int(cudaGetLastError());
}
extern "C" int wr_tree_constant_leaves(size_t bytes, uint8_t value, const uint32_t* key,
                                        uint32_t* output, void* stream) {
    if (!key || !output || !bytes) return int(cudaErrorInvalidValue);
    int leaves = int((bytes + 1023) / 1024);
    const uint32_t fill = static_cast<uint32_t>(value) * 0x01010101u;
    wr_tree_leaf_kernel<true><<<(leaves + 127) / 128, 128, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(nullptr, bytes, fill, key, output, leaves);
    return int(cudaGetLastError());
}
extern "C" int wr_tree_parents(const uint32_t* input, int count,
                                    const uint32_t* key, uint32_t* output,
                                    int root, void* stream) {
    if (!input || !key || !output || count < 2 || (root && count != 2))
        return int(cudaErrorInvalidValue);
    int parents = (count + 1) / 2;
    wr_tree_parent_kernel<<<(parents + 127) / 128, 128, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(input, count, key, output, root);
    return int(cudaGetLastError());
}

__global__ void wr_tree_tail_root_kernel(
    size_t bytes, uint32_t fill, const uint32_t* key, const uint32_t* tail,
    const uint32_t* siblings, uint32_t* output
) {
    // A seed changes only the final leaf; all siblings remain header-constant.
    const uint32_t leaves = uint32_t((bytes + 1023) / 1024);
    const uint32_t leaf = leaves - 1;
    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = key[i];
    for (int block = 0; block < 16; ++block) {
        uint32_t message[16];
        #pragma unroll
        for (int word = 0; word < 16; ++word) {
            const size_t offset = size_t(leaf) * 1024 + block * 64 + word * 4;
            uint32_t value = 0;
            #pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const size_t position = offset + byte;
                uint32_t next = 0;
                if (position >= bytes - 16 && position < bytes) {
                    const unsigned index = unsigned(position - (bytes - 16));
                    next = (tail[index / 4] >> (8 * (index % 4))) & 255u;
                } else if (position < bytes) {
                    next = fill & 255u;
                }
                value |= next << (8 * byte);
            }
            message[word] = value;
        }
        const uint32_t flags = 16 | (block == 0 ? 1 : 0) |
                               (block == 15 ? 2 : 0) |
                               (leaves == 1 && block == 15 ? 8 : 0);
        uint32_t next[8];
        wr_digest_compress(message, cv, uint64_t(leaf), 64, flags, next);
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = next[i];
    }
    // The last node has a left sibling exactly when this level has even count.
    // An unmatched last node is promoted verbatim, matching wr_tree_parents.
    for (uint32_t count = leaves, level = 0; count > 1;
         count = (count + 1) / 2, ++level) {
        if ((count & 1u) == 0) {
            uint32_t message[16], next[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                message[i] = siblings[level * 8 + i];
                message[8 + i] = cv[i];
            }
            wr_digest_compress(message, key, 0, 64,
                               16 | 4 | (count == 2 ? 8 : 0), next);
            #pragma unroll
            for (int i = 0; i < 8; ++i) cv[i] = next[i];
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) output[i] = cv[i];
}

extern "C" int wr_tree_tail_root(
    size_t bytes, uint8_t value, const uint32_t* key, const uint32_t* tail,
    const uint32_t* siblings, uint32_t* output, void* stream
) {
    // Arbitrary partial leaves are supported if all sixteen tail bytes are in
    // the last leaf. WR sizes are multiples of 128 and always satisfy this.
    if (!key || !tail || !siblings || !output || bytes < 16 || bytes > (size_t(1) << 32) ||
        (bytes % 1024 && bytes % 1024 < 16))
        return int(cudaErrorInvalidValue);
    const uint32_t fill = uint32_t(value) * 0x01010101u;
    wr_tree_tail_root_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        bytes, fill, key, tail, siblings, output);
    return int(cudaGetLastError());
}
