// Offline 8x8 transcript benchmark. A block covers 128 rows by 128 columns.
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int kBlock = 128;
constexpr int kThreads = 128;
constexpr int kVectors = kBlock / 4;  // An int4 holds one DP4A word from four rows.
constexpr int kSlab = 128 / 4;       // Packed words per rank-128 transcript step.

__global__ __launch_bounds__(kThreads, 1)
void packed_transcripts(const int4* __restrict__ a, const int4* __restrict__ bt,
                        int n, int k_words, uint32_t* __restrict__ output) {
    __shared__ int4 left[kSlab][kVectors];
    __shared__ int4 right[kSlab][kVectors];
    const int thread = threadIdx.x;
    const int warp = thread / 32;
    const int lane = thread % 32;
    // A warp computes four pairs of tile rows and eight tile columns.
    const int row_pair = (warp / 2) * 4 + lane / 8;
    const int tile_col = (warp % 2) * 8 + lane % 8;
    a += blockIdx.y * k_words * kVectors;
    bt += blockIdx.x * k_words * kVectors;

    int cells[16][8] = {};
    uint32_t words[2][16] = {};
    for (int start = 0; start < k_words; start += kSlab) {
        #pragma unroll
        for (int i = thread; i < kSlab * kVectors; i += kThreads) {
            left[i / kVectors][i % kVectors] = a[start * kVectors + i];
            right[i / kVectors][i % kVectors] = bt[start * kVectors + i];
        }
        __syncthreads();

        #pragma unroll 4
        for (int inner = 0; inner < kSlab; ++inner) {
            int av[16], bv[8];
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                const int4 v = left[inner][(group % 2) * 16 + row_pair * 2 + group / 2];
                av[group * 4] = v.x;
                av[group * 4 + 1] = v.y;
                av[group * 4 + 2] = v.z;
                av[group * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int group = 0; group < 2; ++group) {
                const int4 v = right[inner][group * 16 + tile_col];
                bv[group * 4] = v.x;
                bv[group * 4 + 1] = v.y;
                bv[group * 4 + 2] = v.z;
                bv[group * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int row = 0; row < 16; ++row) {
                #pragma unroll
                for (int col = 0; col < 8; ++col)
                    cells[row][col] = __dp4a(av[row], bv[col], cells[row][col]);
            }
        }

        const int slot = (start / kSlab) % 16;
        #pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
            uint32_t reduction = 0;
            #pragma unroll
            for (int row = 0; row < 8; ++row) {
                #pragma unroll
                for (int col = 0; col < 8; ++col)
                    reduction ^= static_cast<uint32_t>(cells[8 * tile + row][col]);
            }
            // Constant indices let the compiler keep all 16 rolling words in registers.
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i == slot) {
                    const uint32_t old = words[tile][i];
                    words[tile][i] = ((old << 13) | (old >> 19)) ^ reduction;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
        const int row = blockIdx.y * 16 + 2 * row_pair + tile;
        const int col = blockIdx.x * 16 + tile_col;
        #pragma unroll
        for (int i = 0; i < 16; ++i)
            output[(row * (n / 8) + col) * 16 + i] = words[tile][i];
    }
}
}  // namespace

extern "C" int benchmark_transcripts(const int4* a, const int4* bt,
                                     int m, int n, int k, uint32_t* output, void* stream) {
    if (!a || !bt || !output || m < kBlock || n < kBlock || m > 4096 || n > 4096 ||
        m % kBlock || n % kBlock || k < 128 || k > 65536 || k % 128 ||
        ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(bt)) & 15))
        return static_cast<int>(cudaErrorInvalidValue);
    const dim3 grid(n / kBlock, m / kBlock);
    packed_transcripts<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        a, bt, n, k / 4, output);
    return static_cast<int>(cudaGetLastError());
}
