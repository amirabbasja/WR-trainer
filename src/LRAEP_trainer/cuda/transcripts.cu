#include <cuda_runtime.h>
#include <stdint.h>
namespace {
__device__ __forceinline__ uint32_t rotate13(uint32_t value) {
    return (value << 13) | (value >> 19);
}
template <int BM, int BN>
__global__ void transcripts_dp4a(const int* __restrict__ a,
                                 const int* __restrict__ bt,
                                 int m, int n, int k_words,
                                 uint32_t* __restrict__ output) {
    constexpr int THREADS = BM * BN / 16;
    __shared__ int shared_a[BM][33];
    __shared__ int shared_b[BN][33];
    const int tid = threadIdx.x;
    const int lane = tid & 3;
    const int group = tid >> 2;
    const int tile_row = group / (BN / 8);
    const int tile_col = group % (BN / 8);
    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;
    const int col_first = tile_col * 8 + lane + ((tile_col & 4) ? 4 : 0);
    const int col_second = col_first ^ 4;
    int cells[16] = {};
    uint32_t word0 = 0, word1 = 0, word2 = 0, word3 = 0;
    for (int start = 0, slab = 0; start < k_words; start += 32, ++slab) {
        for (int element = tid; element < BM * 32; element += THREADS) {
            const int row = element >> 5;
            const int inner = element & 31;
            const int global_row = row_base + row;
            shared_a[row][inner] = global_row < m
                ? a[global_row * k_words + start + inner] : 0;
        }
        for (int element = tid; element < BN * 32; element += THREADS) {
            const int col = element >> 5;
            const int inner = element & 31;
            const int global_col = col_base + col;
            shared_b[col][inner] = global_col < n
                ? bt[global_col * k_words + start + inner] : 0;
        }
        __syncthreads();
        #pragma unroll 4
        for (int inner = 0; inner < 32; ++inner) {
            const int b0 = shared_b[col_first][inner];
            const int b1 = shared_b[col_second][inner];
            #pragma unroll
            for (int row = 0; row < 8; ++row) {
                const int av = shared_a[tile_row * 8 + row][inner];
                cells[row * 2] = __dp4a(av, b0, cells[row * 2]);
                cells[row * 2 + 1] = __dp4a(av, b1, cells[row * 2 + 1]);
            }
        }
        uint32_t reduction = 0;
        #pragma unroll
        for (int cell = 0; cell < 16; ++cell)
            reduction ^= static_cast<uint32_t>(cells[cell]);
        reduction ^= __shfl_xor_sync(0xffffffffu, reduction, 1, 4);
        reduction ^= __shfl_xor_sync(0xffffffffu, reduction, 2, 4);
        if ((slab & 3) == lane) {
            switch ((slab >> 2) & 3) {
                case 0: word0 = rotate13(word0) ^ reduction; break;
                case 1: word1 = rotate13(word1) ^ reduction; break;
                case 2: word2 = rotate13(word2) ^ reduction; break;
                case 3: word3 = rotate13(word3) ^ reduction; break;
            }
        }
        __syncthreads();
    }
    const int output_row = row_base / 8 + tile_row;
    const int output_col = col_base / 8 + tile_col;
    if (output_row < m / 8 && output_col < n / 8) {
        uint32_t* words = output + (output_row * (n / 8) + output_col) * 16;
        words[lane] = word0;
        words[lane + 4] = word1;
        words[lane + 8] = word2;
        words[lane + 12] = word3;
    }
}
template <int BM, int BN>
void launch(const int* a, const int* bt, int m, int n, int k,
            uint32_t* output, cudaStream_t stream) {
    constexpr int THREADS = BM * BN / 16;
    const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    transcripts_dp4a<BM, BN><<<grid, THREADS, 0, stream>>>(a, bt, m, n, k / 4, output);
}
template <int BM, int BN>
__global__ void transcripts_dp4a_wide(const int* __restrict__ a,
                                      const int* __restrict__ bt,
                                      int m, int n, int k_words,
                                      uint32_t* __restrict__ output) {
    constexpr int GROUP = 2;
    constexpr int THREADS = BM * BN / 32;
    constexpr int COLS = 4;
    constexpr int WORDS = 8;
    __shared__ int shared_a[BM][33];
    __shared__ int shared_b[BN][33];
    const int tid = threadIdx.x;
    const int lane = tid & (GROUP - 1);
    const int group = tid / GROUP;
    const int tile_row = group / (BN / 8);
    const int tile_col = group % (BN / 8);
    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;
    const int swizzle = (tile_col & 4) | ((tile_col & 8) >> 2);
    const int col_first = lane ^ swizzle;
    int cells[8 * COLS] = {};
    uint32_t words[WORDS] = {};
    for (int start = 0, slab = 0; start < k_words; start += 32, ++slab) {
        for (int element = tid; element < BM * 32; element += THREADS) {
            const int row = element >> 5;
            const int inner = element & 31;
            const int global_row = row_base + row;
            shared_a[row][inner] = global_row < m
                ? a[global_row * k_words + start + inner] : 0;
        }
        for (int element = tid; element < BN * 32; element += THREADS) {
            const int col = element >> 5;
            const int inner = element & 31;
            const int global_col = col_base + col;
            shared_b[col][inner] = global_col < n
                ? bt[global_col * k_words + start + inner] : 0;
        }
        __syncthreads();
        #pragma unroll 4
        for (int inner = 0; inner < 32; ++inner) {
            int bv[COLS];
            #pragma unroll
            for (int col = 0; col < COLS; ++col)
                bv[col] = shared_b[tile_col * 8 + (col_first ^ (GROUP * col))][inner];
            #pragma unroll
            for (int row = 0; row < 8; ++row) {
                const int av = shared_a[tile_row * 8 + row][inner];
                #pragma unroll
                for (int col = 0; col < COLS; ++col)
                    cells[row * COLS + col] = __dp4a(av, bv[col], cells[row * COLS + col]);
            }
        }
        uint32_t reduction = 0;
        #pragma unroll
        for (int cell = 0; cell < 8 * COLS; ++cell)
            reduction ^= static_cast<uint32_t>(cells[cell]);
        #pragma unroll
        for (int delta = 1; delta < GROUP; delta *= 2)
            reduction ^= __shfl_xor_sync(0xffffffffu, reduction, delta, GROUP);
        if ((slab & (GROUP - 1)) == lane) {
            const int slot = (slab / GROUP) & (WORDS - 1);
            #pragma unroll
            for (int word = 0; word < WORDS; ++word)
                if (slot == word)
                    words[word] = rotate13(words[word]) ^ reduction;
        }
        __syncthreads();
    }
    const int output_row = row_base / 8 + tile_row;
    const int output_col = col_base / 8 + tile_col;
    if (output_row < m / 8 && output_col < n / 8) {
        uint32_t* destination = output + (output_row * (n / 8) + output_col) * 16;
        #pragma unroll
        for (int word = 0; word < WORDS; ++word)
            destination[lane + GROUP * word] = words[word];
    }
}
template <int BM, int BN>
void launch_wide(const int* a, const int* bt, int m, int n, int k,
                 uint32_t* output, cudaStream_t stream) {
    constexpr int THREADS = BM * BN / 32;
    const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    transcripts_dp4a_wide<BM, BN><<<grid, THREADS, 0, stream>>>(
        a, bt, m, n, k / 4, output);
}
}
extern "C" int LRAEP_transcripts_i8(const signed char* a, const signed char* bt,
                                    int m, int n, int k, unsigned int* output,
                                    void* stream, int variant) {
    if (!a || !bt || !output || (reinterpret_cast<uintptr_t>(a) & 3)
        || (reinterpret_cast<uintptr_t>(bt) & 3) || (reinterpret_cast<uintptr_t>(output) & 3)
        || m <= 0 || n <= 0 || m > 4096 || n > 4096
        || m % 8 || n % 8 || k < 128 || k > 65536 || k % 128)
        return static_cast<int>(cudaErrorInvalidValue);
    const int* packed_a = reinterpret_cast<const int*>(a);
    const int* packed_bt = reinterpret_cast<const int*>(bt);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (variant) {
        case 0:
            if (m <= 256 && n <= 256)
                launch<64, 32>(packed_a, packed_bt, m, n, k, output, cuda_stream);
            else if (m >= 768 && n >= 768)
                launch_wide<128, 64>(packed_a, packed_bt, m, n, k, output, cuda_stream);
            else
                launch<64, 64>(packed_a, packed_bt, m, n, k, output, cuda_stream);
            break;
        case 4: launch<64, 64>(packed_a, packed_bt, m, n, k, output, cuda_stream); break;
        case 1: launch<32, 32>(packed_a, packed_bt, m, n, k, output, cuda_stream); break;
        case 2: launch<32, 64>(packed_a, packed_bt, m, n, k, output, cuda_stream); break;
        case 3: launch<64, 32>(packed_a, packed_bt, m, n, k, output, cuda_stream); break;
        case 5: launch<128, 64>(packed_a, packed_bt, m, n, k, output, cuda_stream); break;
        case 6: launch<64, 128>(packed_a, packed_bt, m, n, k, output, cuda_stream); break;
        case 7: launch_wide<128, 64>(packed_a, packed_bt, m, n, k, output, cuda_stream); break;
        default: return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(cudaGetLastError());
}
