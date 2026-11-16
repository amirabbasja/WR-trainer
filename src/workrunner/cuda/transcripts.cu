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
template <int BM, int BN, int TW>
__global__ void transcripts_dp4a_wide(const int* __restrict__ a,
                                      const int* __restrict__ bt,
                                      int m, int n, int k_words,
                                      uint32_t* __restrict__ output) {
    constexpr int GROUP = 2;
    constexpr int THREADS = BM * BN * GROUP / (8 * TW);
    constexpr int COLS = TW / GROUP;
    constexpr int WORDS = 8;
    __shared__ int shared_a[BM][33];
    __shared__ int shared_b[BN][33];
    const int tid = threadIdx.x;
    const int lane = tid & (GROUP - 1);
    const int group = tid / GROUP;
    const int tile_row = group / (BN / TW);
    const int tile_col = group % (BN / TW);
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
                bv[col] = shared_b[tile_col * TW + (col_first ^ (GROUP * col))][inner];
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
    const int output_col = col_base / TW + tile_col;
    if (output_row < m / 8 && output_col < n / TW) {
        uint32_t* destination = output + (output_row * (n / TW) + output_col) * 16;
        #pragma unroll
        for (int word = 0; word < WORDS; ++word)
            destination[lane + GROUP * word] = words[word];
    }
}
template <int BM, int BN, int TW = 8>
void launch_wide(const int* a, const int* bt, int m, int n, int k,
                 uint32_t* output, cudaStream_t stream) {
    constexpr int THREADS = BM * BN * 2 / (8 * TW);
    const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    transcripts_dp4a_wide<BM, BN, TW><<<grid, THREADS, 0, stream>>>(
        a, bt, m, n, k / 4, output);
}
template <int BM, int BN>
__global__ void transcripts_dp4a_vector(const int* __restrict__ a,
                                        const int* __restrict__ bt,
                                        int m, int n, int k_words,
                                        uint32_t* __restrict__ output) {
    constexpr int TILE_H = 8;
    constexpr int TILE_W = 16;
    constexpr int ROW_LANES = 2;
    constexpr int COL_LANES = 2;
    constexpr int GROUP = ROW_LANES * COL_LANES;
    constexpr int THREADS = BM * BN * GROUP / (TILE_H * TILE_W);
    constexpr int ROWS = TILE_H / ROW_LANES;
    constexpr int COLS = TILE_W / COL_LANES;
    constexpr int WORDS = 16 / GROUP;
    constexpr int TILE_COLS = BN / TILE_W;
    __shared__ int4 shared_a[BM][8];
    __shared__ int4 shared_b[BN][8];
    const int4* a_vectors = reinterpret_cast<const int4*>(a);
    const int4* b_vectors = reinterpret_cast<const int4*>(bt);
    const int k_vectors = k_words / 4;

    const int tid = threadIdx.x;
    const int lane = tid % GROUP;
    const int tile = tid / GROUP;
    const int tile_row = tile / TILE_COLS;
    const int tile_col = tile % TILE_COLS;
    const int row_lane = lane / COL_LANES;
    const int col_lane = lane % COL_LANES;
    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;
    const int local_row = tile_row * TILE_H + row_lane;
    const int local_col = tile_col * TILE_W + col_lane;
    // Distribute the vectors read by neighboring lanes across shared-memory banks.
    const int a_swizzle = (tile_row * ROW_LANES + row_lane) & 7;
    const int b_swizzle = (tile_col * COL_LANES + col_lane) & 7;

    // Four lanes cover one 8x16 tile. Keep each dot product across all rank-128 slabs.
    int cells[ROWS][COLS] = {};
    uint32_t transcript[WORDS] = {};

    for (int start = 0, slab = 0; start < k_vectors; start += 8, ++slab) {
        #pragma unroll
        for (int element = tid; element < BM * 8; element += THREADS) {
            const int row = element / 8;
            const int inner = element % 8;
            const int global_row = row_base + row;
            const int swizzle = ((row / TILE_H) * ROW_LANES + row % ROW_LANES) & 7;
            shared_a[row][inner ^ swizzle] = global_row < m
                ? a_vectors[global_row * k_vectors + start + inner] : make_int4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int element = tid; element < BN * 8; element += THREADS) {
            const int col = element / 8;
            const int inner = element % 8;
            const int global_col = col_base + col;
            const int swizzle = ((col / TILE_W) * COL_LANES + col % COL_LANES) & 7;
            shared_b[col][inner ^ swizzle] = global_col < n
                ? b_vectors[global_col * k_vectors + start + inner] : make_int4(0, 0, 0, 0);
        }
        __syncthreads();

        #pragma unroll 2
        for (int inner = 0; inner < 8; ++inner) {
            int4 bv[COLS];
            #pragma unroll
            for (int col = 0; col < COLS; ++col)
                bv[col] = shared_b[local_col + col * COL_LANES][inner ^ b_swizzle];
            #pragma unroll
            for (int row = 0; row < ROWS; ++row) {
                const int4 av = shared_a[local_row + row * ROW_LANES][inner ^ a_swizzle];
                #pragma unroll
                for (int col = 0; col < COLS; ++col) {
                    int value = cells[row][col];
                    value = __dp4a(av.x, bv[col].x, value);
                    value = __dp4a(av.y, bv[col].y, value);
                    value = __dp4a(av.z, bv[col].z, value);
                    cells[row][col] = __dp4a(av.w, bv[col].w, value);
                }
            }
        }

        uint32_t reduction = 0;
        #pragma unroll
        for (int row = 0; row < ROWS; ++row) {
            #pragma unroll
            for (int col = 0; col < COLS; ++col)
                reduction ^= static_cast<uint32_t>(cells[row][col]);
        }
        #pragma unroll
        for (int delta = 1; delta < GROUP; delta *= 2)
            reduction ^= __shfl_xor_sync(0xffffffffu, reduction, delta, GROUP);

        if (slab % GROUP == lane) {
            const int slot = (slab / GROUP) % WORDS;
            #pragma unroll
            for (int word = 0; word < WORDS; ++word)
                if (slot == word)
                    transcript[word] = rotate13(transcript[word]) ^ reduction;
        }
        __syncthreads();
    }

    const int output_row = row_base / TILE_H + tile_row;
    const int output_col = col_base / TILE_W + tile_col;
    if (output_row < m / TILE_H && output_col < n / TILE_W) {
        uint32_t* destination = output + (output_row * (n / TILE_W) + output_col) * 16;
        #pragma unroll
        for (int word = 0; word < WORDS; ++word)
            destination[lane + GROUP * word] = transcript[word];
    }
}

template <int BM, int BN>
void launch_vector(const int* a, const int* bt, int m, int n, int k,
                   uint32_t* output, cudaStream_t stream) {
    constexpr int THREADS = BM * BN / 32;
    const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    transcripts_dp4a_vector<BM, BN><<<grid, THREADS, 0, stream>>>(
        a, bt, m, n, k / 4, output);
}
}
extern "C" int wr_transcripts_i8(const signed char* a, const signed char* bt,
                                    int m, int n, int k, unsigned int* output,
                                    void* stream, int variant) {
    if (!a || !bt || !output || (reinterpret_cast<uintptr_t>(a) & 3)
        || (reinterpret_cast<uintptr_t>(bt) & 3) || (reinterpret_cast<uintptr_t>(output) & 3)
        || m <= 0 || n <= 0 || m > 4096 || n > 4096
        || m % 8 || n % (variant == 8 ? 16 : 8) || k < 128 || k > 65536 || k % 128)
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
        case 8:
            if (((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(bt)) & 15) == 0)
                launch_vector<64, 64>(packed_a, packed_bt, m, n, k, output, cuda_stream);
            else
                launch_wide<64, 64, 16>(packed_a, packed_bt, m, n, k, output, cuda_stream);
            break;
        default: return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(cudaGetLastError());
}
