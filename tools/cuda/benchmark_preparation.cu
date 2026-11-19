// The offline benchmark uses the runtime's keyed generator for the rank-128 factors.
#include "preparation.cu"

namespace {
constexpr unsigned kBlockRows = 128;
constexpr unsigned kRowsPerTile = kBlockRows / 2;
constexpr unsigned kWordsPerTile = 128;
constexpr unsigned kFactorThreads = 128;
constexpr unsigned kExpandThreads = 512;

__global__ void expand_packed(unsigned cols, int base, const uint32_t* uniform,
                              const uint2* indices, uint32_t* output) {
    // The extra shared column lets adjacent threads transpose an input row
    // without writing the same bank. Each warp later reads the same noise
    // index from 32 different rows, then writes consecutive packed words.
    __shared__ uint32_t table[kRank / 4][kRowsPerTile + 1];
    __shared__ uint2 column_pairs[kWordsPerTile * 4];
    const unsigned block = blockIdx.y / 2;
    const unsigned half = blockIdx.y % 2;
    const unsigned first_word = blockIdx.x * kWordsPerTile;
    const unsigned words_per_row = cols / 4;
    for (unsigned i = threadIdx.x; i < kRowsPerTile * (kRank / 4); i += blockDim.x) {
        const unsigned packed_row = i / (kRank / 4);
        const unsigned word = i % (kRank / 4);
        // One half contains four rows from every eight-row transcript tile.
        const unsigned row = block * kBlockRows + (packed_row / 4) * 8 + half * 4 + packed_row % 4;
        table[word][packed_row] = uniform[row * (kRank / 4) + word];
    }
    for (unsigned i = threadIdx.x; i < kWordsPerTile * 4; i += blockDim.x) {
        const unsigned col = first_word * 4 + i;
        if (col < cols) column_pairs[i] = indices[col];
    }
    __syncthreads();

    for (unsigned i = threadIdx.x; i < kRowsPerTile * kWordsPerTile; i += blockDim.x) {
        const unsigned packed_row = i % kRowsPerTile;
        const unsigned tile_word = i / kRowsPerTile;
        const unsigned word = first_word + tile_word;
        if (word >= words_per_row) break;
        uint32_t packed = 0;
        #pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            const uint2 pair = column_pairs[tile_word * 4 + byte];
            const uint32_t a = table[pair.x / 4][packed_row] >> (8 * (pair.x % 4));
            const uint32_t b = table[pair.y / 4][packed_row] >> (8 * (pair.y % 4));
            packed |= ((base + a - b) & 0xffu) << (8 * byte);
        }
        output[(block * words_per_row + word) * kBlockRows + half * kRowsPerTile + packed_row] = packed;
    }
}
}  // namespace

extern "C" int benchmark_noise(int rows, int cols, int base, int right, const uint32_t* key,
                               int8_t* uniform, uint32_t* indices, uint32_t* output, void* stream) {
    if (rows < 128 || rows > 4096 || rows % kBlockRows || cols < 128 || cols > 65536 ||
        cols % 128 || !key || !uniform || !indices || !output || (reinterpret_cast<uintptr_t>(output) & 15))
        return static_cast<int>(cudaErrorInvalidValue);
    const cudaStream_t current = static_cast<cudaStream_t>(stream);
    const unsigned draws = 4u * rows;
    const unsigned jobs = draws + static_cast<unsigned>(cols) / 8;
    single_noise_factors_kernel<<<(jobs + kFactorThreads - 1) / kFactorThreads, kFactorThreads, 0, current>>>(
        draws, jobs, cols, right != 0, key, uniform, indices);

    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    const dim3 grid((cols / 4 + kWordsPerTile - 1) / kWordsPerTile, rows / kRowsPerTile);
    expand_packed<<<grid, kExpandThreads, 0, current>>>(cols, base, reinterpret_cast<const uint32_t*>(uniform),
        reinterpret_cast<const uint2*>(indices), output);
    return static_cast<int>(cudaGetLastError());
}
