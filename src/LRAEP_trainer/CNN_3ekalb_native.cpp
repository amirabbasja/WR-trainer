#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {
constexpr uint32_t IV[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};
constexpr unsigned PERMUTATION[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
constexpr unsigned CHUNK_START = 1, CHUNK_END = 2, PARENT = 4, ROOT = 8;
constexpr size_t CHUNK_BYTES = 1024, BLOCK_BYTES = 64, STACK_DEPTH = 54;

uint32_t load32(const unsigned char* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 |
           uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
void store32(unsigned char* p, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) p[i] = static_cast<unsigned char>(value >> (8 * i));
}
uint32_t rotate(uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32 - bits));
}
void mix(uint32_t* s, unsigned a, unsigned b, unsigned c, unsigned d, uint32_t x, uint32_t y) {
    s[a] += s[b] + x; s[d] = rotate(s[d] ^ s[a], 16);
    s[c] += s[d]; s[b] = rotate(s[b] ^ s[c], 12);
    s[a] += s[b] + y; s[d] = rotate(s[d] ^ s[a], 8);
    s[c] += s[d]; s[b] = rotate(s[b] ^ s[c], 7);
}
void round(uint32_t* s, const uint32_t* m) {
    mix(s, 0, 4, 8, 12, m[0], m[1]);
    mix(s, 1, 5, 9, 13, m[2], m[3]);
    mix(s, 2, 6, 10, 14, m[4], m[5]);
    mix(s, 3, 7, 11, 15, m[6], m[7]);
    mix(s, 0, 5, 10, 15, m[8], m[9]);
    mix(s, 1, 6, 11, 12, m[10], m[11]);
    mix(s, 2, 7, 8, 13, m[12], m[13]);
    mix(s, 3, 4, 9, 14, m[14], m[15]);
}
void compress(const unsigned char* block, const unsigned char* cv,
              uint64_t counter, unsigned length, unsigned flags, unsigned char* out) {
    uint32_t s[16], m[16], permuted[16], cv_words[8];
    for (unsigned i = 0; i < 8; ++i) s[i] = cv_words[i] = load32(cv + 4 * i);
    for (unsigned i = 0; i < 4; ++i) s[8 + i] = IV[i];
    s[12] = static_cast<uint32_t>(counter);
    s[13] = static_cast<uint32_t>(counter >> 32);
    s[14] = length;
    s[15] = flags;
    for (unsigned i = 0; i < 16; ++i) m[i] = load32(block + 4 * i);
    for (unsigned r = 0; r < 7; ++r) {
        round(s, m);
        if (r != 6) {
            for (unsigned i = 0; i < 16; ++i) permuted[i] = m[PERMUTATION[i]];
            std::memcpy(m, permuted, sizeof(m));
        }
    }
    for (unsigned i = 0; i < 8; ++i) {
        store32(out + 4 * i, s[i] ^ s[i + 8]);
        store32(out + 4 * (i + 8), s[i + 8] ^ cv_words[i]);
    }
}

struct Context {
    uint64_t chunk_index;
    unsigned char key[32];
    unsigned char cv[32];
    unsigned char block[BLOCK_BYTES];
    unsigned char stack[STACK_DEPTH][32];
    unsigned block_len;
    unsigned blocks_compressed;
    unsigned stack_len;
    unsigned flags;
};
static_assert(std::is_trivially_copyable<Context>::value, "Context must support bytewise copies");
static_assert(sizeof(Context) < 2048, "Hasher state must remain bounded");

struct Output {
    unsigned char cv[32];
    unsigned char block[BLOCK_BYTES];
    uint64_t counter;
    unsigned length;
    unsigned flags;
};

unsigned start_flag(const Context& state) {
    return state.blocks_compressed == 0 ? CHUNK_START : 0;
}
size_t chunk_length(const Context& state) {
    return state.blocks_compressed * BLOCK_BYTES + state.block_len;
}
Output chunk_output(const Context& state) {
    Output output{};
    std::memcpy(output.cv, state.cv, 32);
    std::memcpy(output.block, state.block, state.block_len);
    output.counter = state.chunk_index;
    output.length = state.block_len;
    output.flags = state.flags | start_flag(state) | CHUNK_END;
    return output;
}
void chaining_value(const Output& output, unsigned char* cv) {
    unsigned char words[64];
    compress(output.block, output.cv, output.counter, output.length, output.flags, words);
    std::memcpy(cv, words, 32);
}
Output parent_output(const unsigned char* left, const unsigned char* right,
                     const unsigned char* key, unsigned flags) {
    Output output{};
    std::memcpy(output.cv, key, 32);
    std::memcpy(output.block, left, 32);
    std::memcpy(output.block + 32, right, 32);
    output.counter = 0;
    output.length = BLOCK_BYTES;
    output.flags = flags | PARENT;
    return output;
}
void push_chunk(Context& state, unsigned char* cv, uint64_t total_chunks) {
    while ((total_chunks & 1) == 0) {
        Output output = parent_output(state.stack[--state.stack_len], cv, state.key, state.flags);
        chaining_value(output, cv);
        total_chunks >>= 1;
    }
    std::memcpy(state.stack[state.stack_len++], cv, 32);
}
void start_chunk(Context& state, uint64_t chunk_index) {
    state.chunk_index = chunk_index;
    std::memcpy(state.cv, state.key, 32);
    state.block_len = 0;
    state.blocks_compressed = 0;
}
}

extern "C" size_t cnn3_context_size() { return sizeof(Context); }
extern "C" size_t cnn3_context_align() { return alignof(Context); }

extern "C" void cnn3_init(void* context, const unsigned char* key32, unsigned flags) {
    auto& state = *static_cast<Context*>(context);
    std::memset(&state, 0, sizeof(state));
    std::memcpy(state.key, key32, 32);
    state.flags = flags;
    start_chunk(state, 0);
}

extern "C" void cnn3_update(void* context, const unsigned char* data, size_t length) {
    auto& state = *static_cast<Context*>(context);
    while (length != 0) {
        if (chunk_length(state) == CHUNK_BYTES) {
            unsigned char cv[32];
            chaining_value(chunk_output(state), cv);
            const uint64_t total_chunks = state.chunk_index + 1;
            push_chunk(state, cv, total_chunks);
            start_chunk(state, total_chunks);
        }
        if (state.block_len == BLOCK_BYTES) {
            unsigned char result[64];
            compress(state.block, state.cv, state.chunk_index, BLOCK_BYTES,
                     state.flags | start_flag(state), result);
            std::memcpy(state.cv, result, 32);
            ++state.blocks_compressed;
            state.block_len = 0;
        }
        const size_t take = std::min(length, BLOCK_BYTES - state.block_len);
        std::memcpy(state.block + state.block_len, data, take);
        state.block_len += static_cast<unsigned>(take);
        data += take;
        length -= take;
    }
}

extern "C" void cnn3_finalize(const void* context, uint64_t seek,
                              unsigned char* out, size_t length) {
    const auto& state = *static_cast<const Context*>(context);
    Output output = chunk_output(state);
    for (unsigned i = state.stack_len; i != 0; --i) {
        unsigned char cv[32];
        chaining_value(output, cv);
        output = parent_output(state.stack[i - 1], cv, state.key, state.flags);
    }
    uint64_t counter = seek / BLOCK_BYTES;
    size_t skip = seek % BLOCK_BYTES;
    while (length != 0) {
        unsigned char block[BLOCK_BYTES];
        compress(output.block, output.cv, counter, output.length, output.flags | ROOT, block);
        const size_t take = std::min(length, BLOCK_BYTES - skip);
        std::memcpy(out, block + skip, take);
        out += take;
        length -= take;
        ++counter;
        skip = 0;
    }
}

extern "C" void cnn3_compress(const unsigned char* block64, const unsigned char* cv32,
                              uint64_t counter, unsigned length, unsigned flags,
                              unsigned char* out64) {
    compress(block64, cv32, counter, length, flags, out64);
}

extern "C" void cnn3_chunk_cv(const unsigned char* data, unsigned length,
                              const unsigned char* key, uint64_t index,
                              unsigned root, unsigned char* out32) {
    Output output{};
    std::memcpy(output.cv, key, 32);
    output.counter = index;
    const unsigned blocks = length == 0 ? 1 : (length + BLOCK_BYTES - 1) / BLOCK_BYTES;
    for (unsigned i = 0; i < blocks; ++i) {
        const unsigned take = std::min<unsigned>(length - i * BLOCK_BYTES, BLOCK_BYTES);
        std::memset(output.block, 0, BLOCK_BYTES);
        if (take != 0) std::memcpy(output.block, data + i * BLOCK_BYTES, take);
        output.length = take;
        output.flags = 16 | (i == 0 ? CHUNK_START : 0) |
                       (i + 1 == blocks ? CHUNK_END : 0) |
                       (i + 1 == blocks && root ? ROOT : 0);
        chaining_value(output, output.cv);
    }
    std::memcpy(out32, output.cv, 32);
}

extern "C" void cnn3_parent_cv(const unsigned char* left, const unsigned char* right,
                               const unsigned char* key, unsigned root,
                               unsigned char* out32) {
    Output output = parent_output(left, right, key, 16);
    if (root) output.flags |= ROOT;
    chaining_value(output, out32);
}
