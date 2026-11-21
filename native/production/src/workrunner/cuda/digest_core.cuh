#pragma once
#include <cstdint>
template <int Shift>
__device__ __forceinline__ uint32_t wr_digest_rotr(uint32_t value) {
    return (value >> Shift) | (value << (32 - Shift));
}
__device__ __forceinline__ void wr_digest_mix(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
    uint32_t x, uint32_t y
) {
    a += b + x;
    d = wr_digest_rotr<16>(d ^ a);
    c += d;
    b = wr_digest_rotr<12>(b ^ c);
    a += b + y;
    d = wr_digest_rotr<8>(d ^ a);
    c += d;
    b = wr_digest_rotr<7>(b ^ c);
}
template <int I0, int I1, int I2, int I3, int I4, int I5, int I6, int I7,
          int I8, int I9, int I10, int I11, int I12, int I13, int I14, int I15>
__device__ __forceinline__ void wr_digest_round(uint32_t* v, const uint32_t* m) {
    wr_digest_mix(v[0], v[4], v[8], v[12], m[I0], m[I1]);
    wr_digest_mix(v[1], v[5], v[9], v[13], m[I2], m[I3]);
    wr_digest_mix(v[2], v[6], v[10], v[14], m[I4], m[I5]);
    wr_digest_mix(v[3], v[7], v[11], v[15], m[I6], m[I7]);
    wr_digest_mix(v[0], v[5], v[10], v[15], m[I8], m[I9]);
    wr_digest_mix(v[1], v[6], v[11], v[12], m[I10], m[I11]);
    wr_digest_mix(v[2], v[7], v[8], v[13], m[I12], m[I13]);
    wr_digest_mix(v[3], v[4], v[9], v[14], m[I14], m[I15]);
}
__device__ __forceinline__ void wr_digest_compress(
    const uint32_t* message, const uint32_t* cv, uint64_t counter,
    uint32_t block_len, uint32_t flags, uint32_t* output
) {
    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        m[i] = message[i];
    }
    uint32_t v[16] = {
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        static_cast<uint32_t>(counter), static_cast<uint32_t>(counter >> 32),
        block_len, flags,
    };
    wr_digest_round<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>(v, m);
    wr_digest_round<2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8>(v, m);
    wr_digest_round<3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1>(v, m);
    wr_digest_round<10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6>(v, m);
    wr_digest_round<12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4>(v, m);
    wr_digest_round<9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7>(v, m);
    wr_digest_round<11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13>(v, m);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        output[i] = v[i] ^ v[i + 8];
    }
}
__device__ __forceinline__ void wr_digest_keyed64(
    const uint32_t* message, const uint32_t* key, uint32_t* output
) {
    wr_digest_compress(message, key, 0, 64, 27, output);
}
