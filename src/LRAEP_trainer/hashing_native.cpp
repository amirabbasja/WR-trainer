#include <cstdint>
#include <cstring>

static constexpr uint32_t iv[4] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a};
static constexpr unsigned permutation[16] = {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8};
static uint32_t load(const unsigned char* p) { return uint32_t(p[0]) | uint32_t(p[1])<<8 | uint32_t(p[2])<<16 | uint32_t(p[3])<<24; }
static void store(unsigned char* p, uint32_t v) { for (unsigned i=0;i<4;i++) p[i]=v>>(8*i); }
static uint32_t rotate(uint32_t x, unsigned n) { return (x>>n)|(x<<(32-n)); }
static void mix(uint32_t* s, unsigned a,unsigned b,unsigned c,unsigned d,uint32_t x,uint32_t y) {
    s[a]+=s[b]+x; s[d]=rotate(s[d]^s[a],16); s[c]+=s[d]; s[b]=rotate(s[b]^s[c],12);
    s[a]+=s[b]+y; s[d]=rotate(s[d]^s[a],8); s[c]+=s[d]; s[b]=rotate(s[b]^s[c],7);
}
static void compress(const unsigned char* block,const unsigned char* cv,uint64_t counter,unsigned length,unsigned flags,unsigned char* out) {
    uint32_t s[16],m[16],next[16];
    for(unsigned i=0;i<8;i++)s[i]=load(cv+4*i);
    for(unsigned i=0;i<4;i++)s[8+i]=iv[i];
    s[12]=counter;s[13]=counter>>32;s[14]=length;s[15]=flags;
    for(unsigned i=0;i<16;i++)m[i]=load(block+4*i);
    for(unsigned r=0;r<7;r++) {
        mix(s,0,4,8,12,m[0],m[1]);mix(s,1,5,9,13,m[2],m[3]);mix(s,2,6,10,14,m[4],m[5]);mix(s,3,7,11,15,m[6],m[7]);
        mix(s,0,5,10,15,m[8],m[9]);mix(s,1,6,11,12,m[10],m[11]);mix(s,2,7,8,13,m[12],m[13]);mix(s,3,4,9,14,m[14],m[15]);
        for(unsigned i=0;i<16;i++)next[i]=m[permutation[i]];
        std::memcpy(m,next,sizeof(m));
    }
    for(unsigned i=0;i<8;i++)store(out+4*i,s[i]^s[i+8]);
}
extern "C" void lraep_hash_chunk(const unsigned char* data,unsigned length,const unsigned char* key,uint64_t index,unsigned root,unsigned char* out) {
    unsigned char cv[32],block[64];std::memcpy(cv,key,32);
    unsigned blocks=length?(length+63)/64:1;
    for(unsigned i=0;i<blocks;i++) {
        unsigned size=length>i*64?length-i*64:0;if(size>64)size=64;
        std::memset(block,0,64);if(size)std::memcpy(block,data+i*64,size);
        unsigned flags=16|(i==0?1:0)|(i+1==blocks?2:0)|(root&&i+1==blocks?8:0);
        compress(block,cv,index,size,flags,cv);
    }
    std::memcpy(out,cv,32);
}
extern "C" void lraep_hash_parent(const unsigned char* left,const unsigned char* right,const unsigned char* key,unsigned root,unsigned char* out) {
    unsigned char block[64];std::memcpy(block,left,32);std::memcpy(block+32,right,32);
    compress(block,key,0,64,16|4|(root?8:0),out);
}
