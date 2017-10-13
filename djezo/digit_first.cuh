namespace digit_first {

#define BLAKE_IV0 0x6a09e667f3bcc908
#define BLAKE_IV1 0xbb67ae8584caa73b
#define BLAKE_IV2 0x3c6ef372fe94f82b
#define BLAKE_IV3 0xa54ff53a5f1d36f1
#define BLAKE_IV4 0x510e527fade68241 // 0x510e527fade682d1 ^ (128 + 16)
#define BLAKE_IV5 0x9b05688c2b3e6c1f
#define BLAKE_IV6 0xe07c265404be4294 // 0x1f83d9abfb41bd6b ^ 0xffffffffffffffff
#define BLAKE_IV7 0x5be0cd19137e2179

__device__ __forceinline__ uint2 ROR2(const uint2 a, const int offset) 
{
	uint2 result;
	{
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	return result;
}

__device__ __forceinline__ uint2 SWAPUINT2(uint2 value) 
{
	return make_uint2(value.y, value.x);
}

__device__ __forceinline__ uint2 ROR24(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x2107);
	result.y = __byte_perm(a.y, a.x, 0x6543);
	return result;
}

__device__ __forceinline__ uint2 ROR16(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x1076);
	result.y = __byte_perm(a.y, a.x, 0x5432);
	return result;
}

__device__ __forceinline__ void G2(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d, const uint64_t& x, const uint64_t& y) 
{
	a = a + b + x;
	*(uint2*)&d = SWAPUINT2(*(uint2*)&d ^ *(uint2*)&a);
	c = c + d;
	*(uint2*)&b = ROR24(*(uint2*)&b ^ *(uint2*)&c);
	a = a + b + y;
	*(uint2*)&d = ROR16(*(uint2*)&d ^ *(uint2*)&a);
	c = c + d;
	*(uint2*)&b = ROR2(*(uint2*)&b ^ *(uint2*)&c, 63U);
}

typedef union
{
	uint64_t v[16];
	uint32_t v32[32];
	uint4 v128[8];
	blake2b_state blake_ctx;
}
DigitFirstState;

template <uint32_t RB, uint32_t SM>
__global__ void kernelDigitFirst(Equi<RB, SM>* equi, const DigitFirstState state)
{
	using namespace digit_first;

	const uint32_t block = blockIdx.x * blockDim.x + threadIdx.x;

	uint64_t m = (uint64_t)block << 32;

	union
	{
		uint64_t v[16];
		uint32_t v32[32];
		uint4 v128[8];
	};

	v128[0] = state.v128[0];
	v128[1] = state.v128[1];
	v128[2] = state.v128[2];
	v128[3] = state.v128[3];
	v128[4] = state.v128[4];
	v128[5] = state.v128[5];
	v128[6] = state.v128[6];
	v128[7] = state.v128[7];

	// mix 1
	G2(v[0], v[4], v[8], v[12], 0, m);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 2
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], m, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 3
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, m);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 4
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, m);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 5
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, m);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 6
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], m, 0);

	// mix 7
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], m, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 8
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, m);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 9
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], m, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 10
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], m, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 11
	G2(v[0], v[4], v[8], v[12], 0, m);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 12
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], m, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	v[0] ^= state.v[0] ^ v[8];
	v[1] ^= state.v[1] ^ v[9];
	v[2] ^= state.v[2] ^ v[10];
	v[3] ^= state.v[3] ^ v[11];
	v[4] ^= state.v[4] ^ v[12];
	v[5] ^= state.v[5] ^ v[13];
	v32[12] ^= state.v32[12] ^ v32[28];

	uint32_t bexor = __byte_perm(v32[0], 0, 0x4012); // first 20 bits
	uint32_t bucketid;
	asm("bfe.u32 %0, %1, 12, 12;" : "=r"(bucketid) : "r"(bexor));
	uint32_t slotp = atomicAdd(&equi->edata.nslots0[bucketid], 1);
	if (slotp < RB8_NSLOTS)
	{
		Slot* s = &equi->round0trees[bucketid][slotp];

		uint4 tt;
		tt.x = __byte_perm(v32[0], v32[1], 0x1234);
		tt.y = __byte_perm(v32[1], v32[2], 0x1234);
		tt.z = __byte_perm(v32[2], v32[3], 0x1234);
		tt.w = __byte_perm(v32[3], v32[4], 0x1234);
		*(uint4*)(&s->hash[0]) = tt;

		tt.x = __byte_perm(v32[4], v32[5], 0x1234);
		tt.y = __byte_perm(v32[5], v32[6], 0x1234);
		tt.z = 0;
		tt.w = block << 1;
		*(uint4*)(&s->hash[4]) = tt;
	}

	bexor = __byte_perm(v32[6], 0, 0x0123);
	asm("bfe.u32 %0, %1, 12, 12;" : "=r"(bucketid) : "r"(bexor));
	slotp = atomicAdd(&equi->edata.nslots0[bucketid], 1);
	if (slotp < RB8_NSLOTS)
	{
		Slot* s = &equi->round0trees[bucketid][slotp];

		uint4 tt;
		tt.x = __byte_perm(v32[6], v32[7], 0x2345);
		tt.y = __byte_perm(v32[7], v32[8], 0x2345);
		tt.z = __byte_perm(v32[8], v32[9], 0x2345);
		tt.w = __byte_perm(v32[9], v32[10], 0x2345);
		*(uint4*)(&s->hash[0]) = tt;

		tt.x = __byte_perm(v32[10], v32[11], 0x2345);
		tt.y = __byte_perm(v32[11], v32[12], 0x2345);
		tt.z = 0;
		tt.w = (block << 1) + 1;
		*(uint4*)(&s->hash[4]) = tt;
	}
}

static void setheader(blake2b_state *ctx, const char *header, const uint32_t headerLen, const char* nce, const uint32_t nonceLen)
{
	uint32_t le_N = WN;
	uint32_t le_K = WK;
	uint8_t personal[] = "ZcashPoW01230123";
	memcpy(personal + 8, &le_N, 4);
	memcpy(personal + 12, &le_K, 4);
	blake2b_param P[1];
	P->digest_length = HASHOUT;
	P->key_length = 0;
	P->fanout = 1;
	P->depth = 1;
	P->leaf_length = 0;
	P->node_offset = 0;
	P->node_depth = 0;
	P->inner_length = 0;
	memset(P->reserved, 0, sizeof(P->reserved));
	memset(P->salt, 0, sizeof(P->salt));
	memcpy(P->personal, (const uint8_t *)personal, 16);
	blake2b_init_param(ctx, P);
	blake2b_update(ctx, (const uint8_t *)header, headerLen);
	blake2b_update(ctx, (const uint8_t *)nce, nonceLen);
}

} // namespace digit_first

template <uint32_t RB, uint32_t SM>
__forceinline__ void DigitFirst(Equi<RB, SM>* equi,
	const char *tequihash_header, unsigned int tequihash_header_len,
	const char* nonce, unsigned int nonce_len)
{
	using namespace digit_first;

	DigitFirstState state;
	setheader(&state.blake_ctx, tequihash_header, tequihash_header_len, nonce, nonce_len);
	state.v[8] = BLAKE_IV0;
	state.v[9] = BLAKE_IV1;
	state.v[10] = BLAKE_IV2;
	state.v[11] = BLAKE_IV3;
	state.v[12] = BLAKE_IV4;
	state.v[13] = BLAKE_IV5;
	state.v[14] = BLAKE_IV6;
	state.v[15] = BLAKE_IV7;

	kernelDigitFirst<RB, SM> << <NBLOCKS / FD_THREADS, FD_THREADS >> >(equi, state);
}

