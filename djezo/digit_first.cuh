namespace digit_first {

__device__ __constant__ const uint64_t blake_iv[] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

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

__device__ __forceinline__ void G2(uint64_t & a, uint64_t & b, uint64_t & c, uint64_t & d, uint64_t x, uint64_t y) 
{
	a = a + b + x;
	((uint2*)&d)[0] = SWAPUINT2(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
	c = c + d;
	((uint2*)&b)[0] = ROR24(((uint2*)&b)[0] ^ ((uint2*)&c)[0]);
	a = a + b + y;
	((uint2*)&d)[0] = ROR16(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
	c = c + d;
	((uint2*)&b)[0] = ROR2(((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
}

} // namespace digit_first

template <uint32_t RB, uint32_t SM, typename PACKER>
__global__ void DigitFirst(Equi<RB, SM>* eq, uint32_t nonce)
{
	using namespace digit_first;

	const uint32_t block = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ uint64_t hash_h[8];
	uint32_t* hash_h32 = (uint32_t*)hash_h;

	if (threadIdx.x < 16)
		hash_h32[threadIdx.x] = __ldca(&eq->blake_h32[threadIdx.x]);

	__syncthreads();

	uint64_t m = (uint64_t)block << 32 | (uint64_t)nonce;

	union
	{
		uint64_t v[16];
		uint32_t v32[32];
		uint4 v128[8];
	};

	v[0] = hash_h[0];
	v[1] = hash_h[1];
	v[2] = hash_h[2];
	v[3] = hash_h[3];
	v[4] = hash_h[4];
	v[5] = hash_h[5];
	v[6] = hash_h[6];
	v[7] = hash_h[7];
	v[8] = blake_iv[0];
	v[9] = blake_iv[1];
	v[10] = blake_iv[2];
	v[11] = blake_iv[3];
	v[12] = blake_iv[4] ^ (128 + 16);
	v[13] = blake_iv[5];
	v[14] = blake_iv[6] ^ 0xffffffffffffffff;
	v[15] = blake_iv[7];

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

	v[0] ^= hash_h[0] ^ v[8];
	v[1] ^= hash_h[1] ^ v[9];
	v[2] ^= hash_h[2] ^ v[10];
	v[3] ^= hash_h[3] ^ v[11];
	v[4] ^= hash_h[4] ^ v[12];
	v[5] ^= hash_h[5] ^ v[13];
	v32[12] ^= hash_h32[12] ^ v32[28];

	uint32_t bexor = __byte_perm(v32[0], 0, 0x4012); // first 20 bits
	uint32_t bucketid;
	asm("bfe.u32 %0, %1, 12, 12;" : "=r"(bucketid) : "r"(bexor));
	uint32_t slotp = atomicAdd(&eq->edata.nslots0[bucketid], 1);
	if (slotp < RB8_NSLOTS)
	{
		Slot* s = &eq->round0trees[bucketid][slotp];

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
	slotp = atomicAdd(&eq->edata.nslots0[bucketid], 1);
	if (slotp < RB8_NSLOTS)
	{
		Slot* s = &eq->round0trees[bucketid][slotp];

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

