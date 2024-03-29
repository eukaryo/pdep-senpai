﻿/*
MIT License
Copyright (c) 2020 Hiroki Takizawa
*/
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cassert>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>

alignas(64) uint8_t table_16_16[16][16];//x: a側で次に注目する4bitとする。 y:mask側で次に注目する4bitとする。このとき、[x][y]: resultに次に代入すべき4bit。
alignas(64) uint8_t table_16_16_popcount[16];//x: mask側で次に注目する4bitとする。このとき、[x]: a側で何ビットが代入されたか
//
alignas(64) uint8_t table_256_256[256][256];//x: a側で次に注目する8bitとする。 y:mask側で次に注目する8bitとする。このとき、[x][y]: resultに次に代入すべき8bit。
alignas(64) uint8_t table_256_256_popcount[256];//x: mask側で次に注目する8bitとする。このとき、[x]: a側で何ビットが代入されたか
//
alignas(64) uint8_t table_16_16_inv[16][16];//x: mask側で次に注目する4bitとする。 y:a側で次に注目する4bitとする。このとき、[x][y]: resultに次に代入すべき4bit。
alignas(64) uint8_t table_256_256_inv[256][256];//x: mask側で次に注目する8bitとする。 y:a側で次に注目する8bitとする。このとき、[x][y]: resultに次に代入すべき8bit。
//
alignas(64) __m128i table_16_pshufb[16];//x: a側で次に注目する4bitとする。y:mask側で次に注目する4bitとする。このとき、[x].u8[y]: resultに次に代入すべき4bit。


inline uint64_t pext_intrinsics(uint64_t a, uint64_t mask) {
	return _pext_u64(a, mask);
}

inline uint64_t popcount64_intrinsics(uint64_t x) {
	return _mm_popcnt_u64(x);
}

inline uint64_t popcount64_naive(uint64_t x) {
	uint64_t answer = 0;
	for (uint64_t i = 0; i < 64; ++i) {
		if (x & (1ULL << i))++answer;
	}
	return answer;
}

inline uint64_t pext_naive(uint64_t a, uint64_t mask) {
	uint64_t dst = 0, k = 0;

	for (uint64_t m = 0; m < 64; ++m) {
		if (mask & (1ULL << m)) {
			if (a & (1ULL << m)) {
				dst += (1ULL << k);
			}
			++k;
		}
	}

	return dst;
}

inline uint64_t pext_naive2(uint64_t a, uint64_t mask) {
	uint64_t dst = 0;

	for (uint64_t m = 0, k = 0; mask & a; mask /= 2, a /= 2, ++m) {
		const uint64_t flag = mask & 1ULL;
		dst += (flag & a) << k;
		k += flag;
	}

	return dst;
}

void init_tables() {

	//table_16_16
	for (uint64_t x = 0; x < 16; ++x)for (uint64_t y = 0; y < 16; ++y) {
		const uint64_t p = pext_naive(x, y);
		assert(p < 16ULL);
		table_16_16[x][y] = (uint8_t)p;
	}

	//table_16_16_popcount
	for (uint64_t x = 0; x < 16; ++x) {
		const uint64_t p = popcount64_naive(x);
		assert(p <= 4ULL);
		table_16_16_popcount[x] = (uint8_t)p;
	}

	//table_256_256
	for (uint64_t x = 0; x < 256; ++x)for (uint64_t y = 0; y < 256; ++y) {
		const uint64_t p = pext_naive(x, y);
		assert(p < 256ULL);
		table_256_256[x][y] = (uint8_t)p;
	}

	//table_256_256_popcount
	for (uint64_t x = 0; x < 256; ++x) {
		const uint64_t p = popcount64_naive(x);
		assert(p <= 8ULL);
		table_256_256_popcount[x] = (uint8_t)p;
	}

	//table_16_16_inv
	for (uint64_t x = 0; x < 16; ++x)for (uint64_t y = 0; y < 16; ++y) {
		table_16_16_inv[x][y] = table_16_16[y][x];
	}

	//table_256_256_inv
	for (uint64_t x = 0; x < 256; ++x)for (uint64_t y = 0; y < 256; ++y) {
		table_256_256_inv[x][y] = table_256_256[y][x];
	}

	for (uint64_t x = 0; x < 16; ++x) {
		uint8_t tmp[16];
		for (uint64_t y = 0; y < 16; ++y) {
			const uint64_t p = pext_naive(x, y);
			assert(p < 16ULL);
			tmp[y] = (uint8_t)p;
		}
		table_16_pshufb[x] = _mm_loadu_si128((__m128i*)tmp);
	}
}

template<int width, bool is_inv, bool use_popcount_intrinsics>inline uint64_t pext_table(uint64_t a, uint64_t mask) {

	static_assert(width == 16 || width == 256, "invalid width");

	uint64_t dst = 0;

	for (uint64_t m = 0; mask; mask /= width, a /= width) {
		const uint64_t b = mask % width;
		dst += ((uint64_t)(width == 16 ? (is_inv ? table_16_16_inv[b][a % width] : table_16_16[a % width][b]) : (is_inv ? table_256_256_inv[b][a % width] : table_256_256[a % width][b]))) << m;
		m += use_popcount_intrinsics ? popcount64_intrinsics(b) : width == 16 ? table_16_16_popcount[b] : table_256_256_popcount[b];
	}


	return dst;
}

inline uint64_t pext_table_16_16(uint64_t a, uint64_t mask) {
	return pext_table<16, false, false>(a, mask);
}

inline uint64_t pext_table_256_256(uint64_t a, uint64_t mask) {
	return pext_table<256, false, false>(a, mask);
}

inline uint64_t pext_table_16_16_inv(uint64_t a, uint64_t mask) {
	return pext_table<16, true, false>(a, mask);
}

inline uint64_t pext_table_256_256_inv(uint64_t a, uint64_t mask) {
	return pext_table<256, true, false>(a, mask);
}

inline uint64_t pext_table_16_16_pop(uint64_t a, uint64_t mask) {
	return pext_table<16, false, true>(a, mask);
}

inline uint64_t pext_table_256_256_pop(uint64_t a, uint64_t mask) {
	return pext_table<256, false, true>(a, mask);
}

inline uint64_t pext_table_16_16_inv_pop(uint64_t a, uint64_t mask) {
	return pext_table<16, true, true>(a, mask);
}

inline uint64_t pext_table_256_256_inv_pop(uint64_t a, uint64_t mask) {
	return pext_table<256, true, true>(a, mask);
}



//Hacker's Delight chapter 7
inline uint64_t pext_hd7(uint64_t x, uint64_t mask) noexcept {

	x &= mask;

	uint64_t mk = (~mask) << 1;

	for (int i = 0; i < 6; ++i) {
		uint64_t mp = mk ^ (mk << 1);
		mp = mp ^ (mp << 2);
		mp = mp ^ (mp << 4);
		mp = mp ^ (mp << 8);
		mp = mp ^ (mp << 16);
		mp = mp ^ (mp << 32);
		const uint64_t mv = mp & mask;
		mask = (mask ^ mv) | (mv >> (1ULL << i));
		const uint64_t t = x & mv;
		x = (x ^ t) | (t >> (1ULL << i));
		mk &= ~mp;
	}

	return x;
}




inline uint64_t pext_pshufb(uint64_t a, uint64_t mask) {

	__m128i mask_lo = _mm_set_epi64x(0xFFFF'FFFF'FFFF'FFFFULL, (mask & 0x0F0F'0F0F'0F0F'0F0FULL));
	__m128i mask_hi = _mm_set_epi64x(0xFFFF'FFFF'FFFF'FFFFULL, (mask & 0xF0F0'F0F0'F0F0'F0F0ULL) >> 4);

	__m128i bytemask = _mm_set_epi64x(0, 0xFF);
	__m128i answer = _mm_setzero_si128();

	uint64_t acc_pop = 0;

	for (int i = 0; i < 8; ++i) {

		const __m128i x_lo = _mm_shuffle_epi8(table_16_pshufb[a % 16], mask_lo);
		const uint64_t lo_pop = table_16_16_popcount[mask % 16];
		a /= 16;
		mask /= 16;
		const __m128i x_hi = _mm_shuffle_epi8(table_16_pshufb[a % 16], mask_hi);
		const uint64_t hi_pop = table_16_16_popcount[mask % 16];
		a /= 16;
		mask /= 16;

		//この時点で、x_loとx_hiの『下からi番目のバイトの下位4bit』に、『answerの下からi番目のバイトに入れるべき答え』がある。
		//その答えを詰めて入れる必要があり、詰め幅を決めるにはmaskのpopcountをする必要がある。

		mask_lo = _mm_srli_epi64(mask_lo, 8);
		mask_hi = _mm_srli_epi64(mask_hi, 8);

		answer = _mm_or_si128(answer, _mm_slli_epi64(_mm_and_si128(bytemask, _mm_or_si128(x_lo, _mm_slli_epi64(x_hi, (int)lo_pop))), (int)acc_pop));
		acc_pop += lo_pop + hi_pop;
	}

	return (uint64_t)_mm_cvtsi128_si64(answer);
}

inline uint64_t xorshift64(uint64_t x) {
	x = x ^ (x << 7);
	return x ^ (x >> 9);
}

#define DEF_BENCH_PEXT(name) \
void bench_pext_##name() {\
	std::cout << "Bench pext:"#name << std::endl;\
	uint64_t result = 0;\
	uint64_t a = 0x1111222233334444ULL;\
	uint64_t mask = 0x5555666677778888ULL;\
	auto start = std::chrono::system_clock::now();\
	for (int i = 0; i < (1 << 30); ++i) {\
		result ^= pext_##name(a, mask);\
		a = xorshift64(a);\
		mask = xorshift64(mask);\
	}\
	auto end = std::chrono::system_clock::now();\
	std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;\
	std::cout << "result (for validation) = " << std::to_string(result) << std::endl << std::endl;\
}

DEF_BENCH_PEXT(intrinsics)
DEF_BENCH_PEXT(naive)
DEF_BENCH_PEXT(naive2)
DEF_BENCH_PEXT(table_16_16)
DEF_BENCH_PEXT(table_256_256)
DEF_BENCH_PEXT(table_16_16_inv)
DEF_BENCH_PEXT(table_256_256_inv)
DEF_BENCH_PEXT(table_16_16_pop)
DEF_BENCH_PEXT(table_256_256_pop)
DEF_BENCH_PEXT(table_16_16_inv_pop)
DEF_BENCH_PEXT(table_256_256_inv_pop)
DEF_BENCH_PEXT(pshufb)
DEF_BENCH_PEXT(hd7)

int main() {

	init_tables();

	bench_pext_hd7();
	bench_pext_pshufb();
	bench_pext_intrinsics();
	bench_pext_naive();
	bench_pext_naive2();
	bench_pext_table_16_16();
	bench_pext_table_256_256();
	bench_pext_table_16_16_inv();
	bench_pext_table_256_256_inv();
	bench_pext_table_16_16_pop();
	bench_pext_table_256_256_pop();
	bench_pext_table_16_16_inv_pop();
	bench_pext_table_256_256_inv_pop();

	return 0;
}
