#include<iostream>
#include<cstdint>
#include<cassert>
#include<random>
#include<string>
#include<chrono>

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>


inline uint32_t bitscan_forward64(const uint64_t x, uint32_t *dest) {

	//xが非ゼロなら、立っているビットのうち最下位のものの位置をdestに代入して、非ゼロの値を返す。
	//xがゼロなら、ゼロを返す。このときのdestの値は未定義である。

#ifdef _MSC_VER
	return _BitScanForward64(reinterpret_cast<unsigned long *>(dest), x);
#else
	return x ? *dest = __builtin_ctzl(x), 1 : 0;
#endif

}

int select1_naive(uint64_t x, int count) {

	assert(0 <= count && count < 64);
	if (_mm_popcnt_u64(x) <= count)return -1;

	for (int n = 0, i = 0; i < 64; ++i) {
		if (x & (1ULL << i)) {
			if (count == n++)return i;
		}
	}

	assert(0);
	return -1;
}

int select1_bsf(uint64_t x, int count) {

	assert(0 <= count && count < 64);
	if (_mm_popcnt_u64(x) <= count)return -1;

	for (int i = 0; i < count; ++i)x &= x - 1;
	uint32_t index = 0;
	bitscan_forward64(x, &index);
	return int(index);

	assert(0);
	return -1;
}

int select1_popcnt_binarysearch(uint64_t x, int count) {

	assert(0 <= count && count < 64);
	if (_mm_popcnt_u64(x) <= count)return -1;

	int lb = 0, ub = 64;
	while (lb + 1 < ub) {
		const int mid = (lb + ub) / 2;
		const int pop = _mm_popcnt_u64(x & ((1ULL << mid) - 1));
		if (pop <= count) lb = mid;
		else ub = mid;
	}
	return lb;

	assert(0);
	return -1;
}

int select1_pdep(uint64_t x, int count) {

	assert(0 <= count && count < 64);
	if (_mm_popcnt_u64(x) <= count)return -1;

	const uint64_t answer_bit = _pdep_u64(1ULL << count, x);
	uint32_t index = 0;
	bitscan_forward64(answer_bit, &index);
	return int(index);

	assert(0);
	return -1;
}


void test_select1() {

	std::mt19937_64 rnd(123);

	for (int iter = 0; iter < 100000; ++iter) {
		const uint64_t x = rnd();
		const int pop = _mm_popcnt_u64(x);

		for (int i = 0; i < pop; ++i) {
			const int answer1 = select1_naive(x, i);
			const int answer2 = select1_bsf(x, i);
			const int answer3 = select1_popcnt_binarysearch(x, i);
			const int answer4 = select1_pdep(x, i);
			assert(answer1 == answer2 && answer2 == answer3 && answer3 == answer4);
		}
	}
}

inline uint64_t xorshift64(uint64_t x) {
	x = x ^ (x << 7);
	return x ^ (x >> 9);
}

#define DEF_BENCH_SELECT1(name) \
void bench_select1_##name() {\
	std::cout << "Bench select1:"#name << std::endl;\
	int64_t result = 0;\
	uint64_t a = 0x1111222233334444ULL;\
	auto start = std::chrono::system_clock::now();\
	for (int i = 0; i < (1 << 30); ++i) {\
		for(int j = _mm_popcnt_u64(a) - 1; j >= 0 && i < (1 << 30); ++i, --j) {\
			result += select1_##name(a, j);\
		}\
		a = xorshift64(a);\
	}\
	auto end = std::chrono::system_clock::now();\
	std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;\
	std::cout << "result (for validation) = " << std::to_string(result) << std::endl << std::endl;\
}

DEF_BENCH_SELECT1(naive)
DEF_BENCH_SELECT1(bsf)
DEF_BENCH_SELECT1(popcnt_binarysearch)
DEF_BENCH_SELECT1(pdep)

int main() {

	test_select1();

	bench_select1_naive();
	bench_select1_bsf();
	bench_select1_popcnt_binarysearch();
	bench_select1_pdep();

	return 0;
}