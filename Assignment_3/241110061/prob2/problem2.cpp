#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <x86intrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <omp.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::nanoseconds;
#define N (1 << 16)
#define SSE_WIDTH_BITS (128)
#define ALIGN (32)

#define ALIGN_2 (32)
#define AVX2_WIDTH_BITS (256) // the avx2 instrinsics has 256 width bits

/** Helper methods for debugging */

void print_array(const int* array) {
  for (int i = 0; i < N; i++) {
    cout << array[i] << "\t";
  }
  cout << "\n";
}

void print128i_u32(__m128i var, int start) {
  alignas(ALIGN) uint32_t val[4];
  _mm_store_si128((__m128i*)val, var);
  cout << "Values [" << start << ":" << start + 3 << "]: " << val[0] << " "
       << val[1] << " " << val[2] << " " << val[3] << "\n";
}

void print128i_u64(__m128i var) {
  alignas(ALIGN) uint64_t val[2];
  _mm_store_si128((__m128i*)val, var);
  cout << "Values [0:1]: " << val[0] << " " << val[1] << "\n";
}

// serial version with hinting the compiler no-tree-vectorize this piece of serial code
__attribute__((optimize("no-tree-vectorize"))) int
ref_version(int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  int tmp = 0;
  for (int i = 0; i < N; i++) {
    tmp += source[i];
    dest[i] = tmp;
  }
  return tmp;
}

int omp_version(const int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  int tmp = 0;
#pragma omp simd reduction(inscan, + : tmp) // this inscan is helping us to find the inlcusive prefix sum using omp simd reduction
  for (int i = 0; i < N; i++) {
    tmp += source[i];

#pragma omp scan inclusive(tmp) //This is used inside a loop to indicate that a scan (or prefix operation) is
                                //it works with the inscan modifier in a reduction clause to manage how the partial results
                                //are stored during a scan operation.
                                //The keyword inclusive specifies that this is an inclusive scan. In an inclusive scan,
                                //the current element includes itself and all the preceding elements in the operation.
    dest[i] = tmp;

  }
  return tmp;
}

// Tree reduction idea on every 128 bits vector data, involves 2 shifts, 3 adds,
// one broadcast
int sse4_version(const int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  // Return vector of type __m128i with all elements set to zero, to be added as
  // previous sum for the first four elements.
  __m128i offset = _mm_setzero_si128();
  const int stride = SSE_WIDTH_BITS / (sizeof(int) * CHAR_BIT);
  for (int i = 0; i < N; i += stride) {
    // Load 128-bits of integer data from memory into x. source_addr must be
    // aligned on a 16-byte boundary to be safe.
    __m128i x = _mm_load_si128((__m128i*)&source[i]);
    // Let the numbers in x be [d,c,b,a], where a is at source[i].
    __m128i tmp0 = _mm_slli_si128(x, 4);
    // Shift x left by 4 bytes while shifting in zeros. tmp0 becomes [c,b,a,0].
    __m128i tmp1 = _mm_add_epi32(x, tmp0); // Add packed 32-bit integers in x and tmp0.
    // tmp1 becomes [c+d,b+c,a+b,a].
    // Shift tmp1 left by 8 bytes while shifting in zeros.
    __m128i tmp2 = _mm_slli_si128(tmp1, 8); // tmp2 becomes [a+b,a,0,0].
    // Add packed 32-bit integers in tmp2 and tmp1.
    __m128i out = _mm_add_epi32(tmp2, tmp1);
    // out contains [a+b+c+d,a+b+c,a+b,a].
    out = _mm_add_epi32(out, offset);
    // out now includes the sum from the previous set of numbers, given by
    // offset.
    // Store 128-bits of integer data from out into memory. dest_addr must be
    // aligned on a 16-byte boundary to be safe.
    _mm_store_si128((__m128i*)&dest[i], out);
    // _MM_SHUFFLE(z, y, x, w) macro forms an integer mask according to the
    // formula (z << 6) | (y << 4) | (x << 2) | w.
    int mask = _MM_SHUFFLE(3, 3, 3, 3);
    // Bits [7:0] of mask are 11111111 to pick the third integer (11) from out
    // (i.e., a+b+c+d).

    // Shuffle 32-bit integers in out using the control in mask.
    offset = _mm_shuffle_epi32(out, mask);
    // offset now contains 4 copies of a+b+c+d.
  }
  return dest[N - 1];
}

int avx2_version(const int* __restrict__  source, int* __restrict__ dest) {
   __builtin_assume_aligned(source, ALIGN_2);
   __builtin_assume_aligned(dest, ALIGN_2);


    __m256i offset = _mm256_setzero_si256();
    const int stride_jump = AVX2_WIDTH_BITS/(sizeof(int)* CHAR_BIT); // CHAR_BIT is just other way of writing value of 1 Byte = 8 bit
                                                                      //We can also write it as sizeof(int) * 8 if char is always 1 byte

    for(int i=0;  i<N; i+=stride_jump){

        //add comments to explain every instrinscis
        //also it will act as refercne for your preparation
        //
        __m256i vec = _mm256_load_si256((__m256i*)&source[i]);
	// 8 elements loaded from consecutive memory
	//eg [h,g,f,e,d,c,b,a]
        __m256i temp_vec =  _mm256_slli_si256(vec,4);
	// now shifting 128 bit lanes with 4 byte of left shift
	// [g,f,e,0,c,b,a,0]

        temp_vec = _mm256_add_epi32(vec, temp_vec);
	// after adding temp_vec = [h+g, g+f, e+f, e, d+c, b+c, b+a, a]
        __m256i temp_vec_1 = _mm256_slli_si256(temp_vec, 8);
	// shifted 128 bits lane with 8 bytes towards left
	// [e+f, e, 0, 0, b+c, a, 0, 0]
        temp_vec = _mm256_add_epi32(temp_vec_1,  temp_vec);
	// after adding temp_vec = [h+g+e+f, g+f+e, e+f, e, c+d+b+a, c+b+a, b+a, a]
        __m256i mask = _mm256_set1_epi32(3);
        // mask is now set as [3,3,3,3,3,3,3,3]
        temp_vec_1 = _mm256_permutevar8x32_epi32(temp_vec,mask);
        // permutevar8x32_epi32 - Shuffle single-precision (32-bit) floating-point elements in a across lanes using the corresponding index in idx.
        // here idx is also a 256 bit vecter regiser containiong the index with which the values of temp_vec will be choosen and permuted
	// this will make temp_vec_1 vector as [a+b+c+d,a+b+c+d,a+b+c+d,a+b+c+d,a,a,a,a]

	temp_vec_1 = _mm256_permute2x128_si256(temp_vec_1, temp_vec_1, 25);
        // now temp_vec_1 will have only higher order 128 bits of [a+b+c+d,a+b+c+d,a+b+c+d,a+b+c+d,a,a,a,a]
        // becuase of 25 imm8 value choosen
        // [a+b+c+d,a+b+c+d,a+b+c+d,a+b+c+d,0,0,0,0] - rest will be all be zeroes
        __m256i out_vec = _mm256_add_epi32(temp_vec_1, temp_vec);
	// now adding this [a+b+c+d,a+b+c+d,a+b+c+d,a+b+c+d,0,0,0,0] with [h+g+e+f, g+f+e, e+f, e, c+d+b+a, c+b+a, b+a, a]
	// out_vec = [a+b+c+d+e+f+g+h,a+b+c+d+e+f+g,a+b+c+d+e+f,a+b+c+d+e,a+b+c+d,a+b+c,a+b,a] // we got the prefix sum for our of first 8 integers
        out_vec = _mm256_add_epi32(out_vec, offset);
	// original offset was all zeroes so adding with offset will give us same
	// [a+b+c+d+e+f+g+h,a+b+c+d+e+f+g,a+b+c+d+e+f,a+b+c+d+e,a+b+c+d,a+b+c,a+b,a]
	// this offset but will act as soruce for next 256 bit lanes values that will be picked from source
        _mm256_store_si256((__m256i*)&dest[i],out_vec);
	// stroing the ressult in memoprty
        mask = _mm256_set1_epi32(7);
	// creating new mask register . here  7 measn that now we will pick 7th index of
	// [a+b+c+d+e+f+g+h,a+b+c+d+e+f+g,a+b+c+d+e+f,a+b+c+d+e,a+b+c+d,a+b+c,a+b,a] vector
        offset = _mm256_permutevar8x32_epi32(out_vec, mask);
	// offset now has vector has [a+b+c+d+e+f+g+h,a+b+c+d+e+f+g+h,a+b+c+d+e+f+g+h,a+b+c+d+e+f+g+h,a+b+c+d+e+f+g+h,a+b+c+d+e+f+g+h,a+b+c+d+e+f+g+h,a+b+c+d+e+f+g+h]
	// this offset is added in next consecutive 8 integers according to concept of prefix sums


    }


    return dest[N-1]; // this the final value of the inclusive sum of values
}

__attribute__((optimize("no-tree-vectorize"))) int main() {
  int* array = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(array, array + N, 1);

  int* ref_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(ref_res, ref_res + N, 0);
  HRTimer start = HR::now();
  int val_ser = ref_version(array, ref_res);
  HRTimer end = HR::now();
  auto duration1 = duration_cast<microseconds>(end - start).count();
  cout << "Serial version: " << val_ser << " time: " << duration1 << endl;

  // this is the openmp version of prefix sum
  int* omp_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(omp_res, omp_res + N, 0);
  start = HR::now();
  int val_omp = omp_version(array, omp_res);
  end = HR::now();
  auto duration2 = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_omp || printf("OMP result is wrong!\n"));
  cout << "OMP version: " << val_omp << " time: " << duration2 << endl;
  free(omp_res);

  int* sse_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(sse_res, sse_res + N, 0);
  start = HR::now();
  int val_sse = sse4_version(array, sse_res);
  end = HR::now();
  auto duration3 = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_sse || printf("SSE result is wrong!\n"));
  cout << "SSE version: " << val_sse << " time: " << duration3 << endl;
  free(sse_res);

  // setting and calling the avx2 version of inclusive prefix sum version
  int *avx2_res = static_cast<int*>(aligned_alloc(ALIGN_2, N*sizeof(int)));
  std::fill(avx2_res, avx2_res+N, 0);
  start = HR::now();
  int val_avx2 = avx2_version(array, avx2_res);
  end = HR::now();
  //print_array(avx2_res);
  auto duration_avx2 = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_avx2 || printf("AVX2 result is wrong!\n"));
  cout << "AVX2 version: " << val_avx2 << " time: " << duration_avx2 << endl;
  free(avx2_res);
  cout<<endl;

  // comparing the speedups now
  //duration 1 - sequntial
  //duration 2 - SSE4
  //duration 3 - openmp
  //duration 4 - AVX2
  cout<<"Below are the speedups of avx2 over other approaches"<<endl;
  cout<<"speed up of avx2 over sequential approach = "<<(float)duration1/duration_avx2<<endl;
  cout<<"speed up of avx2 over openmp apprach = "<<(float)duration2/duration_avx2<<endl;
  cout<<"speed up of avx2 over sse4 approach = "<<(float)duration3/duration_avx2<<endl;
  cout<<endl;
  cout<<"Below are the speedups of openmp over other approaches"<<endl;
  cout<<"speed up of openmp over sequential approach = "<<(float)duration1/duration2<<endl;
  cout<<"speed up of openmp over sse4 apprach = "<<(float)duration3/duration2<<endl;
  cout<<"speed up of openmp over avx2 approach = "<<(float)duration_avx2/duration2<<endl;
  cout<<endl;
  cout<<"Below are the speedups of sse4 over other approaches"<<endl;
  cout<<"speed up of sse4 over sequential approach = "<<(float)duration1/duration3<<endl;
  cout<<"speed up of sse4 over openmp apprach = "<<(float)duration2/duration3<<endl;
  cout<<"speed up of sse4 over avx2 approach = "<<(float)duration_avx2/duration3<<endl;
  /*cout<<endl;
  cout<<"Below are the speedups of AVX2 over other approaches"<<endl;
  cout<<"speed up of avx2 over sequential approach = "<<(float)duration1/duration_avx2<<endl;
  cout<<"speed up of avx2 over openmp apprach = "<<(float)duration2/duration_avx2<<endl;
  cout<<"speed up of avx2 over sse4 approach = "<<(float)duration3/duration_avx2<<endl;
 */
  return EXIT_SUCCESS;
}
