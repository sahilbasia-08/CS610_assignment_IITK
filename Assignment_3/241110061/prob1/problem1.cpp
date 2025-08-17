#include <chrono>
#include <climits>
#include <cmath>
#include <iostream>
#include <limits>
#include <x86intrin.h>
#include <emmintrin.h>
#include <immintrin.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const static float EPSILON = std::numeric_limits<float>::epsilon();

#define N (1024)
#define SSE4_bit_width (128)
#define ALIGN_2 (32)
#define AVX2_bit_width (256)

void matmul_seq(float** A, float** B, float** C) {
  float sum = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

void matmul_sse4(float** __restrict__ A, float** __restrict__ B, float** __restrict__ C) {
	__builtin_assume_aligned(A, ALIGN_2);
	__builtin_assume_aligned(B, ALIGN_2);
	__builtin_assume_aligned(C, ALIGN_2);
	int stride = SSE4_bit_width/(CHAR_BIT * sizeof(int));
	for(int i = 0; i < N; ++i) {
		for(int k = 0; k < N; k++) {
			float copy = A[i][k];
			for(int j = 0; j < N; j += stride) {
				__m128 x = _mm_load_ps( &B[k][j]);
				__m128 y = _mm_set1_ps(copy); 
				x = _mm_mul_ps(x, y);
				__m128 c = _mm_load_ps( &C[i][j]);
				c = _mm_add_ps(c, x);
				_mm_store_ps( &C[i][j], c);
			}
		}
	}
}

void matmul_avx2(float** __restrict__ A, float** __restrict__ B, float** __restrict__ C) {
	__builtin_assume_aligned(A, ALIGN_2);
	__builtin_assume_aligned(B, ALIGN_2);
	__builtin_assume_aligned(C, ALIGN_2);
	const int stride = AVX2_bit_width / (CHAR_BIT * sizeof(int));
	for(int i = 0; i < N; ++i) {
		for(int k = 0; k < N; ++k) {
			float copy = A[i][k];
			for(int j = 0; j < N; j += stride) {
				__m256 x = _mm256_load_ps( & B[k][j]);
				__m256 y = _mm256_set1_ps(copy);
				x = _mm256_mul_ps(x, y);
				__m256 c = _mm256_load_ps( & C[i][j]);
				c = _mm256_add_ps(x, c);
				_mm256_store_ps( & C[i][j], c);
			}
		}
	}
}

void matmul_sse4_un(float** A, float** B, float** C) {
	//__builtin_assume_aligned(A, ALIGN_2);
	//__builtin_assume_aligned(B, ALIGN_2);
	//__builtin_assume_aligned(C, ALIGN_2);
	int stride = SSE4_bit_width/(CHAR_BIT * sizeof(int));
	for(int i = 0; i < N; ++i) {
		for(int k = 0; k < N; k++) {
			float copy = A[i][k];
			for(int j = 0; j < N; j += stride) {
				__m128 x = _mm_loadu_ps( & B[k][j]);
				__m128 y = _mm_set1_ps(copy); 
				x = _mm_mul_ps(x, y);
				__m128 c = _mm_loadu_ps( &C[i][j]);
				c = _mm_add_ps(c, x);
				_mm_store_ps( &C[i][j], c);
			}
		}
	}
}

void matmul_avx2_un(float** A, float** B, float** C) {
	//__builtin_assume_aligned(A, ALIGN_2);
	//__builtin_assume_aligned(B, ALIGN_2);
	//__builtin_assume_aligned(C, ALIGN_2);
	int stride = AVX2_bit_width / (CHAR_BIT * sizeof(int));
	for(int i = 0; i < N; ++i) {
		for(int k = 0; k < N; ++k) {
			float copy=A[i][k];
			for(int j = 0; j < N; j += stride) {
	
				__m256 x = _mm256_loadu_ps( & B[k][j]);
				__m256 y = _mm256_set1_ps(copy);
				x = _mm256_mul_ps(x, y);
				__m256 c = _mm256_load_ps(&C[i][j]);
				c = _mm256_add_ps(x, c);
				_mm256_storeu_ps( &C[i][j], c);
			}
		}
	}
}


void check_result(float** w_ref, float** w_opt) {
  float maxdiff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
   
      float this_diff = w_ref[i][j] - w_opt[i][j];
      
      if (fabs(this_diff) > EPSILON ) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << EPSILON
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}
// setting for the aligned version of matmul implementaion
int main() {
  auto** A = new alignas(32) float*[N];
  for (int i = 0; i < N; i++) {
    A[i] = new alignas(32) float[N]();
  }
  auto** B = new alignas(32) float*[N];
  for (int i = 0; i < N; i++) {
    B[i] = new alignas(32) float[N]();
  }

  auto** C_seq = new alignas(32) float*[N];
  auto** C_sse4 = new alignas(32) float*[N];
  auto** C_avx2 = new alignas(32) float*[N];
  for (int i = 0; i < N; i++) {
    C_seq[i] = new alignas(32) float[N]();
    C_sse4[i] = new alignas(32) float[N]();
    C_avx2[i] = new alignas(32) float[N]();
  }

 

  // setting everything for unaligned version implementation according to the question given
  auto** A_un = new float*[N];
  for (int i = 0; i < N; i++) {
    A_un[i] = new float[N]();
  }
  auto** B_un = new float*[N];
  for (int i = 0; i < N; i++) {
    B_un[i] = new float[N]();
  }

  auto** C_seq_un = new float*[N];
  auto** C_sse4_un = new float*[N];
  auto** C_avx2_un = new float*[N];
  for (int i = 0; i < N; i++) {
    C_seq_un[i] = new  float[N]();
    C_sse4_un[i] = new  float[N]();
    C_avx2_un[i] = new float[N]();
  } 
  
  // initialize arrays
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = 0.1F;
      B[i][j] = 0.2F;
      C_seq[i][j] = 0.0F;
      C_sse4[i][j] = 0.0F;
      C_avx2[i][j] = 0.0F;
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_un[i][j] = 0.1F;
      B_un[i][j] = 0.2F;
      C_seq_un[i][j] = 0.0F;
      C_sse4_un[i][j] = 0.0F;
      C_avx2_un[i][j] = 0.0F;
    }
  }
  cout<<"Below is aligned version implementation"<<endl;
  HRTimer start = HR::now();
  matmul_seq(A, B, C_seq);
  HRTimer end = HR::now();
  auto duration1 = duration_cast<milliseconds>(end - start).count();
  cout << "Aligned Matmul seq time: " << duration1 << " ms" << endl;

  start = HR::now();
  matmul_sse4(A, B, C_sse4);
  end = HR::now();
  check_result(C_seq, C_sse4);
  auto duration2 = duration_cast<milliseconds>(end - start).count();
  cout << "Aligned  Matmul SSE4 time: " << duration2 << " ms" << endl;

  start = HR::now();
  matmul_avx2(A, B, C_avx2);
  end = HR::now();
  check_result(C_seq, C_avx2);
  auto duration3 = duration_cast<milliseconds>(end - start).count();
  cout << "Aligned Matmul AVX2 time: " << duration3 << " ms" << endl<<endl;
  


  cout<<"Everything below is unaligned version implementation"<<endl;
  start = HR::now();
  matmul_seq(A_un, B_un, C_seq_un);
  end = HR::now();
  auto duration4 = duration_cast<milliseconds>(end - start).count();
  cout << "UnAligned Matmul seq time: " << duration4 << " ms" << endl;
  
  start = HR::now();
  matmul_sse4_un(A_un, B_un, C_sse4_un);
  end = HR::now();
  check_result(C_seq, C_sse4_un);
  auto duration5 = duration_cast<milliseconds>(end - start).count();
  cout << "UnAligned Matmul SSE4 time: " << duration5 << " ms" << endl;

  start = HR::now();
  matmul_avx2_un(A_un, B_un, C_avx2_un);
  end = HR::now();
  check_result(C_seq, C_avx2_un);
  auto duration6 = duration_cast<milliseconds>(end - start).count();
  cout << "UnAligned Matmul AVX2 time: " << duration6 << " ms" << endl;

  //cout<<"Speedups achieved"<<endl<<endl;
  //cout<<"aligned sse4 over sequntial = "<<(float)duration1/duration2<endl;
  //cout<<"aligned avx2 over sequntial = "<<(float)duration1/duration3<endl;
  //cout<<"unaligned sse4 over sequntial = "<<(float)duration4/duration5<endl;
  //cout<<"aligned avx2 over sequntial = "<<(float)duration4/duration6<endl;
  for (int i = 0; i < N; i++) {
    delete[] A[i];
    delete[] B[i];
    delete[] C_seq[i];
    delete[] C_sse4[i];
    delete[] C_avx2[i];
}
  delete[] A;
  delete[] B;
  delete[] C_seq;
  delete[] C_sse4;
  delete[] C_avx2;
  
  
  for (int i = 0; i < N; i++) {
    delete[] A_un[i];
    delete[] B_un[i];
    delete[] C_seq_un[i];
    delete[] C_sse4_un[i];
    delete[] C_avx2_un[i];
}
  delete[] A_un;
  delete[] B_un;
  delete[] C_seq_un;
  delete[] C_sse4_un;
  delete[] C_avx2_un;
  
  return EXIT_SUCCESS;
}
