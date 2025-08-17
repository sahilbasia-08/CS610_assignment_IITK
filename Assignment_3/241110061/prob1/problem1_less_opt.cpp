#include <chrono>
#include <climits>
#include <cmath>
#include <iostream>
#include <limits>
#include <x86intrin.h>
#include <emmintrin.h>
#include <immintrin.h>

using namespace std;

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const static float EPSILON = std::numeric_limits<float>::epsilon();


void print_m256(__m256 vec) {
    // Create an array to hold the results
    float values[8];
    // Store the values from the __m256 vector into the array
    _mm256_storeu_ps(values, vec);

    // Print the values
    std::cout << "Vector values: ";
    for (int i = 0; i < 8; i++) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;
}

#define N (1024)
#define SSE4_bit_width (128)
#define ALIGN_2 (32)
#define AVX2_bit_width (256)

// sequential matrix - matrix multiplication
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
    //__builtin_assume_aligned(A, ALIGN_2);
    //__builtin_assume_aligned(B, ALIGN_2);
    //__builtin_assume_aligned(C, ALIGN_2);
    
    int stride = SSE4_bit_width/(CHAR_BIT*sizeof(int));

    for(int i=0; i<N; ++i){
    	float summ=0;
        for(int j=0; j<N; j++){

             float summ=0;

            for(int k=0; k<N; k+=stride){
                __m128 x = _mm_load_ps(&A[i][k]);
                __m128 y = _mm_set_ps(B[k+3][j],B[k+2][j],B[k+1][j],B[k+0][j]);
               	
               x = _mm_mul_ps(x,y);
               
               x = _mm_hadd_ps(x,x);
               x = _mm_hadd_ps(x,x);
               
	       float *last = (float*)&x;
	       summ+=last[3];

	      
            }
           
            C[i][j]=summ;
            
        }
    }
}

void matmul_sse4_un(float** __restrict__ A, float** __restrict__ B, float** __restrict__ C) {
    //__builtin_assume_aligned(A, ALIGN_2);
    //__builtin_assume_aligned(B, ALIGN_2);
    //__builtin_assume_aligned(C, ALIGN_2);

    int stride = SSE4_bit_width/(CHAR_BIT*sizeof(int));

    for(int i=0; i<N; ++i){
    	float summ=0;
        for(int j=0; j<N; j++){

             float summ=0;

            for(int k=0; k<N; k+=stride){
            	// unaligned version of sse4 instrisics
                __m128 x = _mm_loadu_ps(&A[i][k]);
               	__m128 y = _mm_set_ps(B[k+3][j],B[k+2][j],B[k+1][j],B[k+0][j]);
               
               x = _mm_mul_ps(x,y);
              
               x = _mm_hadd_ps(x,x);
               x = _mm_hadd_ps(x,x);
               
	       float *last = (float*)&x;
	       summ+=last[3];

	       
            }
            
            C[i][j]=summ;
           
        }
    }
}


void matmul_avx2(float** __restrict__ A, float** __restrict__ B, float** __restrict__ C) {
	//__builtin_assume_aligned(A, ALIGN_2);
	//__builtin_assume_aligned(B, ALIGN_2);
	//__builtin_assume_aligned(C, ALIGN_2);
	
	const int stride = AVX2_bit_width/(CHAR_BIT*sizeof(int));

	for(int i=0; i<N; ++i){
		for(int j=0; j<N; ++j){
            	float summ=0;
			for(int k=0; k<N; k+=stride){
				
		        __m256 x = _mm256_load_ps(&A[i][k]);
			    __m256 y = _mm256_set_ps(B[k+7][j], B[k+6][j],B[k+5][j],B[k+4][j],B[k+3][j],B[k+2][j],B[k+1][j],B[k+0][j]);

			    // Element-wise multiplication
			    __m256 c = _mm256_mul_ps(x, y); // Multiply x and y

			    // Horizontal addition to reduce to a single value
			    // Step 1: Permute for horizontal addition
			    __m256 inter_m = _mm256_permute2f128_ps(c, c, 1); // Get the second half of c
			    
			    // Step 2: Horizontal add
			    c = _mm256_hadd_ps(c, inter_m); // Adds pairs: (c[0]+c[4]), (c[1]+c[5]), etc.
			    
			    // Step 3: Further horizontal add to finalize
			    inter_m = _mm256_permute2f128_ps(c, c, 1); // Permute to get pairs for final reduction
			    
			    c = _mm256_hadd_ps(inter_m, c); // Final add to get the sum in the lower part
			    c = _mm256_hadd_ps(c, c); // Final reduction to get the sum
			

		        float *last = (float*)&c;
		        summ+=last[0];
		        
		    	}
		    	
            	C[i][j]=summ;
		}

	}

}




void matmul_avx2_un(float** __restrict__ A, float** __restrict__ B, float** __restrict__ C) {
	//__builtin_assume_aligned(A, ALIGN_2);
	//__builtin_assume_aligned(B, ALIGN_2);
	//__builtin_assume_aligned(C, ALIGN_2);
	
	const int stride = AVX2_bit_width/(CHAR_BIT*sizeof(int));

	for(int i=0; i<N; ++i){
		for(int j=0; j<N; ++j){
            	float summ=0;
			for(int k=0; k<N; k+=stride){
				// unaligned version of avx2 instrisics
		            __m256 x = _mm256_loadu_ps(&A[i][k]);
		        __m256 y = _mm256_set_ps(B[k+7][j], B[k+6][j],B[k+5][j],B[k+4][j],B[k+3][j],B[k+2][j],B[k+1][j],B[k+0][j]);

    // Element-wise multiplication
    __m256 c = _mm256_mul_ps(x, y); // Multiply x and y

    // Horizontal addition to reduce to a single value
    // Step 1: Permute for horizontal addition
    __m256 inter_m = _mm256_permute2f128_ps(c, c, 1); // Get the second half of c
    
    // Step 2: Horizontal add
    c = _mm256_hadd_ps(c, inter_m); // Adds pairs: (c[0]+c[4]), (c[1]+c[5]), etc.
    
    // Step 3: Further horizontal add to finalize
    inter_m = _mm256_permute2f128_ps(c, c, 1); // Permute to get pairs for final reduction
    
    c = _mm256_hadd_ps(inter_m, c); // Final add to get the sum in the lower part
    c = _mm256_hadd_ps(c, c);
    //print_m256(c);
    // At this point, c[0] holds the final result of the dot product.
			
		        float *last = (float*)&c;
		        summ+=last[0];
		        
		    	}
		    	
            	C[i][j]=summ;
		}

	}

}






void check_result(float** w_ref, float** w_opt) {
  float maxdiff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > EPSILON) {
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

