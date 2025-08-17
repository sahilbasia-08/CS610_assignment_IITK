#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <limits>
#include <sys/time.h>
#include <vector>
#include <string>

#define THRESHOLD (std::numeric_limits<float>::epsilon())

using std::cerr;
using std::cout;
using std::endl;
using namespace std;
#define cudaCheckError(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

const uint64_t N = (1 << 6);  

__global__ void kernel2D_linear(const float* dev_input, float* dev_output, int filter_width) {
    int gscope_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gscope_idx >= N * N) return;

    int col_idx = gscope_idx % N;  
    int row_idx = gscope_idx / N;  

    int half_filter_width = filter_width / 2;
    float sum = 0.0f;
    int element_count = 0;

    for (int row_offset = -half_filter_width; row_offset <= half_filter_width; ++row_offset) {
        for (int col_offset = -half_filter_width; col_offset <= half_filter_width; ++col_offset) {
            int neighbor_x = col_idx + col_offset;
            int neighbor_y = row_idx + row_offset;
            if (neighbor_x >= 0 && neighbor_x < N && neighbor_y >= 0 && neighbor_y < N) {
                sum += dev_input[neighbor_y * N + neighbor_x];
                element_count++;
            }
        }
    }

    dev_output[gscope_idx] = sum / element_count;
}




__global__ void kernel3D_linear(const float* dev_input, float* dev_output, int filter_width) {
    int gscope_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gscope_idx >= N * N * N) return;

    int col_idx = gscope_idx % N;              
    int row_idx = (gscope_idx / N) % N;         
    int depth_idx = gscope_idx / (N * N);         

    int half_filter_width = filter_width / 2;
    float sum = 0.0f;
    int element_count = 0;

    for (int depth_offset = -half_filter_width; depth_offset <= half_filter_width; ++depth_offset) {
        for (int row_offset = -half_filter_width; row_offset <= half_filter_width; ++row_offset) {
            for (int col_offset = -half_filter_width; col_offset <= half_filter_width; ++col_offset) {
                int neighbor_x = col_idx + col_offset;
                int neighbor_y = row_idx + row_offset;
                int neighbor_z = depth_idx + depth_offset;
                if (neighbor_x >= 0 && neighbor_x < N && neighbor_y >= 0 && neighbor_y < N && neighbor_z >= 0 && neighbor_z < N) {
                    sum += dev_input[(neighbor_z * N * N) + (neighbor_y * N) + neighbor_x];
                    element_count++;
                }
            }
        }
    }

    dev_output[gscope_idx] = sum / element_count;
}


__global__ void kernel2D_optimized(const float* dev_input, float* dev_output, int filter_width) {
    int gscope_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gscope_idx >= N * N) return;

    int col_idx = gscope_idx % N;  
    int row_idx = gscope_idx / N;  

    int half_filter_width = filter_width / 2;
    float sum = 0.0f;
    int element_count = 0;
    int in_bounds = 0;
    
    for (int row_offset = -half_filter_width; row_offset <= half_filter_width; ++row_offset) {
        for (int col_offset = -half_filter_width; col_offset <= half_filter_width; ++col_offset) {
            int neighbor_x = col_idx + col_offset;
            int neighbor_y = row_idx + row_offset;
            in_bounds = (neighbor_x >= 0 && neighbor_x < N && neighbor_y >= 0 && neighbor_y < N);
            sum += in_bounds ? dev_input[neighbor_y * N + neighbor_x] : 0;
            element_count += in_bounds;
        }
    }

    dev_output[gscope_idx] = sum / element_count;
}


__global__ void kernel3D_optimized(const float* dev_input, float* dev_output, int filter_width) {
    int gscope_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gscope_idx >= N * N * N) return;

    int col_idx = gscope_idx % N;               
    int row_idx = (gscope_idx / N) % N;          
    int depth_idx = gscope_idx / (N * N);        

    int half_filter_width = filter_width / 2;
    float sum = 0.0f;
    int element_count = 0;
    int is_within_bounds = 0;

    for (int depth_offset = -half_filter_width; depth_offset <= half_filter_width; ++depth_offset) {
        for (int row_offset = -half_filter_width; row_offset <= half_filter_width; ++row_offset) {
            for (int col_offset = -half_filter_width; col_offset <= half_filter_width; ++col_offset) {
                int neighbor_x = col_idx + col_offset;
                int neighbor_y = row_idx + row_offset;
                int neighbor_z = depth_idx + depth_offset;
                
                is_within_bounds = (neighbor_x >= 0 && neighbor_x < N && 
                                    neighbor_y >= 0 && neighbor_y < N && 
                                    neighbor_z >= 0 && neighbor_z < N);

                sum += is_within_bounds ? dev_input[(neighbor_z * N * N) + (neighbor_y * N) + neighbor_x] : 0;
                element_count += is_within_bounds;
            }
        }
    }

    dev_output[gscope_idx] = element_count > 0 ? sum / element_count : 0.0f;
}

__global__ void kernel2D_shared(const float* dev_input, float* dev_output, int filter_width) {
    extern __shared__ float shared_mem[];

    int lscope_x = threadIdx.x;
    int lscope_y = threadIdx.y;
    int gscope_x = blockIdx.x * blockDim.x + lscope_x;
    int gscope_y = blockIdx.y * blockDim.y + lscope_y;
    int half_filter_width = filter_width / 2;

    if (gscope_x < N && gscope_y < N) {
        shared_mem[lscope_y * blockDim.x + lscope_x] = dev_input[gscope_y * N + gscope_x];
    }
    __syncthreads();

    float sum = 0.0f;
    int element_count = 0;

    for (int row_offset = -half_filter_width; row_offset <= half_filter_width; ++row_offset) {
        for (int col_offset = -half_filter_width; col_offset <= half_filter_width; ++col_offset) {
            int neighbor_x = lscope_x + col_offset;
            int neighbor_y = lscope_y + row_offset;

            if (neighbor_x >= 0 && neighbor_x < blockDim.x && neighbor_y >= 0 && neighbor_y < blockDim.y) {
                sum += shared_mem[neighbor_y * blockDim.x + neighbor_x];
                element_count++;
            }
        }
    }

    if (gscope_x < N && gscope_y < N) {
        dev_output[gscope_y * N + gscope_x] = sum / element_count;
    }
}


__global__ void kernel3D_shared(const float* dev_input, float* dev_output, int filter_width) {
    extern __shared__ float shared_mem[];

    int lscope_x = threadIdx.x;
    int lscope_y = threadIdx.y;
    int lscope_z = threadIdx.z;
    int gscope_x = blockIdx.x * blockDim.x + lscope_x;
    int gscope_y = blockIdx.y * blockDim.y + lscope_y;
    int gscope_z = blockIdx.z * blockDim.z + lscope_z;
    int half_filter_width = filter_width / 2;

    if (gscope_x < N && gscope_y < N && gscope_z < N) {
        shared_mem[(lscope_z * blockDim.y + lscope_y) * blockDim.x + lscope_x] = dev_input[gscope_z * N * N + gscope_y * N + gscope_x];
    }
    __syncthreads();

    float sum = 0.0f;
    int element_count = 0;

    for (int depth_offset = -half_filter_width; depth_offset <= half_filter_width; ++depth_offset) {
        for (int row_offset = -half_filter_width; row_offset <= half_filter_width; ++row_offset) {
            for (int col_offset = -half_filter_width; col_offset <= half_filter_width; ++col_offset) {
                int neighbor_x = lscope_x + col_offset;
                int neighbor_y = lscope_y + row_offset;
                int neighbor_z = lscope_z + depth_offset;

                if (neighbor_x >= 0 && neighbor_x < blockDim.x &&
                    neighbor_y >= 0 && neighbor_y < blockDim.y &&
                    neighbor_z >= 0 && neighbor_z < blockDim.z) {
                    sum += shared_mem[(neighbor_z * blockDim.y + neighbor_y) * blockDim.x + neighbor_x];
                    element_count++;
                }
            }
        }
    }

    if (gscope_x < N && gscope_y < N && gscope_z < N) {
        dev_output[gscope_z * N * N + gscope_y * N + gscope_x] = sum / element_count;
    }
}




__host__ void check_result(const float* w_ref, const float* w_opt, int size) {
    double maxdiff = 0.0;
    int numdiffs = 0;
    for (int i = 0; i < size; i++) {
        double diff = fabs(w_ref[i] - w_opt[i]);
        if (diff > THRESHOLD) {
            numdiffs++;
            maxdiff = fmax(maxdiff, diff);
        }
    }

    if (numdiffs > 0) {
        cout << numdiffs << " differences found over THRESHOLD; Max Diff = " << maxdiff << endl;
    } else {
        cout << "No differences found between base and optimized versions" << endl;
    }
}

__host__ void check_result2(const float* w_ref, const float* w_opt, int size) {
    double maxdiff = 0.0;
    int numdiffs = 0;

    for (int i = 0; i < size; i++) {
        double diff = fabs(w_ref[i] - w_opt[i]);
        if (diff > THRESHOLD) {
            numdiffs++;
            maxdiff = fmax(maxdiff, diff);
        }
    }

    if (numdiffs > 0) {
        cout << numdiffs << " differences found over THRESHOLD; Max Diff = " << maxdiff << endl;
    } else {
        cout << "No differences found between base and shared_mem versions" << endl;
    }
}

void print2D(const float* A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << A[i * N + j] << "\t";
        }
        cout << "n";
    }
}

void print3D(const float* A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                cout << A[i * N * N + j * N + k] << "\t";
            }
            cout << "n";
        }
        cout << "n";
    }
}

double rtclock() { 
    struct timezone Tzp;
    struct timeval Tp;
    int stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) {
        cout << "Error return from gettimeofday: " << stat << "\n";
    }
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
    const int filterWidth = 3;
    int threadsPerBlock = 256;
    vector<string> speedup;

    
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    
    float *cpu_input2D = new float[N * N];
    float *cpu_output2D = new float[N * N];
    float *cpu_output2D_optimized = new float[N * N];
    float *cpu_output2D_shared = new float[N * N];
    float *dev_input2D, *dev_output2D;

    for (int i = 0; i < N * N; i++) {
        cpu_input2D[i] = (static_cast<float>(rand()) / RAND_MAX) * 100;
    }

    cudaCheckError(cudaMalloc(&dev_input2D, N * N * sizeof(float)));
    cudaCheckError(cudaMalloc(&dev_output2D, N * N * sizeof(float)));
    cudaCheckError(cudaMemcpy(dev_input2D, cpu_input2D, N * N * sizeof(float), cudaMemcpyHostToDevice));

    int blocksPerGrid2D = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    
    
    cudaCheckError(cudaEventRecord(start));
    kernel2D_linear<<<blocksPerGrid2D, threadsPerBlock>>>(dev_input2D, dev_output2D, filterWidth);
    cudaCheckError(cudaMemcpy(cpu_output2D, dev_output2D, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));
    float time2D_normal;
    cudaCheckError(cudaEventElapsedTime(&time2D_normal, start, stop));
    cout << "GPU Execution time for 2D convolution (normal): " << time2D_normal << " ms" << endl;
    
    
    
    cudaCheckError(cudaEventRecord(start));
    kernel2D_optimized<<<blocksPerGrid2D, threadsPerBlock>>>(dev_input2D, dev_output2D, filterWidth);
    cudaCheckError(cudaMemcpy(cpu_output2D_optimized, dev_output2D, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));
    float time2D_opt;
    cudaCheckError(cudaEventElapsedTime(&time2D_opt, start, stop));
    cout << "GPU Execution time for 2D convolution (optimized): " << time2D_opt << " ms" << endl;
   
    check_result(cpu_output2D, cpu_output2D_optimized, N * N);
    
    
    speedup.push_back("Speedup of 2D optimized over 2D normal: " + to_string(time2D_normal / time2D_opt));

   
    int sharedMemSize2D = threadsPerBlock * threadsPerBlock * sizeof(float);
    cudaCheckError(cudaEventRecord(start));
    kernel2D_shared<<<blocksPerGrid2D, threadsPerBlock, sharedMemSize2D>>>(dev_input2D, dev_output2D, filterWidth);
    cudaCheckError(cudaMemcpy(cpu_output2D_shared, dev_output2D, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));
    float time2D_shared;
    cudaCheckError(cudaEventElapsedTime(&time2D_shared, start, stop));
    cout << "GPU Execution time for 2D convolution (shared memory): " << time2D_shared << " ms" << endl;
    
    check_result(cpu_output2D, cpu_output2D_shared, N * N);

   
    speedup.push_back("Speedup of 2D shared memory over 2D normal: " + to_string(time2D_normal / time2D_shared));

    
    delete[] cpu_input2D;
    delete[] cpu_output2D;
    delete[] cpu_output2D_optimized;
    delete[] cpu_output2D_shared;
    cudaCheckError(cudaFree(dev_input2D));
    cudaCheckError(cudaFree(dev_output2D));

   
    float *cpu_input3D = new float[N * N * N];
    float *cpu_output3D = new float[N * N * N];
    float *cpu_output3D_optimized = new float[N * N * N];
    float *cpu_output3D_shared = new float[N * N * N];
    float *dev_input3D, *dev_output3D;

    for (int i = 0; i < N * N * N; i++) {
        cpu_input3D[i] = (static_cast<float>(rand()) / RAND_MAX) * 100;
    }

    cudaCheckError(cudaMalloc(&dev_input3D, N * N * N * sizeof(float)));
    cudaCheckError(cudaMalloc(&dev_output3D, N * N * N * sizeof(float)));
    cudaCheckError(cudaMemcpy(dev_input3D, cpu_input3D, N * N * N * sizeof(float), cudaMemcpyHostToDevice));

    int blocksPerGrid3D = (N * N * N + threadsPerBlock - 1) / threadsPerBlock;
    
    
    cudaCheckError(cudaEventRecord(start));
    kernel3D_linear<<<blocksPerGrid3D, threadsPerBlock>>>(dev_input3D, dev_output3D, filterWidth);
    cudaCheckError(cudaMemcpy(cpu_output3D, dev_output3D, N * N * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));
    float time3D_normal;
    cudaCheckError(cudaEventElapsedTime(&time3D_normal, start, stop));
    cout << "GPU Execution time for 3D convolution (normal): " << time3D_normal << " ms" << endl;
    

   
    cudaCheckError(cudaEventRecord(start));
    kernel3D_optimized<<<blocksPerGrid3D, threadsPerBlock>>>(dev_input3D, dev_output3D, filterWidth);
    cudaCheckError(cudaMemcpy(cpu_output3D_optimized, dev_output3D, N * N * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));
    float time3D_opt;
    cudaCheckError(cudaEventElapsedTime(&time3D_opt, start, stop));
    cout << "GPU Execution time for 3D convolution (optimized): " << time3D_opt << " ms" << endl;
    
    check_result2(cpu_output3D, cpu_output3D_optimized, N * N * N);

    
    speedup.push_back("Speedup of 3D optimized over 3D normal: " + to_string(time3D_normal / time3D_opt));
    
   
    int sharedMemSize3D = threadsPerBlock * threadsPerBlock * threadsPerBlock * sizeof(float);
    cudaCheckError(cudaEventRecord(start));
    kernel3D_shared<<<blocksPerGrid3D, threadsPerBlock, sharedMemSize3D>>>(dev_input3D, dev_output3D, filterWidth);
    cudaCheckError(cudaMemcpy(cpu_output3D_shared, dev_output3D, N * N * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));
    float time3D_shared;
    cudaCheckError(cudaEventElapsedTime(&time3D_shared, start, stop));
    cout << "GPU Execution time for 3D convolution (shared memory): " << time3D_shared << " ms" << endl;
    
    check_result2(cpu_output3D, cpu_output3D_shared, N * N * N);

    
    speedup.push_back("Speedup of 3D shared memory over 3D normal: " + to_string(time3D_normal / time3D_shared));
    cout << endl;

    for (const auto& s : speedup) {
        cout << s << endl;
    }

    
    delete[] cpu_input3D;
    delete[] cpu_output3D;
    delete[] cpu_output3D_optimized;
    delete[] cpu_output3D_shared;
    cudaCheckError(cudaFree(dev_input3D));
    cudaCheckError(cudaFree(dev_output3D));

   
    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}




