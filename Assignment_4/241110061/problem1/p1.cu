#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <limits>
#include <sys/time.h>
#include <vector>

using namespace std;
using std::cerr;
using std::cout;
using std::endl;
const uint64_t N = (128);
#define block_side_1 1
#define block_side_2 2
#define block_side_4 4
#define block_side_8 8

#define X_DIM 16
#define Z_DIM 8
#define X_GRID N/X_DIM
#define Y_GRID N/X_DIM
#define Z_GRID N/Z_DIM

#define threads_per_block 256
#define THRESHOLD (std::numeric_limits<float>::epsilon()) 

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



bool flag = 0;

// TODO: Edit the function definition as required
__global__ void kernel1(const float *device_in, float *device_out) { 
  int t_id_x = blockIdx.x * blockDim.x + threadIdx.x;
  int store = N * N;
  
  int z = t_id_x / store;
  int y = (t_id_x % store) / N;
  int x = t_id_x % N;
  if (x > 0 && x < N - 1 && y > 0 && y < N - 1 && z > 0 && z < N - 1) {
    device_out[t_id_x] = 0.8f * (device_in[(z - 1) * store + y * N + x]
                    + device_in[(z + 1) * store + y * N + x]
                    + device_in[z * store + (y - 1) * N + x]
                    + device_in[z * store + (y + 1) * N + x]
                    + device_in[z * store + y * N + (x - 1)]
                    + device_in[z * store + y * N + (x + 1)]);
  }
 
}

// TODO: Edit the function definition as required
__global__ void kernel2_1(const float *device_in, float *device_out) {
    __shared__ float shared_mem[X_DIM * X_DIM * Z_DIM];

    int thread_idx = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;

    int block_start_x = blockIdx.x * X_DIM;
    int block_start_y = blockIdx.y * X_DIM;
    int block_start_z = blockIdx.z * Z_DIM;

    for (int offset = thread_idx; offset < X_DIM * X_DIM * Z_DIM; offset += total_threads) {
        int lscope_x = offset % X_DIM;
        int lscope_y = (offset / X_DIM) % X_DIM;
        int lscope_z = offset / (X_DIM * X_DIM);

        int gscope_x = block_start_x + lscope_x;
        int gscope_y = block_start_y + lscope_y;
        int gscope_z = block_start_z + lscope_z;

        if (gscope_x > 0 && gscope_x < N - 1 && gscope_y > 0 && gscope_y < N - 1 && gscope_z > 0 && gscope_z < N - 1) {
            shared_mem[offset] = 0.8f * (
                device_in[(gscope_z - 1) * N * N + gscope_y * N + gscope_x] +
                device_in[(gscope_z + 1) * N * N + gscope_y * N + gscope_x] +
                device_in[gscope_z * N * N + (gscope_y - 1) * N + gscope_x] +
                device_in[gscope_z * N * N + (gscope_y + 1) * N + gscope_x] +
                device_in[gscope_z * N * N + gscope_y * N + (gscope_x - 1)] +
                device_in[gscope_z * N * N + gscope_y * N + (gscope_x + 1)]
            );
        }
    }

    __syncthreads();

    for (int offset = thread_idx; offset < X_DIM * X_DIM * Z_DIM; offset += total_threads) {
        int lscope_x = offset % X_DIM;
        int lscope_y = (offset / X_DIM) % X_DIM;
        int lscope_z = offset / (X_DIM * X_DIM);

        int gscope_x = block_start_x + lscope_x;
        int gscope_y = block_start_y + lscope_y;
        int gscope_z = block_start_z + lscope_z;

        if (gscope_x > 0 && gscope_x < N - 1 && gscope_y > 0 && gscope_y < N - 1 && gscope_z > 0 && gscope_z < N - 1) {
            device_out[gscope_z * N * N + gscope_y * N + gscope_x] = shared_mem[offset];
        }
    }
}

__global__ void kernel2_2(const float *device_in, float *device_out) {
    __shared__ float shared_mem[X_DIM * X_DIM * Z_DIM];

    int thread_idx = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;

    int block_start_x = blockIdx.x * X_DIM;
    int block_start_y = blockIdx.y * X_DIM;
    int block_start_z = blockIdx.z * Z_DIM;

    for (int offset = thread_idx; offset < X_DIM * X_DIM * Z_DIM; offset += total_threads) {
        int lscope_x = offset % X_DIM;
        int lscope_y = (offset / X_DIM) % X_DIM;
        int lscope_z = offset / (X_DIM * X_DIM);

        int gscope_x = block_start_x + lscope_x;
        int gscope_y = block_start_y + lscope_y;
        int gscope_z = block_start_z + lscope_z;

        if (gscope_x > 0 && gscope_x < N - 1 && gscope_y > 0 && gscope_y < N - 1 && gscope_z > 0 && gscope_z < N - 1) {
            shared_mem[offset] = 0.8f * (
                device_in[(gscope_z - 1) * N * N + gscope_y * N + gscope_x] +
                device_in[(gscope_z + 1) * N * N + gscope_y * N + gscope_x] +
                device_in[gscope_z * N * N + (gscope_y - 1) * N + gscope_x] +
                device_in[gscope_z * N * N + (gscope_y + 1) * N + gscope_x] +
                device_in[gscope_z * N * N + gscope_y * N + (gscope_x - 1)] +
                device_in[gscope_z * N * N + gscope_y * N + (gscope_x + 1)]
            );
        }
    }

    __syncthreads();

    for (int offset = thread_idx; offset < X_DIM * X_DIM * Z_DIM; offset += total_threads) {
        int lscope_x = offset % X_DIM;
        int lscope_y = (offset / X_DIM) % X_DIM;
        int lscope_z = offset / (X_DIM * X_DIM);

        int gscope_x = block_start_x + lscope_x;
        int gscope_y = block_start_y + lscope_y;
        int gscope_z = block_start_z + lscope_z;

        if (gscope_x > 0 && gscope_x < N - 1 && gscope_y > 0 && gscope_y < N - 1 && gscope_z > 0 && gscope_z < N - 1) {
            device_out[gscope_z * N * N + gscope_y * N + gscope_x] = shared_mem[offset];
        }
    }
}


__global__ void kernel2_4(const float *device_in, float *device_out) {
    __shared__ float shared_mem[X_DIM * X_DIM * Z_DIM];

    int thread_idx = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;

    int block_start_x = blockIdx.x * X_DIM;
    int block_start_y = blockIdx.y * X_DIM;
    int block_start_z = blockIdx.z * Z_DIM;

    for (int offset = thread_idx; offset < X_DIM * X_DIM * Z_DIM; offset += total_threads) {
        int lscope_x = offset % X_DIM;
        int lscope_y = (offset / X_DIM) % X_DIM;
        int lscope_z = offset / (X_DIM * X_DIM);

        int gscope_x = block_start_x + lscope_x;
        int gscope_y = block_start_y + lscope_y;
        int gscope_z = block_start_z + lscope_z;

        if (gscope_x > 0 && gscope_x < N - 1 && gscope_y > 0 && gscope_y < N - 1 && gscope_z > 0 && gscope_z < N - 1) {
            shared_mem[offset] = 0.8f * (
                device_in[(gscope_z - 1) * N * N + gscope_y * N + gscope_x] +
                device_in[(gscope_z + 1) * N * N + gscope_y * N + gscope_x] +
                device_in[gscope_z * N * N + (gscope_y - 1) * N + gscope_x] +
                device_in[gscope_z * N * N + (gscope_y + 1) * N + gscope_x] +
                device_in[gscope_z * N * N + gscope_y * N + (gscope_x - 1)] +
                device_in[gscope_z * N * N + gscope_y * N + (gscope_x + 1)]
            );
        }
    }

    __syncthreads();

    for (int offset = thread_idx; offset < X_DIM * X_DIM * Z_DIM; offset += total_threads) {
        int lscope_x = offset % X_DIM;
        int lscope_y = (offset / X_DIM) % X_DIM;
        int lscope_z = offset / (X_DIM * X_DIM);

        int gscope_x = block_start_x + lscope_x;
        int gscope_y = block_start_y + lscope_y;
        int gscope_z = block_start_z + lscope_z;

        if (gscope_x > 0 && gscope_x < N - 1 && gscope_y > 0 && gscope_y < N - 1 && gscope_z > 0 && gscope_z < N - 1) {
            device_out[gscope_z * N * N + gscope_y * N + gscope_x] = shared_mem[offset];
        }
    }
}


__global__ void kernel2_8(const float *device_in, float *device_out) {
    __shared__ float shared_mem[X_DIM * X_DIM * Z_DIM];

    int thread_idx = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;

    int block_start_x = blockIdx.x * X_DIM;
    int block_start_y = blockIdx.y * X_DIM;
    int block_start_z = blockIdx.z * Z_DIM;

    for (int offset = thread_idx; offset < X_DIM * X_DIM * Z_DIM; offset += total_threads) {
        int lscope_x = offset % X_DIM;
        int lscope_y = (offset / X_DIM) % X_DIM;
        int lscope_z = offset / (X_DIM * X_DIM);

        int gscope_x = block_start_x + lscope_x;
        int gscope_y = block_start_y + lscope_y;
        int gscope_z = block_start_z + lscope_z;

        if (gscope_x > 0 && gscope_x < N - 1 && gscope_y > 0 && gscope_y < N - 1 && gscope_z > 0 && gscope_z < N - 1) {
            shared_mem[offset] = 0.8f * (
                device_in[(gscope_z - 1) * N * N + gscope_y * N + gscope_x] +
                device_in[(gscope_z + 1) * N * N + gscope_y * N + gscope_x] +
                device_in[gscope_z * N * N + (gscope_y - 1) * N + gscope_x] +
                device_in[gscope_z * N * N + (gscope_y + 1) * N + gscope_x] +
                device_in[gscope_z * N * N + gscope_y * N + (gscope_x - 1)] +
                device_in[gscope_z * N * N + gscope_y * N + (gscope_x + 1)]
            );
        }
    }

    __syncthreads();

    for (int offset = thread_idx; offset < X_DIM * X_DIM * Z_DIM; offset += total_threads) {
        int lscope_x = offset % X_DIM;
        int lscope_y = (offset / X_DIM) % X_DIM;
        int lscope_z = offset / (X_DIM * X_DIM);

        int gscope_x = block_start_x + lscope_x;
        int gscope_y = block_start_y + lscope_y;
        int gscope_z = block_start_z + lscope_z;

        if (gscope_x > 0 && gscope_x < N - 1 && gscope_y > 0 && gscope_y < N - 1 && gscope_z > 0 && gscope_z < N - 1) {
            device_out[gscope_z * N * N + gscope_y * N + gscope_x] = shared_mem[offset];
        }
    }
}

__global__ void kernel2_part3(const float *device_in, float *device_out) {
    __shared__ float shared_mem[X_DIM * X_DIM * Z_DIM];

    int thread_idx = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;

    int block_start_x = blockIdx.x * X_DIM;
    int block_start_y = blockIdx.y * X_DIM;
    int block_start_z = blockIdx.z * Z_DIM;

    for (int offset = thread_idx; offset < X_DIM * X_DIM * Z_DIM; offset += total_threads * 2) {
        
        int curr_off = offset;

        if (curr_off < X_DIM * X_DIM * Z_DIM) {
            int lscope_x = curr_off % X_DIM;
            int lscope_y = (curr_off / X_DIM) % X_DIM;
            int lscope_z = curr_off / (X_DIM * X_DIM);

            int gscope_x = block_start_x + lscope_x;
            int gscope_y = block_start_y + lscope_y;
            int gscope_z = block_start_z + lscope_z;

            if (gscope_x > 0 && gscope_x < N - 1 && gscope_y > 0 && gscope_y < N - 1 && gscope_z > 0 && gscope_z < N - 1) {
                shared_mem[curr_off] = 0.8f * (
                    device_in[(gscope_z - 1) * N * N + gscope_y * N + gscope_x] +
                    device_in[(gscope_z + 1) * N * N + gscope_y * N + gscope_x] +
                    device_in[gscope_z * N * N + (gscope_y - 1) * N + gscope_x] +
                    device_in[gscope_z * N * N + (gscope_y + 1) * N + gscope_x] +
                    device_in[gscope_z * N * N + gscope_y * N + (gscope_x - 1)] +
                    device_in[gscope_z * N * N + gscope_y * N + (gscope_x + 1)]
                );
                device_out[gscope_z * N * N + gscope_y * N + gscope_x] = shared_mem[curr_off];
            }
        }

        curr_off += total_threads;
        if (curr_off < X_DIM * X_DIM * Z_DIM) {
            int lscope_x = curr_off % X_DIM;
            int lscope_y = (curr_off / X_DIM) % X_DIM;
            int lscope_z = curr_off / (X_DIM * X_DIM);

            int gscope_x = block_start_x + lscope_x;
            int gscope_y = block_start_y + lscope_y;
            int gscope_z = block_start_z + lscope_z;

            if (gscope_x > 0 && gscope_x < N - 1 && gscope_y > 0 && gscope_y < N - 1 && gscope_z > 0 && gscope_z < N - 1) {
                shared_mem[curr_off] = 0.8f * (
                    device_in[(gscope_z - 1) * N * N + gscope_y * N + gscope_x] +
                    device_in[(gscope_z + 1) * N * N + gscope_y * N + gscope_x] +
                    device_in[gscope_z * N * N + (gscope_y - 1) * N + gscope_x] +
                    device_in[gscope_z * N * N + (gscope_y + 1) * N + gscope_x] +
                    device_in[gscope_z * N * N + gscope_y * N + (gscope_x - 1)] +
                    device_in[gscope_z * N * N + gscope_y * N + (gscope_x + 1)]
                );
                device_out[gscope_z * N * N + gscope_y * N + gscope_x] = shared_mem[curr_off];
            }
        }
    }
}





__host__ void stencil(const float *host_in, float *host_out) { 
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      for (int k = 1; k < N - 1; k++) {
        host_out[i * N * N + j * N + k] = 0.8f * (host_in[(i - 1) * N * N + j * N + k]
                                          + host_in[(i + 1) * N * N + j * N + k]
                                          + host_in[i * N * N + (j - 1) * N + k]
                                          + host_in[i * N * N + (j + 1) * N + k]
                                          + host_in[i * N * N + j * N + (k - 1)]
                                          + host_in[i * N * N + j * N + (k + 1)]);
      }
    }
  }
}

// TODO: Edit the function definition as required
__host__ void check_result(const float* w_ref, const float* w_opt, const uint64_t size) { 
  float maxdiff = 0.0f;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        float this_diff = w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " differences found over threshold " << THRESHOLD
         << "; Max Difference = " << maxdiff << endl<<endl;
  } else {
    cout << "No differences found between base and test versions\n";
    flag = 1;
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
    uint64_t SIZE = N * N * N;
    vector<string> speedups;

    
    float *cpu_in = new float[SIZE];
    float *cpu_out = new float[SIZE];

    
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < SIZE; ++i) {
        cpu_in[i] = static_cast<float>(rand() % 100000 + 1);
    }
    fill_n(cpu_out, SIZE, 0.0f);

    
    double clkbegin = rtclock();
    stencil(cpu_in, cpu_out);
    double clkend = rtclock();
    double cpu_time = (clkend - clkbegin) * 1000;
    cout << "Stencil time on CPU: " << cpu_time << " msec" << endl << endl;
    // TODO: Fill in kernel1
    // TODO: Adapt check_result() and invoke
    // Part 1 -----------------------------------------------------
    float *device_in, *device_out;
    cudaCheckError(cudaMalloc(&device_in, SIZE * sizeof(float)));
    cudaCheckError(cudaMalloc(&device_out, SIZE * sizeof(float)));
    cudaCheckError(cudaMemcpy(device_in, cpu_in, SIZE * sizeof(float), cudaMemcpyHostToDevice));

    
    cudaEvent_t start, end;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&end));

    int number_of_blocks = (SIZE + threads_per_block - 1) / threads_per_block;
    float *cpu_out_device = new float[SIZE];
    fill_n(cpu_out_device, SIZE, 0.0f);

    
    cudaCheckError(cudaEventRecord(start));
    kernel1<<<number_of_blocks, threads_per_block>>>(device_in, device_out);
    cudaCheckError(cudaMemcpy(cpu_out_device, device_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaEventRecord(end));
    cudaCheckError(cudaEventSynchronize(end));

    float kernel_time;
    cudaCheckError(cudaEventElapsedTime(&kernel_time, start, end));
    cout << "Time taken by kernel1 execution is: " << kernel_time << " ms" << endl;

    check_result(cpu_out, cpu_out_device, N);
    if (flag == 1) {
        flag = 0;
        cout << "Between kernel1 and cpu stencil implementation " << endl << endl;
    }
    speedups.push_back("Speedup of kernel1 over stencil = " + to_string(cpu_time / kernel_time));
     // TODO: Fill in kernel2
    // TODO: Adapt check_result() and invoke
    // Part 2 -----------------------------------------------------
float *device_out2, *device_out3, *device_out4, *device_out5;
cudaCheckError(cudaMalloc(&device_out2, SIZE * sizeof(float)));
cudaCheckError(cudaMalloc(&device_out3, SIZE * sizeof(float)));
cudaCheckError(cudaMalloc(&device_out4, SIZE * sizeof(float)));
cudaCheckError(cudaMalloc(&device_out5, SIZE * sizeof(float)));

dim3 dimBlock, dimGrid;
vector<pair<int, dim3>> kernel_configs = {
    {1, dim3(X_DIM, block_side_1, Z_DIM)},
    {2, dim3(X_DIM, block_side_2, Z_DIM)},
    {4, dim3(X_DIM, block_side_4, Z_DIM)},
    {8, dim3(X_DIM, block_side_8, Z_DIM)}
};
vector<float*> device_outputs = {device_out2, device_out3, device_out4, device_out5};

for (int i = 0; i < kernel_configs.size(); ++i) {
    fill_n(cpu_out_device, SIZE, 0.0f);
    dimBlock = kernel_configs[i].second;
    dimGrid = dim3(X_GRID, Y_GRID, Z_GRID);

    cudaCheckError(cudaEventRecord(start));

    switch (kernel_configs[i].first) {
        case 1: kernel2_1<<<dimGrid, dimBlock>>>(device_in, device_outputs[i]); break;
        case 2: kernel2_2<<<dimGrid, dimBlock>>>(device_in, device_outputs[i]); break;
        case 4: kernel2_4<<<dimGrid, dimBlock>>>(device_in, device_outputs[i]); break;
        case 8: kernel2_8<<<dimGrid, dimBlock>>>(device_in, device_outputs[i]); break;
    }

    cudaCheckError(cudaMemcpy(cpu_out_device, device_outputs[i], SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaEventRecord(end));
    cudaCheckError(cudaEventSynchronize(end));
    cudaCheckError(cudaEventElapsedTime(&kernel_time, start, end));

    cout << "Time taken by kernel2_" << kernel_configs[i].first << " with block side = " 
         << kernel_configs[i].first << " execution is: " << kernel_time << " ms" << endl;
    check_result(cpu_out, cpu_out_device, N);
    if (flag == 1) {
        flag = 0;
        cout << "Between kernel2_" << kernel_configs[i].first << " and cpu stencil implementation " << endl << endl;
    }
    speedups.push_back("Speedup of kernel2_" + to_string(kernel_configs[i].first) + 
                       " over stencil = " + to_string(cpu_time / kernel_time));
}


    // Part 3 -----------------------------------------------------
    fill_n(cpu_out_device, SIZE, 0.0f);
    dim3 dimBlock5(X_DIM, block_side_2, Z_DIM);
    dim3 dimGrid5(X_GRID, Y_GRID, Z_GRID);
    float *device_out6;
    cudaCheckError(cudaMalloc(&device_out6, SIZE * sizeof(float)));
    cudaCheckError(cudaEventRecord(start));
    kernel2_part3<<<dimGrid5, dimBlock5>>>(device_in, device_out6);
    cudaCheckError(cudaMemcpy(cpu_out_device, device_out6, SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaEventRecord(end));
    cudaCheckError(cudaEventSynchronize(end));
    cudaCheckError(cudaEventElapsedTime(&kernel_time, start, end));

    cout << "Time taken by kernel2_part3 with block side = 2 execution is: " << kernel_time << " ms" << endl;
    check_result(cpu_out, cpu_out_device, N);
    if (flag == 1) {
        flag = 0;
        cout << "Between kernel2_part3 and cpu stencil implementation " << endl << endl;
    }
    speedups.push_back("Speedup of kernel2_part3 over stencil = " + to_string(cpu_time / kernel_time));

    // Part 4 -----------------------------------------------------
    // Using pinned memory
    float *cpu_in_pinned, *cpu_out_pinned, *device_out7;
    cudaCheckError(cudaHostAlloc(&cpu_in_pinned, SIZE * sizeof(float), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc(&cpu_out_pinned, SIZE * sizeof(float), cudaHostAllocDefault));
    memcpy(cpu_in_pinned, cpu_in, SIZE * sizeof(float));
    fill_n(cpu_out_pinned, SIZE, 0.0f);

    cudaCheckError(cudaMemcpy(device_in, cpu_in_pinned, SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMalloc(&device_out7, SIZE * sizeof(float)));
    cudaCheckError(cudaEventRecord(start));
    kernel2_part3<<<dimGrid5, dimBlock5>>>(device_in, device_out7);
    cudaCheckError(cudaMemcpy(cpu_out_pinned, device_out7, SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaEventRecord(end));
    cudaCheckError(cudaEventSynchronize(end));
    cudaCheckError(cudaEventElapsedTime(&kernel_time, start, end));

    cout << "Time taken by kernel2_part4 execution is = " << kernel_time << " ms" << endl;
    check_result(cpu_out, cpu_out_pinned, N);
    if (flag == 1) {
        flag = 0;
        cout << "Between kernel2_part4 and cpu stencil implementation " << endl << endl;
    }
    speedups.push_back("Speedup of kernel2_part4 pinned memory over stencil = " + to_string(cpu_time / kernel_time));

    // Part 5 -----------------------------------------------------
    // Using unified memory
    float *cpu_in_unified, *cpu_out_unified;
    cudaCheckError(cudaMallocManaged(&cpu_in_unified, SIZE * sizeof(float)));
    cudaCheckError(cudaMallocManaged(&cpu_out_unified, SIZE * sizeof(float)));
    memcpy(cpu_in_unified, cpu_in, SIZE * sizeof(float));
    fill_n(cpu_out_unified, SIZE, 0.0f);

    cudaCheckError(cudaEventRecord(start));
    kernel2_part3<<<dimGrid5, dimBlock5>>>(cpu_in_unified, cpu_out_unified);
    cudaCheckError(cudaEventRecord(end));
    cudaCheckError(cudaEventSynchronize(end));
    cudaCheckError(cudaEventElapsedTime(&kernel_time, start, end));

    cout << "Time taken by kernel2_part5 execution is = " << kernel_time << " ms" << endl;
    check_result(cpu_out, cpu_out_unified, N);
    if (flag == 1) {
        flag = 0;
        cout << "Between kernel2_part5 and cpu stencil implementation " << endl << endl;
    }
    speedups.push_back("Speedup of kernel2_part5 unified memory over stencil = " + to_string(cpu_time / kernel_time));

    // Printing all speedups
    cout << endl;
    for (const auto &speedup : speedups) {
        cout << speedup << endl;
    }

    // TODO: Free memory
    cudaCheckError(cudaFree(device_in));
    cudaCheckError(cudaFree(device_out));
    for (auto out : device_outputs) {
        cudaCheckError(cudaFree(out));
    }
    cudaCheckError(cudaFree(device_out6));
    cudaCheckError(cudaFree(device_out7));
    cudaCheckError(cudaFree(cpu_in_unified));
    cudaCheckError(cudaFree(cpu_out_unified));
    cudaCheckError(cudaFreeHost(cpu_in_pinned));
    cudaCheckError(cudaFreeHost(cpu_out_pinned));

    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(end));

    delete[] cpu_in;
    delete[] cpu_out;
    delete[] cpu_out_device;

    return EXIT_SUCCESS;
}

