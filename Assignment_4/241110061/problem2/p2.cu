#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

using namespace std;
using namespace std::chrono;

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


void exclusive_prefix_sum(uint32_t* dev_output, uint32_t* dev_input, int total_elements, int threads_per_block);

__host__ void thrust_sum(const uint32_t* input, uint32_t* output, const uint64_t N) {
    thrust::device_vector<uint32_t> d_input(input, input + N);
    thrust::device_vector<uint32_t> d_output(N);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    thrust::copy(d_output.begin(), d_output.end(), output);
}


__global__ void cuda_sum(uint32_t* dev_output, uint32_t* dev_input, uint32_t* block_sums, int total_elements, int elems_per_block) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int offset = block_id * elems_per_block;

    __shared__ uint32_t shared_data[1024];

    
    shared_data[2 * thread_id] = (2 * thread_id + offset < total_elements) ? dev_input[offset + 2 * thread_id] : 0;
    shared_data[2 * thread_id + 1] = (2 * thread_id + 1 + offset < total_elements) ? dev_input[offset + 2 * thread_id + 1] : 0;

    

    
    for (int jump_idx = 1; jump_idx < elems_per_block; jump_idx *= 2) {
        int index = (thread_id + 1) * jump_idx * 2 - 1;
        if (index < elems_per_block) {
            shared_data[index] += shared_data[index - jump_idx];
        }
        __syncthreads();
    }

   
    if (thread_id == 0) {
        block_sums[block_id] = shared_data[elems_per_block - 1];
        shared_data[elems_per_block - 1] = 0;  
    }
    

    
    for (int jump_idx = elems_per_block / 2; jump_idx > 0; jump_idx /= 2) {
        int index = (thread_id + 1) * jump_idx * 2 - 1;
        if (index < elems_per_block) {
            uint32_t temp = shared_data[index - jump_idx];
            shared_data[index - jump_idx] = shared_data[index];
            shared_data[index] += temp;
        }
        __syncthreads();
    }

    
    if (2 * thread_id + offset < total_elements) dev_output[offset + 2 * thread_id] = shared_data[2 * thread_id];
    if (2 * thread_id + 1 + offset < total_elements) dev_output[offset + 2 * thread_id + 1] = shared_data[2 * thread_id + 1];
}


__global__ void add_block_sums(uint32_t* dev_output, uint32_t* block_sums_output, int total_elements, int elems_per_block) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int offset = block_id * elems_per_block;

    uint32_t add_value = block_sums_output[block_id];
    if (offset + thread_id < total_elements) {
        dev_output[offset + thread_id] += add_value;
    }
}


__host__ void exclusive_prefix_sum(uint32_t* dev_output, uint32_t* dev_input, int total_elements, int threads_per_block) {
    int elems_per_block = 2 * threads_per_block;
    int block_count = (total_elements + elems_per_block - 1) / elems_per_block;

    uint32_t* block_sums;
    uint32_t* block_sums_output;
    cudaCheckError(cudaMalloc(&block_sums, block_count * sizeof(uint32_t)));
    cudaCheckError(cudaMalloc(&block_sums_output, block_count * sizeof(uint32_t)));

    
    cuda_sum<<<block_count, threads_per_block>>>(dev_output, dev_input, block_sums, total_elements, elems_per_block);
    cudaCheckError(cudaDeviceSynchronize());

   
    if (block_count > 1) {
        exclusive_prefix_sum(block_sums_output, block_sums, block_count, threads_per_block);
        add_block_sums<<<block_count, elems_per_block>>>(dev_output, block_sums_output, total_elements, elems_per_block);
        cudaCheckError(cudaDeviceSynchronize());
    }

    cudaCheckError(cudaFree(block_sums));
    cudaCheckError(cudaFree(block_sums_output));
}

__host__ void check_result(const uint32_t* w_ref, const uint32_t* w_opt, const uint64_t size) {
    for (uint64_t i = 0; i < size; i++) {
        if (w_ref[i] != w_opt[i]) {
            cout << "Differences found between the two arrays.\n";
            assert(false);
        }
    }
    cout << "No differences found between base and test versions\n";
}

int main(int argc, char* argv[]) {
    uint64_t N = (argc > 1) ? atoll(argv[1]) : (1 << 24);
    int threads_per_block = (argc > 2) ? atoi(argv[2]) : 512;

    auto* h_input = new uint32_t[N];
    fill_n(h_input, N, 1);
    auto total_size = N * sizeof(uint32_t);

    
    auto* h_thrust_ref = new uint32_t[N];
    fill_n(h_thrust_ref, N, 0);
    auto start = high_resolution_clock::now();
    thrust_sum(h_input, h_thrust_ref, N);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();

    uint32_t *dev_input, *dev_output;
    cudaCheckError(cudaMallocManaged(&dev_input, total_size));
    cudaCheckError(cudaMallocManaged(&dev_output, total_size));
    copy(h_input, h_input + N, dev_input);

    cudaEvent_t kernel_start, kernel_end;
    float kernel_time;
    cudaCheckError(cudaEventCreate(&kernel_start));
    cudaCheckError(cudaEventCreate(&kernel_end));

    cudaCheckError(cudaEventRecord(kernel_start));
    exclusive_prefix_sum(dev_output, dev_input, N, threads_per_block);
    cudaCheckError(cudaEventRecord(kernel_end));
    cudaCheckError(cudaEventSynchronize(kernel_end));

    cudaCheckError(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end));

    cout << "Time taken by Thrust implementation: " << duration << " ms" << endl;
    cout << "Time taken by CUDA implementation: " << kernel_time << " ms" << endl;
    check_result(h_thrust_ref, dev_output, N);
    cout << endl;
    cout << "CUDA speedup over Thrust: " << duration / kernel_time << endl << endl;
    cout << "Last value in CUDA output: " << dev_output[N - 1] << endl;
    cout << "Last value in Thrust output: " << h_thrust_ref[N - 1] << endl;

    cudaCheckError(cudaFree(dev_input));
    cudaCheckError(cudaFree(dev_output));
    delete[] h_thrust_ref;
    delete[] h_input;

    cudaCheckError(cudaEventDestroy(kernel_start));
    cudaCheckError(cudaEventDestroy(kernel_end));

    return EXIT_SUCCESS;
}

