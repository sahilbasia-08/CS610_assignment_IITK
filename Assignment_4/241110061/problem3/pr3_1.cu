#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <ctime>
#include <vector>
#include <iostream>

int threadsPerBlock = 512;  // Threads per block (modifiable)
int blocksPerGrid = 1024;   // Blocks per grid (modifiable)

void performGridSearch(double paramB[30], double paramA[120], double thresholdK);

double paramA[120];
double paramB[30];

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

// CUDA kernel function
__global__ void computeKernel(double* output, double* paramB, double* paramC, double* paramE, int* gridSteps, long offset) {
  long index = blockIdx.x * blockDim.x + threadIdx.x;
  long tempIndex = index + offset;
  int gridPositions[10];
  
  for (int i = 9; i >= 0; --i) {
    gridPositions[i] = tempIndex % gridSteps[i];
    tempIndex /= gridSteps[i];
  }

  double computedValues[10];
  __shared__ double sharedMatrix[120];
  index *= 10;

  for (int i = 0; i < 10; ++i) {
    computedValues[i] = paramB[3 * i] + gridPositions[i] * paramB[3 * i + 2];
  }

  if (threadIdx.x < 120) {
    sharedMatrix[threadIdx.x] = paramC[threadIdx.x];
  }
  __syncthreads();

  double comparisonResult[10];
  for (int i = 0; i < 10; ++i) {
    comparisonResult[i] = -sharedMatrix[10 * 10 + i];
    for (int j = 0; j < 10; ++j) {
      comparisonResult[i] += sharedMatrix[j * 10 + i] * computedValues[j];
    }
    if (fabs(comparisonResult[i]) > paramE[i]) {
      return;
    }
  }

  for (int i = 0; i < 10; ++i)
    output[index + i] = computedValues[i];
}

int main() {
  int indexA = 0, indexB = 0;

  FILE* fileA = fopen("./disp.txt", "r");
  if (fileA == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }
  while (!feof(fileA) && fscanf(fileA, "%lf", &paramA[indexA]) == 1) {
    indexA++;
  }
  fclose(fileA);

  FILE* fileB = fopen("./grid.txt", "r");
  if (fileB == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }
  while (!feof(fileB) && fscanf(fileB, "%lf", &paramB[indexB]) == 1) {
    indexB++;
  }
  fclose(fileB);

  double thresholdK = 0.3;

  cudaEvent_t start, stop;
  float elapsedTime;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));

  cudaCheckError(cudaEventRecord(start, 0));
  printf("Running program, please wait...\n");
  performGridSearch(paramB, paramA, thresholdK);
  cudaCheckError(cudaEventRecord(stop, 0));
  cudaCheckError(cudaEventSynchronize(stop));
  cudaCheckError(cudaEventElapsedTime(&elapsedTime, start, stop));

  printf("Total time = %f seconds\n", elapsedTime / 1000.0f);

  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(stop));

  return EXIT_SUCCESS;
}

__host__ inline void performGridSearch(double paramB[30], double paramA[120], double thresholdK) {
  long pointsCount = 0;
  long linesSaved = 0;

  double* d_gridValues;
  cudaCheckError(cudaMalloc((void**)&d_gridValues, 30 * sizeof(double)));
  cudaCheckError(cudaMemcpy(d_gridValues, paramB, 30 * sizeof(double), cudaMemcpyHostToDevice));

  double thresholdArray[10];
  double* d_thresholdArray;
  cudaCheckError(cudaMalloc((void**)&d_thresholdArray, 10 * sizeof(double)));

  double matrix[12][10];
  double* d_matrix;
  cudaCheckError(cudaMalloc((void**)&d_matrix, 120 * sizeof(double)));
  for (int i = 0; i < 120; i++) {
    matrix[i % 12][i / 12] = paramA[i];
  }
  cudaCheckError(cudaMemcpy(d_matrix, matrix, 120 * sizeof(double), cudaMemcpyHostToDevice));

  FILE* outputFile = fopen("./results-v3_1.txt", "w");
  if (outputFile == NULL) {
    printf("Error in creating file!");
    exit(1);
  }

  for (int i = 0; i < 10; ++i)
    thresholdArray[i] = thresholdK * matrix[11][i];
  cudaCheckError(cudaMemcpy(d_thresholdArray, thresholdArray, 10 * sizeof(double), cudaMemcpyHostToDevice));

  int gridSteps[10];
  int* d_gridSteps;
  cudaCheckError(cudaMalloc((void**)&d_gridSteps, 10 * sizeof(int)));
  for (int i = 0; i < 10; ++i)
    gridSteps[i] = floor((paramB[3 * i + 1] - paramB[3 * i]) / paramB[3 * i + 2]);
  cudaCheckError(cudaMemcpy(d_gridSteps, gridSteps, 10 * sizeof(int), cudaMemcpyHostToDevice));

  long totalLoops = 1;
  for (int i = 0; i < 10; ++i) totalLoops *= gridSteps[i];

  long offset = 0;
  
  double* h_output, *d_output;
  h_output = static_cast<double*>(malloc(threadsPerBlock * blocksPerGrid * 10 * sizeof(double)));
  cudaCheckError(cudaHostAlloc((void**)&d_output, threadsPerBlock * blocksPerGrid * 10 * sizeof(double), cudaHostAllocDefault));

  while (totalLoops) {
    int step = threadsPerBlock * blocksPerGrid;
    if (step > totalLoops) {
      if (blocksPerGrid == 1) {
        threadsPerBlock >>= 1;
      } else {
        blocksPerGrid >>= 1;
      }
    } else {
      cudaCheckError(cudaMemset(d_output, 0, threadsPerBlock * blocksPerGrid * 10 * sizeof(double)));
      computeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_gridValues, d_matrix, d_thresholdArray, d_gridSteps, offset);
      cudaCheckError(cudaMemcpy(h_output, d_output, threadsPerBlock * blocksPerGrid * 10 * sizeof(double), cudaMemcpyDeviceToHost));

      for (long j = 0; j < blocksPerGrid * threadsPerBlock; ++j) {
        if (h_output[j * 10] == 0) continue;

        ++pointsCount;
        ++linesSaved;

        for (int k = 0; k < 10; ++k) {
          if (k < 9) {
            fprintf(outputFile, "%.6f\t", h_output[j * 10 + k]);
          } else {
            fprintf(outputFile, "%.6f", h_output[j * 10 + k]);
          }
        }
        fprintf(outputFile, "\n");

        if (linesSaved >= 11608) {
          fclose(outputFile);
          free(h_output);
          cudaFree(d_gridValues);
          cudaFree(d_matrix);
          cudaFree(d_thresholdArray);
          cudaFree(d_gridSteps);
          cudaFreeHost(d_output);
          printf("\nResult points: %ld\n", pointsCount);  
          return;
        }
      }

      offset += step;
      totalLoops -= step;
    }
  }

  cudaDeviceSynchronize();
  fclose(outputFile);
  printf("\nResult points: %ld\n", pointsCount);

  free(h_output);
  cudaFree(d_gridValues);
  cudaFree(d_matrix);
  cudaFree(d_thresholdArray);
  cudaFree(d_gridSteps);
  cudaFreeHost(d_output);
}

