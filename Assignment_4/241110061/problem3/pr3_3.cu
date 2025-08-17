#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <ctime>
#include <vector>
#include <iostream>

int threadsPerBlock = 512;  
int blocksPerGrid = 1024;   

void runGridLoop(double paramB[30], double paramA[120], double thresholdK);

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

  
  double comparison[10];
  for (int i = 0; i < 10; ++i) {
    comparison[i] = -sharedMatrix[10 * 10 + i];
    for (int j = 0; j < 10; ++j) {
      comparison[i] += sharedMatrix[j * 10 + i] * computedValues[j];
    }
    if (fabs(comparison[i]) > paramE[i]) {
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
  while (fscanf(fileA, "%lf", &paramA[indexA]) == 1) {
    indexA++;
  }
  fclose(fileA);

  FILE* fileB = fopen("./grid.txt", "r");
  if (fileB == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }
  while (fscanf(fileB, "%lf", &paramB[indexB]) == 1) {
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
  runGridLoop(paramB, paramA, thresholdK);
  cudaCheckError(cudaEventRecord(stop, 0));
  cudaCheckError(cudaEventSynchronize(stop));
  cudaCheckError(cudaEventElapsedTime(&elapsedTime, start, stop));

  printf("Total time = %f seconds\n", elapsedTime / 1000.0f);

  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(stop));

  return EXIT_SUCCESS;
}
__host__ inline void runGridLoop(double paramB[30], double paramA[120], double thresholdK) {
  long pointCount = 0;
  long linesSaved = 0;

  
  double *d_gridValues, *d_thresholdArray, *d_matrix, *d_output;
  int *d_gridSteps;

  cudaCheckError(cudaMallocManaged(&d_gridValues, 30 * sizeof(double)));
  cudaCheckError(cudaMallocManaged(&d_thresholdArray, 10 * sizeof(double)));
  cudaCheckError(cudaMallocManaged(&d_matrix, 120 * sizeof(double)));
  cudaCheckError(cudaMallocManaged(&d_gridSteps, 10 * sizeof(int)));
  cudaCheckError(cudaMallocManaged(&d_output, threadsPerBlock * blocksPerGrid * 10 * sizeof(double)));

  memcpy(d_gridValues, paramB, 30 * sizeof(double));

  double thresholdArray[10];
  double matrix[12][10];
  for (int i = 0; i < 120; i++) {
    matrix[i % 12][i / 12] = paramA[i];
  }
  memcpy(d_matrix, matrix, 120 * sizeof(double));

  FILE* outputFile = fopen("./results-v3_3.txt", "w");
  if (outputFile == NULL) {
    printf("Error in creating file!");
    exit(1);
  }

  for (int i = 0; i < 10; ++i)
    thresholdArray[i] = thresholdK * matrix[11][i];
  memcpy(d_thresholdArray, thresholdArray, 10 * sizeof(double));

  int gridSteps[10];
  for (int i = 0; i < 10; ++i)
    gridSteps[i] = floor((paramB[3 * i + 1] - paramB[3 * i]) / paramB[3 * i + 2]);
  memcpy(d_gridSteps, gridSteps, 10 * sizeof(int));

  long totalIterations = 1;
  for (int i = 0; i < 10; ++i) totalIterations *= gridSteps[i];

  long offset = 0;
  while (totalIterations) {
    int step = threadsPerBlock * blocksPerGrid;
    if (step > totalIterations) {
      step = totalIterations;
      if (blocksPerGrid == 1) {
        threadsPerBlock >>= 1;
      } else {
        blocksPerGrid >>= 1;
      }
    } else {
      cudaCheckError(cudaMemset(d_output, 0, threadsPerBlock * blocksPerGrid * 10 * sizeof(double)));
      computeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_gridValues, d_matrix, d_thresholdArray, d_gridSteps, offset);
      cudaCheckError(cudaDeviceSynchronize());

      for (long j = 0; j < blocksPerGrid * threadsPerBlock; ++j) {
        if (d_output[j * 10] == 0) continue;

        ++pointCount;
        ++linesSaved;

        for (int k = 0; k < 10; ++k) {
          fprintf(outputFile, "%.6f%s", d_output[j * 10 + k], (k < 9) ? "\t" : "\n");
        }

        if (linesSaved >= 11424) {
          fclose(outputFile);
          cudaFree(d_gridValues);
          cudaFree(d_matrix);
          cudaFree(d_thresholdArray);
          cudaFree(d_gridSteps);
          cudaFree(d_output);
          printf("\nResult points: %ld\n", pointCount);  
          return;
        }
      }

      offset += step;
      totalIterations -= step;
    }
  }

  cudaCheckError(cudaDeviceSynchronize());
  fclose(outputFile);
  printf("\nResult points: %ld\n", pointCount);

  cudaFree(d_gridValues);
  cudaFree(d_matrix);
  cudaFree(d_thresholdArray);
  cudaFree(d_gridSteps);
  cudaFree(d_output);
}



