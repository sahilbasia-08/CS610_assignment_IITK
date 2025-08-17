#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/tuple.h>
#include <iostream>
#include <fstream>

#define NSEC_SEC_MUL (1.0e9)


const long batchSize = 1024 * 1024;

struct Params {
    double* b;
    double* a;
    double kk;
};

struct CalculateXValues {
    double* b;

    CalculateXValues(double* _b) : b(_b) {}

    __host__ __device__
    thrust::tuple<double, double, double, double, double, double, double, double, double, double> operator()(long index) const {
        double x[10];
        for (int i = 9; i >= 0; --i) {
            x[i] = b[3 * i] + (index % static_cast<int>((b[3 * i + 1] - b[3 * i]) / b[3 * i + 2])) * b[3 * i + 2];
            index /= static_cast<int>((b[3 * i + 1] - b[3 * i]) / b[3 * i + 2]);
        }
        return thrust::make_tuple(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]);
    }
};

struct CheckConstraints {
    double* a;
    double kk;

    CheckConstraints(double* _a, double _kk) : a(_a), kk(_kk) {}

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& x_tuple) const {
        double x[10] = {thrust::get<0>(x_tuple), thrust::get<1>(x_tuple), thrust::get<2>(x_tuple), thrust::get<3>(x_tuple), 
                        thrust::get<4>(x_tuple), thrust::get<5>(x_tuple), thrust::get<6>(x_tuple), thrust::get<7>(x_tuple), 
                        thrust::get<8>(x_tuple), thrust::get<9>(x_tuple)};

        double q[10];
        for (int i = 0; i < 10; ++i) {
            q[i] = -a[10 * i + 9];
            for (int j = 0; j < 10; ++j) {
                q[i] += a[10 * i + j] * x[j];
            }
        }

        double e[10] = {kk * a[9], kk * a[19], kk * a[29], kk * a[39], kk * a[49], 
                        kk * a[59], kk * a[69], kk * a[79], kk * a[89], kk * a[99]};
        
        for (int i = 0; i < 10; ++i) {
            if (fabs(q[i]) > e[i]) {
                return false;
            }
        }
        return true;
    }
};

void performGridSearch(thrust::host_vector<double>& h_b, thrust::host_vector<double>& h_a, double kk) {
    thrust::device_vector<double> d_b = h_b;
    thrust::device_vector<double> d_a = h_a;

    long total_combinations = 1;
    for (int i = 0; i < 10; ++i) {
        total_combinations *= static_cast<long>((h_b[3 * i + 1] - h_b[3 * i]) / h_b[3 * i + 2]);
    }

    int total_batches = (total_combinations + batchSize - 1) / batchSize;
    std::ofstream outFile("results-v4.txt");


    for (int batch_num = 0; batch_num < total_batches; ++batch_num) {
        long offset = batch_num * batchSize;
        long currentBatchSize = std::min(batchSize, total_combinations - offset);

        thrust::device_vector<thrust::tuple<double, double, double, double, double, double, double, double, double, double>> d_x_values(currentBatchSize);

        auto start = thrust::make_counting_iterator(offset);
        auto end = thrust::make_counting_iterator(offset + currentBatchSize);


        thrust::transform(thrust::device, start, end, d_x_values.begin(), CalculateXValues(thrust::raw_pointer_cast(d_b.data())));


        thrust::device_vector<thrust::tuple<double, double, double, double, double, double, double, double, double, double>> valid_x_values(currentBatchSize);

   
        auto valid_end = thrust::copy_if(thrust::device, d_x_values.begin(), d_x_values.end(), valid_x_values.begin(), CheckConstraints(thrust::raw_pointer_cast(d_a.data()), kk));

      
        thrust::host_vector<thrust::tuple<double, double, double, double, double, double, double, double, double, double>> h_valid_x_values(valid_x_values.begin(), valid_end);

        
        for (const auto& x_tuple : h_valid_x_values) {
            outFile << thrust::get<0>(x_tuple) << "\t" << thrust::get<1>(x_tuple) << "\t"
                    << thrust::get<2>(x_tuple) << "\t" << thrust::get<3>(x_tuple) << "\t"
                    << thrust::get<4>(x_tuple) << "\t" << thrust::get<5>(x_tuple) << "\t"
                    << thrust::get<6>(x_tuple) << "\t" << thrust::get<7>(x_tuple) << "\t"
                    << thrust::get<8>(x_tuple) << "\t" << thrust::get<9>(x_tuple) << "\n";
        }

       
        float progress = (static_cast<float>(batch_num + 1) / total_batches) * 100;
        printf("Progress: Batch %d/%d (%.2f%% complete)\n", batch_num + 1, total_batches, progress);
    }

    outFile.close();
    }

int main() {
    thrust::host_vector<double> h_a(120);
    FILE* file_a = fopen("./disp.txt", "r");
    if (!file_a) {
        printf("Error: could not open disp.txt\n");
        return 1;
    }
    for (int i = 0; i < 120; i++) {
        if (fscanf(file_a, "%lf", &h_a[i]) != 1) {
            printf("Error: fscanf failed while reading disp.txt\n");
            fclose(file_a);
            return 1;
        }
    }
    fclose(file_a);

    thrust::host_vector<double> h_b(30);
    FILE* file_b = fopen("./grid.txt", "r");
    if (!file_b) {
        printf("Error: could not open grid.txt\n");
        return 1;
    }
    for (int i = 0; i < 30; i++) {
        if (fscanf(file_b, "%lf", &h_b[i]) != 1) {
            printf("Error: fscanf failed while reading grid.txt\n");
            fclose(file_b);
            return 1;
        }
    }
    fclose(file_b);

    double kk = 0.3;
    performGridSearch(h_b, h_a, kk);

    return EXIT_SUCCESS;
}

