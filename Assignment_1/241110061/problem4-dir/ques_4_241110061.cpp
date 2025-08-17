#include <cassert>

#include <chrono>

#include <iostream>

#include <papi.h>

#include <vector>

using namespace std;
using namespace std::chrono;
using HR = high_resolution_clock;
using HRTimer = HR::time_point;
#define N (2048)
void matmul_ijk(const uint32_t * A,const uint32_t * B, uint32_t * C,const int SIZE)
{
	for(int i = 0; i < SIZE; i++)
	{
		for(int j = 0; j < SIZE; j++)
		{
			uint32_t sum = 0.0;
			for(int k = 0; k < SIZE; k++)
			{
				sum += A[i * SIZE + k] * B[k * SIZE + j];
			}
			C[i * SIZE + j] += sum;
		}
	}
}
void matmul_ijk_blocking(const uint32_t * A,const uint32_t * B, uint32_t * C,	const int SIZE)
{
	for(int i = 0; i < SIZE; i++)
	{
		for(int j = 0; j < SIZE; j++)
		{
			uint32_t sum = 0.0;
			for(int k = 0; k < SIZE; k++)
			{
				sum += A[i * SIZE + k] * B[k * SIZE + j];
			}
			C[i * SIZE + j] += sum;
		}
	}
}
// 									this function is to send same block size that we are using in blocking during MxM
void matmul_ijk_blocking_2(const uint32_t * A,const uint32_t * B, uint32_t * C,	const int n,const int blk)
{
	for(int ii = 0; ii < n; ii += blk)
	{
		for(int jj = 0; jj < n; jj += blk)
		{
			for(int kk = 0; kk < n; kk += blk)
			{
				for(int i = ii; i < min(blk + ii, n); ++i)
				{
					for(int j = jj; j < min(blk + jj, n); ++j)
					{
						uint32_t sum = 0.0;
						for(int k = kk; k < min(blk + kk, n); ++k)
						{
							sum += A[i * n + k] * B[k * n + j];
						}
						C[i * n + j] += sum;
					}
				}
			}
		}
	}
}
// 									this function is to send different block size that we are using in blocking during MxM
void matmul_ijk_blocking_3(const uint32_t * A, const uint32_t * B, uint32_t * C, const int n, const int blk1, const int blk2, const int blk3)
{
	for(int ii = 0; ii < n; ii += blk1)
	{
		for(int jj = 0; jj < n; jj += blk2)
		{
			for(int kk = 0; kk < n; kk += blk3)
			{
				for(int i = ii; i < min(blk1 + ii, n); ++i)
				{
					for(int j = jj; j < min(blk2 + jj, n); ++j)
					{
						uint32_t sum = 0.0;
						for(int k = kk; k < min(blk3 + kk, n); ++k)
						{
							sum += A[i * n + k] * B[k * n + j];
						}
						C[i * n + j] += sum;
					}
				}
			}
		}
	}
}
void init(uint32_t * mat, const int SIZE)
{
	for(int i = 0; i < SIZE; i++)
	{
		for(int j = 0; j < SIZE; j++)
		{
			mat[i * SIZE + j] = 1;
		}
	}
}
void print_matrix(const uint32_t * mat,	const int SIZE)
{
	for(int i = 0; i < SIZE; i++)
	{
		for(int j = 0; j < SIZE; j++)
		{
			cout << mat[i * SIZE + j] << "\t";
		}
		cout << "\n";
	}
}
void check_result(const uint32_t * ref,	const uint32_t * opt,
		const int SIZE)
{
	for(int i = 0; i < SIZE; i++)
	{
		for(int j = 0; j < SIZE; j++)
		{
			if(ref[i * SIZE + j] != opt[i * SIZE + j])
			{
				assert(false && "Diff found between sequential and blocked versions!\n");
			}
		}
	}
}
// above all are user defined functions
int main()
{
	uint32_t * A = new uint32_t[N * N];
	uint32_t * B = new uint32_t[N * N];
	uint32_t * C_seq = new uint32_t[N * N];
	
	vector < double > block_mean_time;
	vector < vector < long long int >> events_store; //ans;
	
	HRTimer start;
	HRTimer end;
	double unblocked_mean_time = 0;
	int blk[5] = {
		4,
		8,
		16,
		32,
		64
	};
	double duration = 0;
	init(A, N);
	init(B, N);
	cout<<"Program execution started"<<endl;
	
	
	for(int i = 0; i < 3; ++i)
	{
	//cout<<i<<endl;
		init(C_seq, N);
		//                                          this is the start of matrix multiploaction implementation without blocking
		start = HR::now();
		matmul_ijk(A, B, C_seq, N);
		end = HR::now();
		unblocked_mean_time += duration_cast < microseconds > (end - start).count();
		//                                                                        without blocking MxM ends here
	}
	unblocked_mean_time /= 3; //        						mean of time used in unblocked MxM - 
	
	
	//									Same blocking execution
	
	
	uint32_t * C_blk = new uint32_t[N * N];
	for(int i = 0; i < 5; ++i)
	{
	//cout<<i<<endl;
		duration = 0;
		for(int j = 0; j < 3; ++j)
		{
			init(C_blk, N);
			//    							papi counters tracking for all blocking methods
			int retval = PAPI_library_init(PAPI_VER_CURRENT);
			if(retval != PAPI_VER_CURRENT && retval > 0)
			{
				cerr << "PAPI library version mismatch: " << retval << " != " << PAPI_VER_CURRENT << "\n";
				exit(EXIT_FAILURE);
			}
			else if(retval < 0)
			{
				cerr << "PAPI library initialization error: " << retval << " != " << PAPI_VER_CURRENT << "\n";
				exit(EXIT_FAILURE);
			}
			int eventset = PAPI_NULL;
			retval = PAPI_create_eventset( & eventset);
			if(PAPI_OK != retval)
			{
				cerr << "Error at PAPI_create_eventset()" << endl;
				exit(EXIT_FAILURE);
			}
			if (PAPI_add_named_event(eventset, "perf::PERF_COUNT_HW_CACHE_L1D:ACCESS") != PAPI_OK) {
        		std::cerr << "PAPI add native event in perf::PERF_COUNT_HW_CACHE_L1D:READS!" << std::endl;
        		return -1;
	    		}
	    		if (PAPI_add_named_event(eventset, "perf::PERF_COUNT_HW_CACHE_L1D:MISS") != PAPI_OK) {
				cerr << "PAPI add native event error in perf::PERF_COUNT_HW_CACHE_L1D:MISS!" << std::endl;
				return -1;
	    		}
		      	if (PAPI_add_event(eventset, PAPI_L2_DCH) != PAPI_OK) {
				cerr << "Error in PAPI_add_event PAPI_L2_DCH!\n";
		       		 exit(EXIT_FAILURE);
		      	}
		      	if (PAPI_add_event(eventset, PAPI_L2_DCM) != PAPI_OK) {
				cerr << "Error in PAPI_add_event PAPI_L2_DCM!\n";
		       		 exit(EXIT_FAILURE);
		      	}
		      	//if (PAPI_add_named_event(eventset, "perf::CACHE-MISSES") != PAPI_OK) {
			//	std::cerr << "PAPI add native event error in perf::CACHE-MISSES!" << std::endl;
			//	return -1;
	    		//}
	    		// if (PAPI_add_named_event(eventset, "PERF_COUNT_HW_CPU_CYCLES") != PAPI_OK) {
				// std::cerr << "PAPI add native event error in perf::CPU_CLOCK!" << std::endl;
				// return -1;
	    		// }
			if (PAPI_add_named_event(eventset, "L2_PREFETCH_HIT_L3") != PAPI_OK) {
				std::cerr << "PAPI add native event error in L2_PREFETCH_HIT_L3!" << std::endl;
				return -1;
	    		}


			if(PAPI_OK != retval)
			{
				cerr << "Error at PAPI_create_eventset()" << endl;
				exit(EXIT_FAILURE);
			}
			
			retval = PAPI_start(eventset);
			if(PAPI_OK != retval)
			{
				cerr << "Error at PAPI_start()" << endl;
				exit(EXIT_FAILURE);
			}
			// 										starting the blocking method MxM
			start = HR::now();
			matmul_ijk_blocking_2(A, B, C_blk, N, blk[i]);
			end = HR::now();
			duration += duration_cast < microseconds > (end - start).count();
			long long int arr[5] = {
				0
			};
			retval = PAPI_stop(eventset, arr);
			if(PAPI_OK != retval)
			{
				cerr << "Error at PAPI_stop()" << endl;
				exit(EXIT_FAILURE);
			}
			vector < long long int > temp;
			temp.push_back(arr[0]);
			temp.push_back(arr[1]);
			temp.push_back(arr[2]);
			temp.push_back(arr[3]);
			temp.push_back(arr[4]);
			
			events_store.push_back(temp);
			PAPI_cleanup_eventset(eventset);
			PAPI_destroy_eventset( & eventset);
			PAPI_shutdown();
		}
		duration /= 3;
		block_mean_time.push_back(duration);
	}
	
	
	
	
	
	
	// 									these array are implemented to allot different block sizes to each of A,B,C
	int blk1[5] = {
		4,
		8,
		4,
		16,
		32
	}; // for A
	int blk2[5] = {
		8,
		8,
		8,
		4,
		32
	}; // for B
	int blk3[5] = {
		16,
		16,
		4,
		32,
		64
	}; // for C
	// implementing differetn block sizes now
	
	for(int i = 0; i < 5; ++i)
	{
	//cout<<i<<endl;
		uint32_t * C_blk = new uint32_t[N * N];
		duration = 0;
		for(int j = 0; j < 3; ++j)
		{
			init(C_blk, N);
			//    										papi counters tracking for all blocking methods
			int retval = PAPI_library_init(PAPI_VER_CURRENT);
			if(retval != PAPI_VER_CURRENT && retval > 0)
			{
				cerr << "PAPI library version mismatch: " << retval << " != " << PAPI_VER_CURRENT << "\n";
				exit(EXIT_FAILURE);
			}
			else if(retval < 0)
			{
				cerr << "PAPI library initialization error: " << retval << " != " << PAPI_VER_CURRENT << "\n";
				exit(EXIT_FAILURE);
			}
			int eventset = PAPI_NULL;
			retval = PAPI_create_eventset( & eventset);
			if(PAPI_OK != retval)
			{
				cerr << "Error at PAPI_create_eventset()" << endl;
				exit(EXIT_FAILURE);
			}
			if (PAPI_add_named_event(eventset, "perf::PERF_COUNT_HW_CACHE_L1D:ACCESS") != PAPI_OK) {
        		std::cerr << "PAPI add native event in perf::PERF_COUNT_HW_CACHE_L1D:ACCESS!" << std::endl;
        		return -1;
	    		}
	    		if (PAPI_add_named_event(eventset, "perf::PERF_COUNT_HW_CACHE_L1D:MISS") != PAPI_OK) {
				cerr << "PAPI add native event error in perf::PERF_COUNT_HW_CACHE_L1D:MISS!" << std::endl;
				return -1;
	    		}
		      	if (PAPI_add_event(eventset, PAPI_L2_DCH) != PAPI_OK) {
				cerr << "Error in PAPI_add_event PAPI_L2_DCH!\n";
		       		 exit(EXIT_FAILURE);
		      	}
		      	if (PAPI_add_event(eventset, PAPI_L2_DCM) != PAPI_OK) {
				cerr << "Error in PAPI_add_event PAPI_L2_DCM!\n";
		       		 exit(EXIT_FAILURE);
		      	}
		      	//if (PAPI_add_named_event(eventset, "perf::CACHE-MISSES") != PAPI_OK) {
				//std::cerr << "PAPI add native event error in perf::CACHE-MISSES!" << std::endl;
				//return -1;
	    		//}
    			// if (PAPI_add_named_event(eventset, "PERF_COUNT_HW_CPU_CYCLES") != PAPI_OK) {
				// std::cerr << "PAPI add native event error in perf::CPU-CLOCK!" << std::endl;
				// return -1;
	    		// }
			if (PAPI_add_named_event(eventset, "L2_PREFETCH_HIT_L3") != PAPI_OK) {
				cerr << "PAPI add native event error in L2_PREFETCH_HIT_L3!" << std::endl;
				return -1;
	    		}


			retval = PAPI_start(eventset);
			if(PAPI_OK != retval)
			{
				cerr << "Error at PAPI_start()" << endl;
				exit(EXIT_FAILURE);
			}
			// 										starting the blocking method MxM
			start = HR::now();
			matmul_ijk_blocking_3(A, B, C_blk, N, blk1[i], blk2[i], blk3[i]);
			end = HR::now();
			duration += duration_cast < microseconds > (end - start).count();
			
			long long int arr[5] = {
				0
			};
			retval = PAPI_stop(eventset, arr);
			if(PAPI_OK != retval)
			{
				cerr << "Error at PAPI_stop()" << endl;
				exit(EXIT_FAILURE);
			}
			vector < long long int > temp;
			temp.push_back(arr[0]);
			temp.push_back(arr[1]);
			temp.push_back(arr[2]);
			temp.push_back(arr[3]);
			temp.push_back(arr[4]);
			
			events_store.push_back(temp);
			
			PAPI_cleanup_eventset(eventset);
			PAPI_destroy_eventset( & eventset);
			PAPI_shutdown();
		}
		duration /= 3;
		block_mean_time.push_back(duration);
		if(i==3){
			cout<<"The program execution is just going to end"<<endl;
		}
	}
	
	// Now output part and calculation of speedup down
	
	
	// this is for speedup calculation
	vector < double > speedups;
	vector < vector < long long int >> data_counters;
	vector < string > str;
	str.push_back("Blocking used for A is 4, B is 4, C is 4.");
	str.push_back("Blocking used for A is 8, B is 8, C is 8.");
	str.push_back("Blocking used for A is 16, B is 16, C is 16.");
	str.push_back("Blocking used for A is 32, B is 32, C is 32.");
	str.push_back("Blocking used for A is 64, B is 64, C is 64.");
	str.push_back("Blocking used for A is 4, B is 8, C is 16.");
	str.push_back("Blocking used for A is 8, B is 8, C is 16.");
	str.push_back("Blocking used for A is 4, B is 8, C is 4.");
	str.push_back("Blocking used for A is 16, B is 4, C is 32.");
	str.push_back("Blocking used for A is 32, B is 32, C is 64.");
	for(int i = 0; i < 5; ++i)
	{
		speedups.push_back(unblocked_mean_time / block_mean_time[i]);
	}
	for(int i = 0; i < 5; ++i)
	{
		speedups.push_back(unblocked_mean_time / block_mean_time[i + 5]);
	}
	double max = speedups[0];
	
	
	int index = 0;
	for(int i = 0; i < 10; ++i)
	{
		cout << str[i] << " The speedup achieved is = " << speedups[i] << endl;
		if(speedups[i] > max)
		{
			index = i;
			max = speedups[i];
		}
	}
	cout << endl << "The max speedup is achieved when " << endl << str[index] <<" And the value of max speedup is ="<< " " << max << endl;
	cout << "##########################################################################" << endl;
	cout << "PAPI performance counters used are {perf::PERF_COUNT_HW_CACHE_L1D:ACCESS, perf::PERF_COUNT_HW_CACHE_L1D:MISS, PAPI_L2_DCM, PAPI_L2_DCH ,L2_PREFETCH_HIT_L3}" << endl;
	
	for(int i=0; i<30; i+=3){
		vector<long long int> temp;
		for(int j=0; j<4; ++j){
			long long sum= events_store[i][j] + events_store[i+1][j] + events_store[i+2][j] ;
			temp.push_back(sum/3);
		}
		data_counters.push_back(temp);
	}
	
	for(int i=0; i<10; ++i){
		cout<<str[i]<<endl<<"The average L1 data cache access are = "<<data_counters[i][0]<<endl<<"The average L1 data cache misses are = "<<data_counters[i][1]<<endl<<"The average L2 data cache hits are = "<<data_counters[i][2]<<endl<<"The average L2 data cache misses are = "<<data_counters[i][3]<<endl<<"The average L2 prefetch hits in L3 are are = "<<data_counters[i][4]<<endl<<endl; 
	}
	
	check_result(C_seq, C_blk, N);
	return EXIT_SUCCESS;
}
