#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <unistd.h>
#include <vector>
#include <string>
#include <omp.h> // OpenMP library

using namespace std;

long long buffer_size_M = 0;
long long L = 0;
long long T = 0;
bool file_done = false;
string R; // Input file path
string W; // Output file path
bool exit_consumer = true;
int total_threads_to_be_used = 0;

bool shared_buffer_not_full = true;
bool shared_buffer_not_empty = false;

int t_count = 0;
vector<string> shared_buffer;
vector<string> mem_buffer; // Contains all data of the file
vector<string> writing_buffer; // Contains all data that will be written to a file

void store_file(string R) {
    fstream temp_file(R);
    if (!temp_file) {
        cerr << "Error opening input file." << endl;
        exit(0);
    }
    string str;
    while (getline(temp_file, str)) {
        mem_buffer.push_back(str);
    }
}

void write_file(string W) {
    ofstream write_file(W);
    if (!write_file) {
        cerr << "Error opening output file." << endl;
        exit(0);
    }
    for (const string &line : writing_buffer) {
        write_file << line << endl;
    }
}

// Condition wait function for producers (wait until shared_buffer_not_full is true)
void cond_wait1(omp_lock_t *lock) {
    while (!shared_buffer_not_full) {
        #pragma omp flush(shared_buffer_not_full)
        // Small pause to avoid busy-waiting

        usleep(1000);
    }
    omp_set_lock(lock); // Re-acquire the lock
}

// Condition wait function for consumers (wait until shared_buffer_not_empty is true)
void cond_wait2(omp_lock_t *lock) {
    while (!shared_buffer_not_empty && !file_done) {
        #pragma omp flush(shared_buffer_not_empty, file_done)

        usleep(1000); // Small pause to avoid busy-waiting
    }
    omp_set_lock(lock); // Re-acquire the lock
}

// Custom barrier for synchronizing producer threads
void barrier(omp_lock_t *lock4) {
    omp_set_lock(lock4);
    t_count++;
    omp_unset_lock(lock4);

    while (t_count < T); // Wait for all threads
}

// Producer function
void producer(omp_lock_t *lock, omp_lock_t *lock2, omp_lock_t *lock3, omp_lock_t *lock4) {
    barrier(lock); // Wait for all threads to start

    while (true) {
        vector<string> local_buff;

        // Critical section to get data from the memory buffer
        #pragma omp critical
        {
            while (local_buff.size() < L && !mem_buffer.empty()) {
                local_buff.push_back(mem_buffer.front());
                mem_buffer.erase(mem_buffer.begin());
            }
        }

        if (local_buff.empty()) {
            omp_set_lock(lock4);
            total_threads_to_be_used--;
            if (total_threads_to_be_used == 0) {
                file_done = true;
                exit_consumer = false;
                #pragma omp flush(shared_buffer_not_empty, file_done, exit_consumer)
            }
            omp_unset_lock(lock4);
            break;
        }

        // Write local buffer to shared buffer
        omp_set_lock(lock3);
        for (const string &line : local_buff) {
            omp_set_lock(lock);

            while (shared_buffer.size() >= buffer_size_M) {
                shared_buffer_not_full = false;
                //cout<<omp_get_thread_num()<<endl;
                #pragma omp flush(shared_buffer_not_full)
                omp_unset_lock(lock);
                cond_wait1(lock); // Wait until space is available in the buffer
            }
            shared_buffer.push_back(line);
            shared_buffer_not_empty = true;
            #pragma omp flush(shared_buffer_not_empty)
            omp_unset_lock(lock);
        }
        omp_unset_lock(lock3);

        local_buff.clear();
    }

    return;
}

// Consumer function
void consumer(omp_lock_t *lock) {
    while (exit_consumer) {
        omp_set_lock(lock);

        while (shared_buffer.empty() && !file_done) {
            shared_buffer_not_empty = false;
            #pragma omp flush(shared_buffer_not_empty)
            omp_unset_lock(lock);
            cond_wait2(lock); // Wait for buffer to have data
        }

        if (shared_buffer.empty() && file_done) {
            omp_unset_lock(lock);
            break;
        }

        // Transfer from shared buffer to writing buffer
        while (!shared_buffer.empty()) {
            writing_buffer.push_back(shared_buffer.front());
            shared_buffer.erase(shared_buffer.begin());
        }
        shared_buffer_not_full = true;
        #pragma omp flush(shared_buffer_not_full)
        omp_unset_lock(lock);
    }

    write_file(W);
    return;
}

int main(int argc, char* argv[]) {
    if(argc!=6){
        cout<<"To correctly run the program, run command - " << argv[0] << " [path_to_input_to_read]  [number_of_producer_threads] 	[number_of_line_each_thread_should_read] [size_of_shared_memory_buffer] [path_to_output_file]"<<endl;
        return 1;
    }

    R=argv[1]; // input file path
    W=argv[5]; // output file path
    T=stoi(argv[2]); // number of threads
    L=stoi(argv[3]); // number of lines to be read from each thread
    buffer_size_M=stoi(argv[4]);  // size of shared_buffer
    store_file(R);
    total_threads_to_be_used=T;
    // initializing the omp lock variable to use in custom barrier implmentation
    omp_lock_t lock, lock2, lock3, lock4;
    omp_init_lock(&lock);
    omp_init_lock(&lock2);
    omp_init_lock(&lock3);
    omp_init_lock(&lock4);
    if(buffer_size_M==0){
        cerr<<"Error, the shared_buffer size is empty. Shared_buffer can'be empty"<<endl;
        return 0;
    }
    //checking the output file
    if(!write_file){
        cerr<<"There is some error in opening file"<<endl;
        return 0;
    }
    if(T==0){
        cerr<<"In this case of T=0 threads, the program will not run correctly. Please give number of thread greater than 0"<<endl;
        exit(0);
    }
    cout<<"Program is running !.."<<endl;
    // This is my first approach
    // in this I used if else approach to make threads run the producer and consumer function
    // the logic is implmeneted in such a way that the threads will randomly pick any function
    // that is randomlly out of T+1 threads one will become consumer and rest prodcuer
    // both approach discussed below are giving smae result
    //
    // approach 1 starts
    /*int count=0; // making this shared variable
    omp_set_num_threads(T+1);
    #pragma omp parallel shared(count)
    {
        int thread_no; // using this count variable to keep check of how many threads to pass to producer function
        // this is critical section
        // becuase to ensure that each thread will atomically will
        // update the count variable to prevent any kind of race condition
        //
        #pragma omp critical
        {
            thread_no = count;
            count++;
        }
        // just mimicking the problem 2 of assignment 2
        //
        if(thread_no<T){ // Out of T+1 threads, T threas will be run producer function
            producer(&lock, &lock2, &lock3, &lock4);
        }
        else{       // rest will be consumer thread
            consumer(&lock4);
        }

    }

    approach 1 ends
    */
     
     // the second approach i implemented after the last class of openmp whihc talked about task and ssection 
    // this is my second approach of deploying the openmp threads
    // this is using the nested parallelism and sections constructs of
    // openmp and inside one section deploying more threads as producer thread
    // whereas the other section executed by left over thread and will be treated
    // as consumer threads
    omp_set_nested(1);
    #pragma omp parallel sections num_threads(2) // out of these 2 threads one will become the master
                                                 // threads of calling T producer threads
    {
    	#pragma omp section
    	{
    		#pragma omp parallel num_threads(T) // here T threads will be producer threads and in this one is master thread
                                                // that created other T-1 threads
    		{
    			producer(&lock, &lock2, &lock3, &lock4); // openmp locks given as arguments
                                                         // all these are shared
    		}
    	}
    	#pragma omp section // this will be run by other threads from omp parallel section part
                            // and thus therefore obeys the language of question as was given in assignment 2
    	{
    		consumer(&lock4);
    	}
    }


    // destroying the locks to release the resources
    omp_destroy_lock(&lock);
    omp_destroy_lock(&lock2);
    omp_destroy_lock(&lock3);
    omp_destroy_lock(&lock4);
    cout<<"PROGRAM EXECUTED SUCCESSFULLY"<<endl;
    return EXIT_SUCCESS;
}
