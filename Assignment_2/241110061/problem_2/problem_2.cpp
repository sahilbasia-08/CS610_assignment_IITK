#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <pthread.h>
#include <queue>
#include <math.h>

/*<pthread mutex t enter>
In case of shared buffer size < number of lines read by each thread, this mutex allows
each thread to atomically write his share in the shared buffer, without interference
from other producer threads. But consumer threads can compete for access to the
shared buffer.

>pthread mutex t mutex2
This mutex allows each thread to read content from the file(in my case I have reduced
I/O to only 1 time, by writing the content of file in one shared memory buffer). It
synchronizes the threads to read the content, without any race conditions. Also when
one thread is writing its content in a shared buffer, this mutex allows other threads
to store the content of the file(self-created buffer) in their local buffer parallelly with
other threads that is writing content in the shared buffer.

>pthread mutex t mutex
This mutex is used between the consumer thread and the producer thread. This
is used to synchronize these 2 threads to access the shared buffer, from where the
consumer will read content and write in a file, whereas the producer thread will write
the content of his local buffer in the shared buffer

>pthread barrier t BARRIER
This barrier prevents the threads from running their function unless all threads are
created. So each thread will execute it’s thread function only when all threads are
created.

>pthread cond t shared buffer not empty
This conditional variable is used in the context, of the producer. If the shared buffer
is full, then the producer have to wait until the consumer extracts and removes the
content of the shared buffer. When content of shared buffer is extracted by consumer,
then he will signal the waiting producer to write more content in shared buffer.

>pthread cond t shared buffer not full
Similarly, this conditional variable is used to tell the consumer that the buffer is not
full, and thus if the shared buffer has no data, then the consumer will wait
*/


using namespace std;

long long buffer_size_M = 0;
long long L = 0;
long long T = 0;
bool file_done = false;
string R; //paths input
string W; //paths output
bool exit_consumer = true; // this flag to tell consumer to exit
int total_threads_to_be_used=0;
pthread_barrier_t BARRIER; // barrier of threads
pthread_mutex_t enter=PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t shared_buffer_not_empty = PTHREAD_COND_INITIALIZER;
pthread_cond_t shared_buffer_not_full = PTHREAD_COND_INITIALIZER;

vector<string> shared_buffer;
vector<string> mem_buffer; // contains all data of the file
vector<string> writing_buffer; // contains all data that will be written to a file

void store_file(string R) {
    fstream temp_file(R);

    if(!temp_file){
        cerr<<"Either the input_file is not present or there is some error in opening file"<<endl;
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
        cerr<<"Either the output_writing_file is not present or there is some error in opening file"<<endl;
        exit(0);
    }
    for(int i=0; i<writing_buffer.size(); ++i){
        write_file<<writing_buffer[i]<<endl;  
    }
}

void* consumer_eat_file(void* arg) {
    while (exit_consumer) {

        //cout<<"enter consumer"<<endl;
        pthread_mutex_lock(&mutex);
        while (shared_buffer.empty() && !file_done) {
           //cout<<"consumer got wait"<<endl;
            pthread_cond_wait(&shared_buffer_not_empty, &mutex); // consumer will wait and release mutex lock incase the buffer is empty. one situation can be when consumer first takes the lock
        }
       // cout<<"consumer not wait"<<endl;
        if (shared_buffer.size()==0 && file_done) {
            pthread_mutex_unlock(&mutex);
            break; // in case the shared buffer empty and file data is also finished then consumer use finished
        }

        while (shared_buffer.size()!=0) {
            writing_buffer.push_back(shared_buffer.front());
            shared_buffer.erase(shared_buffer.begin());
        }

        pthread_cond_signal(&shared_buffer_not_full); //this is used to tell waiting thread that buffer is now empty
        pthread_mutex_unlock(&mutex);
    }
    //cout<<"consumer_exit()"<<endl;
    write_file(W);
    return NULL;
}

void* producer_read_file(void* arg) {
    pthread_barrier_wait(&BARRIER); // Wait for all producer threads to be created

    while (true) {
        vector<string> local_buff;
        pthread_mutex_lock(&mutex2);
        while (local_buff.size() < L && !mem_buffer.size()==0) {
            local_buff.push_back(mem_buffer.front());
            mem_buffer.erase(mem_buffer.begin());
        }
        pthread_mutex_unlock(&mutex2);

        if (local_buff.size()==0) { // this condition will check if some threads with 0 local_buff, then they will break form loop from here only
            pthread_mutex_lock(&mutex); // to give mutual exclussion to total_threads_to_be_used variabel access
            total_threads_to_be_used--; // this is done, cause maybe some thread with data is interleaved and consumer waiting for him, so only when all producer threads exit, then only the consumer will exit
            pthread_cond_signal(&shared_buffer_not_empty); // this tells consumer, to eat the content, this is because maybe one thread is waiting, when buffer was filled and mutex was taken by another producer thread. So if this threadu content emtpy then exit, but wake consumer 
            pthread_mutex_unlock(&mutex);
            break; // exiting the threads here only if local_buff is empty, as they have no use in writing to share_buffer
        }
        pthread_mutex_lock(&enter); // this is used to ensure one one thread atomically writes his complete share of L lines to the shared_buffer if shared_buffer.size()< L

        for (int i = 0; i < local_buff.size(); ++i) {
            pthread_mutex_lock(&mutex); // Ensure mutual exclusion for shared_buffer
            //cout<<"producer enter"<<endl;
            //cout<<pthread_self()<<endl;
            while (shared_buffer.size() >= buffer_size_M) {
                //cout << "producer waiting" << endl;
                pthread_cond_wait(&shared_buffer_not_full, &mutex);
            }
            
            //cout<<"producer not waiting"<<endl;
            shared_buffer.push_back(local_buff[i]);
            pthread_cond_signal(&shared_buffer_not_empty); // To signal the consumer, that data has been added and may not be empty. In case of shared_buffer size is large this will help

            pthread_mutex_unlock(&mutex);
        }

        // Unlock the enter mutex after all items are added to the shared_buffer
        pthread_mutex_unlock(&enter);
    }
    //cout<<"Producer_threads_exit();"<<endl;
    if (total_threads_to_be_used == 0) {
        file_done = true;
        exit_consumer = false;
        pthread_cond_broadcast(&shared_buffer_not_empty); // Notify consumer to exit
    }
    return NULL;
}

int main(int argc, char *argv[]){
    if(argc!=6){
        cout<<"To correctly run the program, run command - " << argv[0] << " [path_to_input_to_read]  [number_of_producer_threads] [number_of_line_each_thread_should_read] [size_of_shared_memory_buffer] [path_to_output_file]"<<endl;
        return 1;
    }

    R=argv[1]; // input file path
    W=argv[5]; // output file path
    T=stoi(argv[2]); // number of threads
    L=stoi(argv[3]); // number of lines to be read from each thread
    buffer_size_M=stoi(argv[4]);  // size of shared_buffer

    pthread_barrier_init(&BARRIER, NULL, T); // barrier setup
    store_file(R); // accessing the file only 1 time to reduce i/o operation at time of threads execution

    total_threads_to_be_used=T; // used this variable in producer function
    if(buffer_size_M==0){
        cerr<<"Error, the shared_buffer size is empty. Shared_buffer can'be empty"<<endl;
        return 0;
    }

    //checking output file
    if(!write_file){
        cerr<<"There is some error in opening file"<<endl;
        return 0;
    }
    
    //creating the vector threads
    vector<pthread_t> threads(T);
    for(int i=0; i<T; ++i){
        if(pthread_create(&threads[i], NULL, producer_read_file, NULL) !=0){
            cerr<<"Threads creation failed"<<endl;
            return 0;
        }
    }
    //creating the consumer thread
    pthread_t consumer;
    pthread_create(&consumer, NULL, consumer_eat_file, NULL);

    //waiting for producer and consumer thread to complete
    for(int i=0; i<T; ++i){
        if(pthread_join(threads[i], NULL)!=0){
            cerr<<"Joining threads failed"<<endl;
            return 0;
        }
    }
    
    pthread_join(consumer, NULL);
    
    //destroying all the mutex and barries and condvars
    pthread_barrier_destroy(&BARRIER);
    pthread_mutex_destroy(&mutex);
    pthread_mutex_destroy(&mutex2);
    
    pthread_cond_destroy(&shared_buffer_not_empty); 
    pthread_cond_destroy(&shared_buffer_not_full); 
    cout<<"PROGRAM EXECUTED SUCCESSFULLY"<<endl;
    

}
