// Coarse-grained locking implies 1 lock for the whole map
// Fine-grained locking implies 1 lock for each key in the map, which is
// encouraged

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <pthread.h>
#include <queue>
#include <string>
#include <unistd.h>
using std::cerr;
using std::cout;
using std::endl;
using std::ios;
// Max different files
const int MAX_FILES = 10;
const int MAX_SIZE = 10;
int MAX_THREADS = 5;


struct t_data {
  uint32_t tid;
};

// struct to keep track of the number of occurrences of a word
struct word_tracker {
  //uint64_t word_count[5]; 						// i have commented this. This is because if have made this a local variable for each thread. 
  									// now each thread will seperately count the words of the file they have opened
									// this reduces the false sharing, as earlier this was shared variable. So multiple threads will try to write the variable
									// and this can result in false sharing
  // each uint64_t is 8 bytes
  uint64_t total_lines_processed;  					
  // added a padding of '0' character to fill the whole cache block. The cache line size is 64 byte
  // unint64_t is 8 byte { used sizeof(total_lines_processed) to check that
  char padding_1[64-sizeof(uint64_t)];       				
  uint64_t total_words_processed;
  // similarly here did the same thing of adding padding. this whole structure and it's content is shared
  // therefore different threads will access this shared content. So to reduce false sharing, padding is added
  char padding_2[64-sizeof(uint64_t)];       					
  pthread_mutex_t word_count_mutex;   
         
} tracker;

// Shared queue, to be read by producers
std::queue<std::string> shared_pq;
// updates to shared queue
pthread_mutex_t pq_mutex = PTHREAD_MUTEX_INITIALIZER;

// lock var to update to total line counter
pthread_mutex_t line_count_mutex = PTHREAD_MUTEX_INITIALIZER;

// each thread read a file and put the tokens in line into std out
void *thread_runner(void *);

void print_usage(char *prog_name) {
  cerr << "usage: " << prog_name << " <producer count> <input file>\n";
  exit(EXIT_FAILURE);
}

// void print_counters() {
//   for (int id = 0; id < MAX_THREADS; ++id) {
//     std::cout << "Thread " << id << " counter: " << tracker.word_count[id]
//               << '\n';
//   }
// }

// this is function to enfore that number of threads given in command line argument must be equal to number of total files to read
void check(int file_count, std::string input){ 
  // finding the number of file counts 
  // According to question if we have N files to read, then N threads must be present as each thread will open one file
  // So this below functionality test wheather the above written condition is satisfied or not
  std::fstream file;
  file.open(input);
  if(!file){
    cerr<<"!Error in opening the input file"<<endl;
    exit(1);
  }
  std::string temp;
  while(!file.eof()){
    getline(file, temp);
    file_count++;
  }
  file.close();

  if(file_count!=MAX_THREADS){
    cerr<<endl<<"Error in command, enter the number of threads same as no of files to read in input file"<<endl;
    cout<<"For given input file, their are "<<file_count<<" files to read. So give number of threads as 5."<<endl;
    exit(1);
  }
}

void fill_producer_buffer(std::string &input) {
  std::fstream input_file;
  input_file.open(input, ios::in);
  if (!input_file.is_open()) {
    cerr << "Error opening the top-level input file!" << endl;
    exit(EXIT_FAILURE);
  }

  std::filesystem::path p(input);
  std::string line;
  while (getline(input_file, line)) {
    shared_pq.push(p.parent_path() / line);
  }
}

int thread_count = 0;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    print_usage(argv[0]);
  }

  thread_count = strtol(argv[1], NULL, 10);
  MAX_THREADS = thread_count;
  std::string input = argv[2];
  fill_producer_buffer(input);

  pthread_t threads_worker[thread_count];

  int file_count=0;
  // this function checks the condition of number of files = number of threads according to ques lanuage
  check(file_count, input); 

  struct t_data *args_array =
      (struct t_data *)malloc(sizeof(struct t_data) * thread_count);
  // for (int i = 0; i < thread_count; i++)
  //   tracker.word_count[i] = 0;
  tracker.total_lines_processed = 0;
  tracker.word_count_mutex = PTHREAD_MUTEX_INITIALIZER;
  for (int i = 0; i < thread_count; i++) {
    args_array[i].tid = i;
    pthread_create(&threads_worker[i], nullptr, thread_runner,
                   (void *)&args_array[i]);
  }

  for (int i = 0; i < thread_count; i++)
    pthread_join(threads_worker[i], NULL);

  //print_counters();			// i have implemented this printing counter functionality in each threads function
  cout << "Total words processed: " << tracker.total_words_processed << "\n";
  cout << "Total line processed: " << tracker.total_lines_processed << "\n";
  
  return EXIT_SUCCESS;
}



// TODO: inefficient counting of total words
void *thread_runner(void *th_args) {
  struct t_data *args = (struct t_data *)th_args;
  uint32_t thread_id = args->tid;
  std::fstream input_file;
  std::string fileName; 
  std::string line;
  // Added this word_count local variable for each threads. TO reduce false sharing bug, I added this
  uint64_t word_count=0;  // added these local variables for word count and line count for each thread
  			  // now each thread will not contend/ compete for this variable
  			  
  uint64_t line_count=0;               
  pthread_mutex_lock(&pq_mutex);
  fileName = shared_pq.front();
  shared_pq.pop();
  pthread_mutex_unlock(&pq_mutex);

  input_file.open(fileName.c_str(), ios::in);
  if (!input_file.is_open()) {
    cerr << "Error opening input file from a thread!" << endl;
    exit(EXIT_FAILURE);
  }

  while (getline(input_file, line)) {
    //pthread_mutex_lock(&line_count_mutex);
    //tracker.total_lines_processed++;
    //pthread_mutex_unlock(&line_count_mutex);
    std::string delimiter = " ";
    line_count++;
    size_t pos = 0;
    std::string token;
    while ((pos = line.find(delimiter)) != std::string::npos) {
      token = line.substr(0, pos);
      // false sharing: on word count // because their was false sharing on word count variable I moved it from shared to local variable.
      				      // For each thread now word count is different
      //tracker.word_count[thread_id]++;
      word_count++;
      
      // true sharing: on lock and total word variable
      
      //pthread_mutex_lock(&tracker.word_count_mutex);
      //tracker.total_words_processed++;
      //pthread_mutex_unlock(&tracker.word_count_mutex);
      line.erase(0, pos + delimiter.length());
    }
  }
  // moved total_lines_processed shared variable out of all loops to prevent repeated competetion to access the variable by each thread
  // the number of access to this shared variable are tremendously decreased
  pthread_mutex_lock(&line_count_mutex);
  tracker.total_lines_processed+=line_count;
  pthread_mutex_unlock(&line_count_mutex);
    
  pthread_mutex_lock(&tracker.word_count_mutex);
  tracker.total_words_processed+=word_count;  // I moved this total_word variable outside the inner loop for counting the number of words
					      // Since mutex were lock and release many numbeer of times inside the words counting loop, so their was to many access
					      //  to memory location for same shared variable by all the other threads
					      // to reduce true sharing
  pthread_mutex_unlock(&tracker.word_count_mutex);
  input_file.close();
  cout<<"Thread "<<thread_id<<" readed this many words "<<word_count+1<<endl; // here each thread is printing total number of words in file readed by that thread
								 // Here is some logical error, as according to question this word count is number of space in the file i.e. " "
								 // So i changed the output as word_count+1 -- LOGICAL ERROR
  pthread_exit(nullptr);
}
