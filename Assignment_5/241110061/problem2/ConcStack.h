#include <iostream>
#include <atomic>

using namespace std;

class Node {
public:
    int value;      
    Node* next;     
};


static atomic<Node*> top{nullptr};
static atomic<uint64_t> aba_pop_count{0};


void push(int value) {
    Node* new_node = new Node();
    new_node->value = value;

    Node* prev_top;
    do {
        prev_top = top.load(memory_order_relaxed);
        new_node->next = prev_top;
    } while (!top.compare_exchange_weak(prev_top, new_node));
}

int pop() {
    Node* prev_top;
    do {
        prev_top = top.load(memory_order_acquire);
        if (!prev_top) {
            return -1; 
        }

        uint64_t current_aba_pop_count = aba_pop_count.load(memory_order_relaxed);
        if (!aba_pop_count.compare_exchange_weak(current_aba_pop_count, current_aba_pop_count + 1)) {
            continue; 
        }

    } while (!top.compare_exchange_weak(prev_top, prev_top->next));

    int value = prev_top->value;
    delete prev_top; 
    return value;
}





