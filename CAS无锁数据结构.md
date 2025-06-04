### CAS无锁数据结构



#### 无锁队列

```c++
#include <atomic>

using namespace std;

struct Node {
    Node(): next_(nullptr) {}
    Node(int num) : num_(num), next_(nullptr) {}
    int num_;
    atomic<Node*> next_;
};

class Queue {
public:
    Queue();
    ~Queue();
    void push_front(int n);
    void push_back(int n);
    void pop();
    Node* top();
private:
    atomic<Node*> head_;
    atomic<Node*> tail_;
};

Queue::Queue() {
    Node* dummy = new Node();
    head_.store(dummy);
    tail_.store(dummy);
}

Queue::~Queue() {
    Node* curr = head_.load();
    while(curr != nullptr) {
        Node* next = curr->next_.load();
        delete curr;
        curr = next;
    }
}

void Queue::push_back(int n) {
    Node* new_node = new Node(n);
    Node* curr_tail = tail_.load();
    Node* null_node = nullptr;
    do {
        while (curr_tail->next_ != nullptr) {
            curr_tail = curr_tail->next_;
        }
    } while (!curr_tail->next_.compare_exchange_weak(null_node, new_node));
    tail_.store(new_node);
}

void Queue::push_front_1(int n) {
    Node* new_node = new Node(n);
    Node* head = head_.load();
    do {
        new_node->next_.store(head);
    }
    while (!head_.compare_exchange_weak(head, new_node));
}
void Queue::push_front(int n) {
    Node* new_node = new Node(n);
    Node* head_next = head_.load()->next_;
    do {
        new_node->next_.store(head_next);
    }
    while (!head_.load()->next_.compare_exchange_weak(head_next, new_node));
}

void Queue::pop() {
    Node* curr_head = head_.load();
    do {
        if (curr_head->next_ == nullptr) return;
    } while (!head_.compare_exchange_weak(curr_head, curr_head->next_));
    delete curr_head;
}

//https://zhuanlan.zhihu.com/p/555622467
//https://blog.csdn.net/m0_61840987/article/details/145591041
```



```c++
#include <iostream>
#include <atomic>
#include <thread>
#include <stdexcept>

template <typename T, size_t Capacity>
class LockFreeArrayQueue {
private:
    static_assert(Capacity > 0, "Capacity must be positive");
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

    // 组合索引和计数器的结构，用于解决ABA问题
    struct alignas(16) IndexAndCounter {
        size_t index;
        size_t counter;
        
        bool operator==(const IndexAndCounter& other) const {
            return index == other.index && counter == other.counter;
        }
    };

    // 保证缓存行对齐，减少false sharing
    alignas(64) std::atomic<IndexAndCounter> head;
    alignas(64) std::atomic<IndexAndCounter> tail;
    T data[Capacity];

public:
    LockFreeArrayQueue() : head(IndexAndCounter{0, 0}), tail(IndexAndCounter{0, 0}) {}

    // 检查队列是否为空（线程安全，近似值）
    bool empty() const {
        IndexAndCounter current_head = head.load(std::memory_order_acquire);
        IndexAndCounter current_tail = tail.load(std::memory_order_acquire);
        return current_head.index == current_tail.index;
    }

    // 检查队列是否已满（线程安全，近似值）
    bool full() const {
        IndexAndCounter current_head = head.load(std::memory_order_acquire);
        IndexAndCounter current_tail = tail.load(std::memory_order_acquire);
        return ((current_tail.index + 1) & (Capacity - 1)) == current_head.index;
    }

    // 返回队列中元素数量（线程安全，近似值）
    size_t size() const {
        IndexAndCounter current_head = head.load(std::memory_order_acquire);
        IndexAndCounter current_tail = tail.load(std::memory_order_acquire);
        return (current_tail.index - current_head.index) & (Capacity - 1);
    }

    // 返回队列容量
    constexpr size_t capacity() const {
        return Capacity;
    }

    // 入队操作（线程安全，使用双字CAS解决ABA问题）
    bool enqueue(const T& value) {
        IndexAndCounter current_tail;
        IndexAndCounter new_tail;
        IndexAndCounter current_head;
        
        while (true) {
            current_tail = tail.load(std::memory_order_acquire);
            current_head = head.load(std::memory_order_acquire);
            new_tail.index = (current_tail.index + 1) & (Capacity - 1);
            
            // 检查队列是否已满
            if (new_tail.index == current_head.index) {
                return false;
            }
            
            // 准备新tail值（计数器+1）
            new_tail.counter = current_tail.counter + 1;
            
            // 尝试CAS操作
            if (tail.compare_exchange_weak(current_tail, new_tail,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
                break;
            }
            
            // CAS失败，重试
        }
        
        // 在获取的位置写入数据
        data[current_tail.index] = value;
        return true;
    }

    // 出队操作（线程安全，使用双字CAS解决ABA问题）
    bool dequeue(T& value) {
        IndexAndCounter current_head;
        IndexAndCounter new_head;
        IndexAndCounter current_tail;
        
        while (true) {
            current_head = head.load(std::memory_order_acquire);
            current_tail = tail.load(std::memory_order_acquire);
            
            // 检查队列是否为空
            if (current_head.index == current_tail.index) {
                return false;
            }
            
            // 预取数据
            value = data[current_head.index];
            
            // 准备新head值（索引+1，计数器+1）
            new_head.index = (current_head.index + 1) & (Capacity - 1);
            new_head.counter = current_head.counter + 1;
            
            // 尝试CAS操作
            if (head.compare_exchange_weak(current_head, new_head,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
                break;
            }
            
            // CAS失败，重试
        }
        return true;
    }

    // 尝试查看队首元素（线程安全）
    bool try_peek(T& value) const {
        IndexAndCounter current_head = head.load(std::memory_order_acquire);
        IndexAndCounter current_tail = tail.load(std::memory_order_acquire);
        
        if (current_head.index == current_tail.index) {
            return false;
        }
        
        value = data[current_head.index];
        return true;
    }
};
```



#### 生产者消费者

```c++
#include <queue>
#include <mutex>
#include <condition_variable>

class ProducerConsumer {
private:
    std::queue<int> buffer;          // 共享缓冲区（队列）
    size_t max_size;                 // 缓冲区最大容量
    std::mutex mtx;                  // 互斥锁，保护缓冲区的访问
    std::condition_variable not_full;  // 条件变量：缓冲区未满时通知生产者
    std::condition_variable not_empty; // 条件变量：缓冲区非空时通知消费者

public:
    ProducerConsumer(size_t size) : max_size(size) {}

    // 生产者函数：向缓冲区添加数据
    void produce(int data) {
        std::unique_lock<std::mutex> lock(mtx); // 自动加锁

        // 如果缓冲区已满，等待直到有空间
        while (buffer.size() >= max_size) {
            not_full.wait(lock);  // 释放锁，等待被唤醒后重新加锁
        }

        buffer.push(data);        // 生产数据
        std::cout << "Produced: " << data << std::endl;

        not_empty.notify_one();   // 通知一个消费者可以消费
        // lock在作用域结束后自动释放
    }

    // 消费者函数：从缓冲区取出数据
    int consume() {
        std::unique_lock<std::mutex> lock(mtx); // 自动加锁

        // 如果缓冲区为空，等待直到有数据
        while (buffer.empty()) {
            not_empty.wait(lock); // 释放锁，等待被唤醒后重新加锁
        }

        int data = buffer.front();
        buffer.pop();             // 消费数据
        std::cout << "Consumed: " << data << std::endl;

        not_full.notify_one();    // 通知一个生产者可以生产
        return data;
        // lock在作用域结束后自动释放
    }
};
```

