#include<iostream>
#include<pthread.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<algorithm>
using namespace std;
// 三个变量 分别对应矩阵规模 线程数量 分块大小
// 第四个变量为静态/动态划分
const int N = 10000;
const int arrlen = 10000; // arrlen是常量，实验时不用变
const int thread_count = 20;
const int segment = N / 50;
const int block_num = N / segment; // 分块数量
int index_size = block_num / thread_count; // 每个线程对应的index数量
int flag[block_num]; // 保证不重复取样
float matrix[N][arrlen];
float copymatrix[N][arrlen];
int next_arr = 0; // 动态分配
int next_n = N / 20; // 粗粒度动态分配
pthread_mutex_t mutex;
// linux sort()函数并不是冒泡排序，自己手动实现一个
void bubblesort(float *start, float *end) {
    // 从小到大 比较相邻元素，相邻元素顺序错误则交换
    for(int i = arrlen - 2; i >= 0; i--) {
        for(int j = i; j < arrlen - 1; j++) {
            if(*(start + j) > *(start + j + 1)) {
                swap(*(start + j), *(start + j + 1));
            }
            else {
                // 当小于下一个值时，则直接跳出即可，移动结束
                break;
            }
        }
    }
}
void init_matrix() {
        // 四种类型，完全升序 1/4逆序+3/4升序 1/2逆序+1/2升序 完全逆序
    int seg = N / 4;
    int ratio;
    srand(unsigned(time(NULL)));
    for(int i = 0; i < N; i++) {
        if(i < seg) {
            ratio = 0;
        }
        else if(i < seg * 2) {
            ratio = 32;
        }
        else if(i < seg * 3) {
            ratio = 64;
        }
        else {
            ratio = 128;
        }
        // 每一行都有ratio/128的概率逆序，(128-ratio)/128的概率升序
        if((rand() & 127) < ratio) {
            // 按位与运算比取模运算快得多，因此用按位与。
            // 逆序
            for(int j = 0; j < arrlen; j++) {
                matrix[i][j] = arrlen - j;
            }
        }
        else {
            // 升序
            for(int j = 0; j < arrlen; j++) {
                matrix[i][j] = j;
            }
        }
    }
}
void copy_matrix() {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < arrlen; j++) {
            copymatrix[i][j] = matrix[i][j];
        }
    }
}
void *static_array_sort(void *arg) {
    long long thread_id = (long long)arg; // index数组
    clock_t start = clock();
    for(int i = thread_id * index_size * segment; i < (thread_id + 1) * index_size * segment; i++) {
        // 如果用sort就会出现升序反而慢的魔幻情况，因为sort在数据量大的时候用的是快排
        stable_sort(copymatrix[i], copymatrix[i] + arrlen);
    }
    clock_t End = clock();
    cout << "thread " << thread_id << " use " << (double)(End - start) / CLOCKS_PER_SEC * 1000
    << "ms to complete" << endl;
    pthread_exit(NULL);
}
void *static_array_sort_interval(void *arg) {
    long long thread_id = (long long)arg; // index数组
    clock_t start = clock();
    for(int i = 0; i < index_size; i++) {
        int id = i * thread_count + thread_id;
        for(int j = id * segment; j < (id + 1) * segment; j++) {
            stable_sort(copymatrix[j], copymatrix[j] + arrlen);
        }
        // cout << "already sort from " << id * segment << " to " << (id + 1) * segment << endl;
    }
    clock_t End = clock();
    cout << "thread " << thread_id << " use " << (double)(End - start) / CLOCKS_PER_SEC * 1000
    << "ms to complete" << endl;
    pthread_exit(NULL);
}
void *static_array_sort_random(void *arg) {
    long long thread_id = (long long)arg; // index数组
    // 随机生成block
    int index[index_size];
    int index_num = 0; // private
    while(index_num < index_size) {
        int ran = rand() % block_num;
        // 临界区
        pthread_mutex_lock(&mutex);
        if(!flag[ran]) {
            index[index_num] = ran;
            flag[ran] = 1;
            index_num++;
        }
        pthread_mutex_unlock(&mutex);
    }
    // 从排序部分开始计时
    clock_t start = clock();
    for(int i = 0; i < index_size; i++) {
        int idx = index[i];
        for(int j = idx * segment; j < (idx + 1) * segment; j++) {
            stable_sort(copymatrix[j], copymatrix[j] + arrlen);
        }
    }
    // cout << "already sort from " << id * segment << " to " << (id + 1) * segment << endl;
    clock_t End = clock();
    cout << "thread " << thread_id << " use " << (double)(End - start) / CLOCKS_PER_SEC * 1000
    << "ms to complete" << endl;
    pthread_exit(NULL);
}
void *dynamic_array_sort(void *arg) {
    long long thread_id = (long long)arg;
    int task = 0;
    clock_t start = clock();
    while(1) {
        pthread_mutex_lock(&mutex);
        task = next_arr;
        next_arr += next_n;
        pthread_mutex_unlock(&mutex);
        if(task >= N) {
            break;
        }
        for(int i = task; i < task + next_n; i++)
            stable_sort(copymatrix[task], copymatrix[task] + arrlen);
    }
    clock_t End = clock();
    cout << "thread " << thread_id << " use " << (double)(End - start) / CLOCKS_PER_SEC * 1000
    << "ms to complete" << endl;
    pthread_exit(NULL);
}
int main() {
    init_matrix();
    copy_matrix();
    pthread_t pthread[thread_count];
    // 每个线程在(thread_id * index_size, (thread_id + 1) * index_size) * segment范围内排序
    cout << "static continuation sort:" << endl;
    for(long long i = 0; i < thread_count; i++) {
        pthread_create(&pthread[i], NULL, static_array_sort, (void*)i);
    }
    for(int i = 0; i < thread_count; i++) {
        pthread_join(pthread[i], NULL);
    }
    cout << endl << endl;
    // 每个线程在(k * thread_count + thread_id)(k<=index_size)范围内排序
    copy_matrix();
    cout << "static interval sort:" << endl;
    for(long long i = 0; i < thread_count; i++) {
        pthread_create(&pthread[i], NULL, static_array_sort_interval, (void*)i);
    }
    for(int i = 0; i < thread_count; i++) {
        pthread_join(pthread[i], NULL);
    }
    cout << endl << endl;
    // 每个线程随机到对应的行进行排序
    copy_matrix();
    memset(flag, 0, sizeof(flag));
    pthread_mutex_init(&mutex, NULL);
    cout << "static random sort:" << endl;
    for(long long i = 0; i < thread_count; i++) {
        pthread_create(&pthread[i], NULL, static_array_sort_random, (void*)i);
    }
    for(int i = 0; i < thread_count; i++) {
        pthread_join(pthread[i], NULL);
    }
    cout << endl << endl;
    pthread_mutex_destroy(&mutex);
    // 动态分配线程
    copy_matrix();
    pthread_mutex_init(&mutex, NULL);
    cout << "dymanic sort:" << endl;
    for(long long i = 0; i < thread_count; i++) {
        pthread_create(&pthread[i], NULL, dynamic_array_sort, (void*)i);
    }
    for(int i = 0; i < thread_count; i++) {
        pthread_join(pthread[i], NULL);
    }
    cout << endl << endl;
    pthread_mutex_destroy(&mutex);
    return 0;
}
