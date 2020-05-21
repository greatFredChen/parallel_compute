#include<iostream>
#include<pthread.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<algorithm>
using namespace std;
// �������� �ֱ��Ӧ�����ģ �߳����� �ֿ��С
// ���ĸ�����Ϊ��̬/��̬����
const int N = 10000;
const int arrlen = 10000; // arrlen�ǳ�����ʵ��ʱ���ñ�
const int thread_count = 20;
const int segment = N / 50;
const int block_num = N / segment; // �ֿ�����
int index_size = block_num / thread_count; // ÿ���̶߳�Ӧ��index����
int flag[block_num]; // ��֤���ظ�ȡ��
float matrix[N][arrlen];
float copymatrix[N][arrlen];
int next_arr = 0; // ��̬����
int next_n = N / 20; // �����ȶ�̬����
pthread_mutex_t mutex;
// linux sort()����������ð�������Լ��ֶ�ʵ��һ��
void bubblesort(float *start, float *end) {
    // ��С���� �Ƚ�����Ԫ�أ�����Ԫ��˳������򽻻�
    for(int i = arrlen - 2; i >= 0; i--) {
        for(int j = i; j < arrlen - 1; j++) {
            if(*(start + j) > *(start + j + 1)) {
                swap(*(start + j), *(start + j + 1));
            }
            else {
                // ��С����һ��ֵʱ����ֱ���������ɣ��ƶ�����
                break;
            }
        }
    }
}
void init_matrix() {
        // �������ͣ���ȫ���� 1/4����+3/4���� 1/2����+1/2���� ��ȫ����
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
        // ÿһ�ж���ratio/128�ĸ�������(128-ratio)/128�ĸ�������
        if((rand() & 127) < ratio) {
            // ��λ�������ȡģ�����ö࣬����ð�λ�롣
            // ����
            for(int j = 0; j < arrlen; j++) {
                matrix[i][j] = arrlen - j;
            }
        }
        else {
            // ����
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
    long long thread_id = (long long)arg; // index����
    clock_t start = clock();
    for(int i = thread_id * index_size * segment; i < (thread_id + 1) * index_size * segment; i++) {
        // �����sort�ͻ�������򷴶�����ħ���������Ϊsort�����������ʱ���õ��ǿ���
        stable_sort(copymatrix[i], copymatrix[i] + arrlen);
    }
    clock_t End = clock();
    cout << "thread " << thread_id << " use " << (double)(End - start) / CLOCKS_PER_SEC * 1000
    << "ms to complete" << endl;
    pthread_exit(NULL);
}
void *static_array_sort_interval(void *arg) {
    long long thread_id = (long long)arg; // index����
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
    long long thread_id = (long long)arg; // index����
    // �������block
    int index[index_size];
    int index_num = 0; // private
    while(index_num < index_size) {
        int ran = rand() % block_num;
        // �ٽ���
        pthread_mutex_lock(&mutex);
        if(!flag[ran]) {
            index[index_num] = ran;
            flag[ran] = 1;
            index_num++;
        }
        pthread_mutex_unlock(&mutex);
    }
    // �����򲿷ֿ�ʼ��ʱ
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
    // ÿ���߳���(thread_id * index_size, (thread_id + 1) * index_size) * segment��Χ������
    cout << "static continuation sort:" << endl;
    for(long long i = 0; i < thread_count; i++) {
        pthread_create(&pthread[i], NULL, static_array_sort, (void*)i);
    }
    for(int i = 0; i < thread_count; i++) {
        pthread_join(pthread[i], NULL);
    }
    cout << endl << endl;
    // ÿ���߳���(k * thread_count + thread_id)(k<=index_size)��Χ������
    copy_matrix();
    cout << "static interval sort:" << endl;
    for(long long i = 0; i < thread_count; i++) {
        pthread_create(&pthread[i], NULL, static_array_sort_interval, (void*)i);
    }
    for(int i = 0; i < thread_count; i++) {
        pthread_join(pthread[i], NULL);
    }
    cout << endl << endl;
    // ÿ���߳��������Ӧ���н�������
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
    // ��̬�����߳�
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
