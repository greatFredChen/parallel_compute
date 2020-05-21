#include<iostream>
#include<time.h>
#include<pthread.h>
#include<nmmintrin.h>
#include<semaphore.h>
#include <windows.h>
using namespace std;
const int N = 1800;
const int thread_count = 8;
float equation[N][N + 1]; // 原矩阵
float copyequation[N][N + 1]; // 拷贝矩阵
float resequation[N][N + 1]; // 结果矩阵，用于回代
pthread_barrier_t barrier;
// 计时模块
LARGE_INTEGER Freq, beginTime, endTime;
int last_j[N];
int last_n = N / 20;
pthread_mutex_t mutex;
double runtime;
void init_matrix() {
    srand(unsigned(time(NULL)));
    for(int i = 0; i < N; i++) {
        for(int j = 0; j <= N; j++) {
            equation[i][j] = rand() % (N * (N + 1) - 1) + 1; // 保证equation元素大于0
        }
    }
}
void init_last_j()
{
    for(int i = 0; i < N; i++)
    {
        last_j[i] = N - 1;
    }
}
void copy_matrix(float (*matrix)[N + 1], float (*copymatrix)[N + 1]) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j <= N; j++) {
            copymatrix[i][j] = matrix[i][j];
        }
    }
}
void print_matrix(float (*matrix)[N + 1]) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j <= N; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}
void *p_Gaussian(void *arg) {
    long long thread_id = (long long)arg;
    for(int k = 0; k < N; k++) {
        int segment = (N - k) / thread_count;
        int seg_start = k + 1 + thread_id * segment;
        int seg_end = (thread_id == thread_count - 1)? N + 1: seg_start + segment;
        for(int j = seg_start; j < seg_end; j++) {
            copyequation[k][j] = copyequation[k][j] / copyequation[k][k];
        }
        // cout << "thread " << thread_id << " from " << seg_start << " to "
        // << seg_end << endl;
        pthread_barrier_wait(&barrier);
        copyequation[k][k] = 1;
        // Gauss Elimination
        seg_end = (thread_id == thread_count - 1)? N : seg_start + segment;
        for(int i = seg_start; i < seg_end; i++) {
            for(int j = k + 1; j <= N; j++) {
                copyequation[i][j] -= copyequation[i][k] * copyequation[k][j];
            }
            copyequation[i][k] = 0;
        }
        // 为了不影响下一步计算，这里需要barrier
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}
void SSE_Gaussian() {
    // 并行化
    __m128 t1, t2, t3, t4;
    for(int k = 0; k < N; k++) {
        float mkk = copyequation[k][k];
        float mkkarray[4] = {mkk, mkk, mkk, mkk};
        t1 = _mm_loadu_ps(mkkarray); // t1不能变，不能被覆盖掉！
        for(int j = k; j <= N - ((N - k + 1) % 4); j += 4) {
            t2= _mm_loadu_ps(copyequation[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(copyequation[k] + j, t3);
        }
        // 剩下的元素用串行算法处理
        for(int j = N - ((N - k + 1) % 4) + 1; j <= N; j++) {
            copyequation[k][j] /= mkk;
        }

        // 化简矩阵
        for(int i = k + 1; i < N; i++) {
            float mikarray[4] = {copyequation[i][k], copyequation[i][k],
            copyequation[i][k], copyequation[i][k]};
            t1 = _mm_loadu_ps(mikarray);
            for(int j = k + 1; j <= N - ((N - k) % 4); j += 4) {
                t2 = _mm_loadu_ps(copyequation[k] + j);
                t3 = _mm_loadu_ps(copyequation[i] + j);
                t4 = _mm_sub_ps(t3, _mm_mul_ps(t1, t2));
                _mm_storeu_ps(copyequation[i] + j, t4);
            }
            // 串行算法解决剩下的项
            for(int j = N - ((N - k) % 4) + 1; j <= N; j++) {
                copyequation[i][j] -= mikarray[0] * copyequation[k][j];
            }
            copyequation[i][k] = 0;
        }
    }
}
void *p_SSE_Gaussian(void *arg) {
    long long thread_id = (long long)arg;
    for(int k = 0; k < N; k++) {
        int segment = (N - k) / thread_count;
        int seg_start = k + 1 + thread_id * segment;
        int seg_end = (thread_id == thread_count - 1)? N + 1: seg_start + segment;
        for(int j = seg_start; j < seg_end; j++) {
            copyequation[k][j] = copyequation[k][j] / copyequation[k][k];
        }
        // cout << "thread " << thread_id << " from " << seg_start << " to "
        // << seg_end << endl;
        pthread_barrier_wait(&barrier);
        copyequation[k][k] = 1;
        // Gauss Elimination
        seg_end = (thread_id == thread_count - 1)? N : seg_start + segment;
        for(int i = seg_start; i < seg_end; i++) {
            // 针对这里使用SSE并行提速
            __m128 t1, t2, t3, t4;
            float cik = copyequation[i][k];
            float cikarray[4] = {cik, cik, cik, cik}; //
            t1 = _mm_loadu_ps(cikarray);
            for(int j = k + 1; j < N - ((N - k) % 4); j += 4) {
                t2 = _mm_loadu_ps(copyequation[i] + j);
                t3 = _mm_loadu_ps(copyequation[k] + j);
                t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
                // 存储到矩阵中
                _mm_storeu_ps(copyequation[i] + j, t4);
            }
            // 串行算法解决剩下项
            for(int j = N - ((N - k) % 4) + 1; j <= N; j++) {
                copyequation[i][j] -= cik * copyequation[k][j];
            }
            copyequation[i][k] = 0;
        }
        // 为了不影响下一步计算，这里需要barrier
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}
void Gaussian_serial() {
    for(int k = 0; k < N; k++) {
        // 将每一行的项都除以matrix[k][k],这样就不用特意去求A矩阵了，第matrix[i][k]项就对应A[k][i]
        for(int j = k + 1; j <= N; j++) {
            copyequation[k][j] = copyequation[k][j] / copyequation[k][k];
        }
        copyequation[k][k] = 1;

        // 化简
        for(int i = k + 1; i < N; i++) {
            for(int j = k + 1; j <= N; j++) {
                copyequation[i][j] -= copyequation[i][k] * copyequation[k][j];
            }
            copyequation[i][k] = 0;
        }
    }
}
void back_substitution_serial() {
    for(int i = N - 1; i >= 0; i--)
    {
        for(int j = N - 1; j > i; j--)
        {// 先减去右边的
            resequation[i][N] -= resequation[i][j] * resequation[j][N];
            resequation[i][j] = 0;
        }
    }
}
void *pthread_back(void *arg)
{
    long long thread_id = (long long)arg;
    for(int i = N - 1; i >= 0; i--)
    {
        int task = N - 1;
        while(1)
        {
            pthread_mutex_lock(&mutex);
            task = last_j[i];
            last_j[i] -= last_n;
            pthread_mutex_unlock(&mutex);
            if(task <= i) break;
            int End = (task - last_n <= i)? i: task - last_n;
            for(int j = task; j > End; j--)
            {
                resequation[i][N] -= resequation[i][j] * resequation[j][N];
                resequation[i][j] = 0;
            }
        }
    }
    pthread_exit(NULL);
}
void back_substitution_SSE() {
    __m128 t1, t2, t3, sum;
    float t;
    for(int i = N - 1; i >= 0; i--)
    {
        sum = _mm_setzero_ps();
        for(int j = i + 1; j < N - ((N - i - 1) % 4); j += 4)
        { // N - ((N - i - 1) % 4)
            float rjnarray[4] = {resequation[j][N], resequation[j + 1][N], resequation[j + 2][N], resequation[j + 3][N]};
            t2 = _mm_loadu_ps(resequation[i] + j);
            t3 = _mm_loadu_ps(rjnarray);
            t1 = _mm_mul_ps(t2, t3);
            sum = _mm_add_ps(sum, t1);
            _mm_storeu_ps(resequation[i] + j, _mm_setzero_ps()); // 归0
        }
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        _mm_store_ss(&t, sum);
        resequation[i][N] -= t;

        // 剩下的用串行解决
        for(int j = N - ((N - i - 1) % 4); j < N; j++)
        {
            resequation[i][N] -= resequation[i][j] * resequation[j][N];
            resequation[i][j] = 0;
        }
    }
}
int main() {
    for(int t = 0; t < 5; t++)
    {
    init_matrix();
    copy_matrix(equation, copyequation);
    pthread_t thread[thread_count];
    pthread_barrier_init(&barrier, NULL, thread_count);
    // pthread并行 静态
    cout << "开始pthread 高斯算法运行！" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(long long i = 0; i < thread_count; i++) {
        pthread_create(&thread[i], NULL, p_Gaussian, (void*)i);
    }
    for(int i = 0; i < thread_count; i++) {
        pthread_join(thread[i], NULL);
    }
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "pthread 高斯算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    pthread_barrier_destroy(&barrier);
    // print_matrix(copyequation);

    // SSE并行
    copy_matrix(equation, copyequation);
    cout << "开始SSE 高斯算法运行！" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    SSE_Gaussian();
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "SSE 高斯算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    // print_matrix(copyequation);

    // 串行部分
    copy_matrix(equation, copyequation);
    cout << "开始串行 高斯算法运行!" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    Gaussian_serial();
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "串行 高斯算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    // print_matrix(copyequation);

    // pthread + SSE并行
    copy_matrix(equation, copyequation);
    pthread_barrier_init(&barrier, NULL, thread_count);
    cout << "开始多线程 SSE 高斯算法运行！" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(long long i = 0; i < thread_count; i++) {
        pthread_create(&thread[i], NULL, p_SSE_Gaussian, (void*)i);
    }
    for(int i = 0; i < thread_count; i++) {
        pthread_join(thread[i], NULL);
    }
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "多线程 SSE 高斯算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    pthread_barrier_destroy(&barrier);
    // print_matrix(copyequation);

    // 回代 串行
    copy_matrix(copyequation, resequation);
    cout << "开始串行回代算法！" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    back_substitution_serial();
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "串行回代算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    // print_matrix(resequation);

    // 回代 SSE
    copy_matrix(copyequation, resequation);
    cout << "开始SSE回代算法！" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    back_substitution_SSE();
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "SSE回代算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    // print_matrix(resequation);

    // 多线程 回代
    copy_matrix(copyequation, resequation);
    cout << "开始pthread回代算法！" << endl;
    init_last_j();
    pthread_mutex_init(&mutex, NULL);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(long long i = 0; i < thread_count; i++)
    {
        pthread_create(&thread[i], NULL, pthread_back, (void*)i);
    }
    for(int i = 0; i < thread_count; i++)
    {
        pthread_join(thread[i], NULL);
    }
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "pthread回代算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    pthread_mutex_destroy(&mutex);
    // print_matrix(resequation);
    }
    return 0;
}
