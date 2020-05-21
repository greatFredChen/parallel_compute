#include<iostream>
#include<time.h>
#include<pthread.h>
#include<nmmintrin.h>
#include<semaphore.h>
#include <windows.h>
using namespace std;
const int N = 1800;
const int thread_count = 8;
float equation[N][N + 1]; // ԭ����
float copyequation[N][N + 1]; // ��������
float resequation[N][N + 1]; // ����������ڻش�
pthread_barrier_t barrier;
// ��ʱģ��
LARGE_INTEGER Freq, beginTime, endTime;
int last_j[N];
int last_n = N / 20;
pthread_mutex_t mutex;
double runtime;
void init_matrix() {
    srand(unsigned(time(NULL)));
    for(int i = 0; i < N; i++) {
        for(int j = 0; j <= N; j++) {
            equation[i][j] = rand() % (N * (N + 1) - 1) + 1; // ��֤equationԪ�ش���0
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
        // Ϊ�˲�Ӱ����һ�����㣬������Ҫbarrier
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}
void SSE_Gaussian() {
    // ���л�
    __m128 t1, t2, t3, t4;
    for(int k = 0; k < N; k++) {
        float mkk = copyequation[k][k];
        float mkkarray[4] = {mkk, mkk, mkk, mkk};
        t1 = _mm_loadu_ps(mkkarray); // t1���ܱ䣬���ܱ����ǵ���
        for(int j = k; j <= N - ((N - k + 1) % 4); j += 4) {
            t2= _mm_loadu_ps(copyequation[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(copyequation[k] + j, t3);
        }
        // ʣ�µ�Ԫ���ô����㷨����
        for(int j = N - ((N - k + 1) % 4) + 1; j <= N; j++) {
            copyequation[k][j] /= mkk;
        }

        // �������
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
            // �����㷨���ʣ�µ���
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
            // �������ʹ��SSE��������
            __m128 t1, t2, t3, t4;
            float cik = copyequation[i][k];
            float cikarray[4] = {cik, cik, cik, cik}; //
            t1 = _mm_loadu_ps(cikarray);
            for(int j = k + 1; j < N - ((N - k) % 4); j += 4) {
                t2 = _mm_loadu_ps(copyequation[i] + j);
                t3 = _mm_loadu_ps(copyequation[k] + j);
                t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
                // �洢��������
                _mm_storeu_ps(copyequation[i] + j, t4);
            }
            // �����㷨���ʣ����
            for(int j = N - ((N - k) % 4) + 1; j <= N; j++) {
                copyequation[i][j] -= cik * copyequation[k][j];
            }
            copyequation[i][k] = 0;
        }
        // Ϊ�˲�Ӱ����һ�����㣬������Ҫbarrier
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}
void Gaussian_serial() {
    for(int k = 0; k < N; k++) {
        // ��ÿһ�е������matrix[k][k],�����Ͳ�������ȥ��A�����ˣ���matrix[i][k]��Ͷ�ӦA[k][i]
        for(int j = k + 1; j <= N; j++) {
            copyequation[k][j] = copyequation[k][j] / copyequation[k][k];
        }
        copyequation[k][k] = 1;

        // ����
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
        {// �ȼ�ȥ�ұߵ�
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
            _mm_storeu_ps(resequation[i] + j, _mm_setzero_ps()); // ��0
        }
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        _mm_store_ss(&t, sum);
        resequation[i][N] -= t;

        // ʣ�µ��ô��н��
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
    // pthread���� ��̬
    cout << "��ʼpthread ��˹�㷨���У�" << endl;
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
    cout << "pthread ��˹�㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;
    pthread_barrier_destroy(&barrier);
    // print_matrix(copyequation);

    // SSE����
    copy_matrix(equation, copyequation);
    cout << "��ʼSSE ��˹�㷨���У�" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    SSE_Gaussian();
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "SSE ��˹�㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;
    // print_matrix(copyequation);

    // ���в���
    copy_matrix(equation, copyequation);
    cout << "��ʼ���� ��˹�㷨����!" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    Gaussian_serial();
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "���� ��˹�㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;
    // print_matrix(copyequation);

    // pthread + SSE����
    copy_matrix(equation, copyequation);
    pthread_barrier_init(&barrier, NULL, thread_count);
    cout << "��ʼ���߳� SSE ��˹�㷨���У�" << endl;
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
    cout << "���߳� SSE ��˹�㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;
    pthread_barrier_destroy(&barrier);
    // print_matrix(copyequation);

    // �ش� ����
    copy_matrix(copyequation, resequation);
    cout << "��ʼ���лش��㷨��" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    back_substitution_serial();
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "���лش��㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;
    // print_matrix(resequation);

    // �ش� SSE
    copy_matrix(copyequation, resequation);
    cout << "��ʼSSE�ش��㷨��" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    back_substitution_SSE();
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "SSE�ش��㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;
    // print_matrix(resequation);

    // ���߳� �ش�
    copy_matrix(copyequation, resequation);
    cout << "��ʼpthread�ش��㷨��" << endl;
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
    cout << "pthread�ش��㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;
    pthread_mutex_destroy(&mutex);
    // print_matrix(resequation);
    }
    return 0;
}
