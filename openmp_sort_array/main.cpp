#include <iostream>
#include <omp.h>
#include <algorithm>
#include <time.h>
#include <windows.h>
using namespace std;
const int N = 20000;
const int arrlen = 10000;
const int block_kind = 5;
const int thread_count = 50;
const int next_n = 100;
int min_chunk = 100;
float matrix[N][arrlen];
float sortmatrix[N][arrlen];
LARGE_INTEGER Freq, beginTime, endTime;
double runtime = 0.0;
void init_matrix()
{
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
void copy_matrix()
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < arrlen; j++)
        {
            sortmatrix[i][j] = matrix[i][j];
        }
    }
}
int main()
{
    init_matrix();
    // serial 串行
    copy_matrix();
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(int i = 0; i < N; i++)
    {
        stable_sort(sortmatrix[i], sortmatrix[i] + arrlen);
    }
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "串行算法时间(N=" << N << "): " << runtime << "ms" << endl;

    // omp static
    copy_matrix();
    cout << "static:" << endl;
    for(int i = 0; i < block_kind; i++)
    {
        int block_size = (i + 1) * 50;
        cout << "block_size:" << block_size << endl;
        double wtime = omp_get_wtime();
        #pragma omp parallel for num_threads(thread_count) schedule(static, block_size)
        for(int j = 0; j < N; j++)
        {
            stable_sort(sortmatrix[j], sortmatrix[j] + arrlen);
        }
        wtime = omp_get_wtime() - wtime;
        cout << wtime * 1000 << "ms" << endl;
        cout << endl << endl;
    }

    // omp dynamic
    copy_matrix();
    cout << "dynamic:" << endl;
    double wtime = omp_get_wtime();
    #pragma omp parallel for num_threads(thread_count) schedule(dynamic, next_n)
    for(int i = 0; i < N; i++)
    {
        stable_sort(sortmatrix[i], sortmatrix[i] + arrlen);
    }
    wtime = omp_get_wtime() - wtime;
    cout << wtime * 1000 << "ms" << endl;

    // omp guided
    copy_matrix();
    cout << "guided:" << endl;
    wtime = omp_get_wtime();
    #pragma omp parallel for num_threads(thread_count) schedule(guided, min_chunk)
    for(int i = 0; i < N; i++)
    {
        stable_sort(sortmatrix[i], sortmatrix[i] + arrlen);
    }
    wtime = omp_get_wtime() - wtime;
    cout << wtime * 1000 << "ms" << endl;
    return 0;
}
