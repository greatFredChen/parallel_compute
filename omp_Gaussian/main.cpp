#include <iostream>
#include <omp.h>
#include <windows.h>
#include <algorithm>
#include <time.h>
using namespace std;
const int N = 2500;
const int thread_count = 4;
float equation[N][N + 1]; // 原矩阵
float copyequation[N][N + 1]; // 拷贝矩阵
float resequation[N][N + 1]; // 结果矩阵，用于回代
LARGE_INTEGER Freq, beginTime, endTime;
double runtime;
void init_matrix(float (*matrix)[N + 1])
{
    srand(unsigned(time(NULL)));
    for(int i = 0; i < N; i++) {
        for(int j = 0; j <= N; j++) {
            matrix[i][j] = rand() % (N * (N + 1) - 1) + 1; // 保证equation元素大于0
        }
    }
}
void copy_matrix(float (*matrix)[N + 1], float (*copymatrix)[N + 1])
{
    for(int i = 0; i < N; i++) {
        for(int j = 0; j <= N; j++) {
            copymatrix[i][j] = matrix[i][j];
        }
    }
}
void print_matrix(float (*matrix)[N + 1])
{
    for(int i = 0; i < N; i++) {
        for(int j = 0; j <= N; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}
void serial_Gaussian(float (*matrix)[N + 1])
{
    for(int k = 0; k < N; k++)
    {
        // 将每一行的项都除以matrix[k][k],这样就不用特意去求A矩阵了，第matrix[i][k]项就对应A[k][i]
        for(int j = k + 1; j <= N; j++)
        {
            matrix[k][j] /= matrix[k][k];
        }
        matrix[k][k] = 1;

        // 化简
        for(int i = k + 1; i < N; i++)
        {
            for(int j = k + 1; j <= N; j++)
            {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}
void omp_Gaussian_static(float (*matrix)[N + 1])
{
    #pragma omp parallel num_threads(thread_count)
    for(int k = 0; k < N; k++)
    {
        #pragma omp for schedule(static)
        for(int j = k + 1; j <= N; j++)
        {
            matrix[k][j] /= matrix[k][k];
        }
        #pragma omp barrier
        matrix[k][k] = 1;

        #pragma omp for schedule(static)
        for(int i = k + 1; i < N; i++)
        {
            for(int j = k + 1; j <= N; j++)
            {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
        #pragma omp barrier
    }
}
void omp_Gaussian_dynamic(float (*matrix)[N + 1])
{
    #pragma omp parallel num_threads(thread_count)
    for(int k = 0; k < N; k++)
    {
        #pragma omp for schedule(dynamic, N / 20)
        for(int j = k + 1; j <= N; j++)
        {
            matrix[k][j] /= matrix[k][k];
        }
        #pragma omp barrier
        matrix[k][k] = 1;

        #pragma omp for schedule(dynamic, N / 20)
        for(int i = k + 1; i < N; i++)
        {
            for(int j = k + 1; j <= N; j++)
            {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
        #pragma omp barrier
    }
}
void omp_Gaussian_SSE_static(float (*matrix)[N + 1])
{
    #pragma omp parallel num_threads(thread_count)
    for(int k = 0; k < N; k++)
    {
        #pragma omp for schedule(static)
        for(int j = k + 1; j <= N; j++)
        {
            matrix[k][j] /= matrix[k][k];
        }
        #pragma omp barrier
        matrix[k][k] = 1;

        #pragma omp for schedule(static)
        for(int i = k + 1; i < N; i++)
        {
            __m128 t1, t2, t3, t4;
            float cik = matrix[i][k];
            float cikarray[4] = {cik, cik, cik, cik}; //
            t1 = _mm_loadu_ps(cikarray);
            for(int j = k + 1; j < N - ((N - k) % 4); j += 4)
            {
                t2 = _mm_loadu_ps(matrix[i] + j);
                t3 = _mm_loadu_ps(matrix[k] + j);
                t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
                // 存储到矩阵中
                _mm_storeu_ps(matrix[i] + j, t4);
            }
            // 串行算法解决剩下项
            for(int j = N - ((N - k) % 4) + 1; j <= N; j++)
            {
                matrix[i][j] -= cik * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
        #pragma omp barrier
    }
}
void omp_Gaussian_SSE_dynamic(float (*matrix)[N + 1])
{
    #pragma omp parallel num_threads(thread_count)
    for(int k = 0; k < N; k++)
    {
        #pragma omp for schedule(dynamic, N / 20)
        for(int j = k + 1; j <= N; j++)
        {
            matrix[k][j] /= matrix[k][k];
        }
        #pragma omp barrier
        matrix[k][k] = 1;

        #pragma omp for schedule(dynamic, N / 20)
        for(int i = k + 1; i < N; i++)
        {
            __m128 t1, t2, t3, t4;
            float cik = matrix[i][k];
            float cikarray[4] = {cik, cik, cik, cik}; //
            t1 = _mm_loadu_ps(cikarray);
            for(int j = k + 1; j < N - ((N - k) % 4); j += 4)
            {
                t2 = _mm_loadu_ps(matrix[i] + j);
                t3 = _mm_loadu_ps(matrix[k] + j);
                t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
                // 存储到矩阵中
                _mm_storeu_ps(matrix[i] + j, t4);
            }
            // 串行算法解决剩下项
            for(int j = N - ((N - k) % 4) + 1; j <= N; j++)
            {
                matrix[i][j] -= cik * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
        #pragma omp barrier
    }
}
void SSE_Gaussian(float (*matrix)[N + 1])
{
    // 并行化
    __m128 t1, t2, t3, t4;
    for(int k = 0; k < N; k++)
    {
        float mkk = matrix[k][k];
        float mkkarray[4] = {mkk, mkk, mkk, mkk};
        t1 = _mm_loadu_ps(mkkarray); // t1不能变，不能被覆盖掉！
        for(int j = k; j <= N - ((N - k + 1) % 4); j += 4)
        {
            t2= _mm_loadu_ps(matrix[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(matrix[k] + j, t3);
        }
        // 剩下的元素用串行算法处理
        for(int j = N - ((N - k + 1) % 4) + 1; j <= N; j++)
        {
            matrix[k][j] /= mkk;
        }

        // 化简矩阵
        for(int i = k + 1; i < N; i++)
        {
            float mikarray[4] = {matrix[i][k], matrix[i][k],
            matrix[i][k], matrix[i][k]};
            t1 = _mm_loadu_ps(mikarray);
            for(int j = k + 1; j <= N - ((N - k) % 4); j += 4)
            {
                t2 = _mm_loadu_ps(matrix[k] + j);
                t3 = _mm_loadu_ps(matrix[i] + j);
                t4 = _mm_sub_ps(t3, _mm_mul_ps(t1, t2));
                _mm_storeu_ps(matrix[i] + j, t4);
            }
            // 串行算法解决剩下的项
            for(int j = N - ((N - k) % 4) + 1; j <= N; j++)
            {
                matrix[i][j] -= mikarray[0] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}
void back_sub(float (*matrix)[N + 1])
{
    for(int i = N - 1; i >= 0; i--)
    {
        for(int j = N - 1; j > i; j--)
        {// 先减去右边的
            matrix[i][N] -= matrix[i][j] * matrix[j][N];
            matrix[i][j] = 0;
        }
    }
}
void back_sub_SSE(float (*matrix)[N + 1])
{
    __m128 t1, t2, t3, sum;
    float t;
    for(int i = N - 1; i >= 0; i--)
    {
        sum = _mm_setzero_ps();
        for(int j = i + 1; j < N - ((N - i - 1) % 4); j += 4)
        { // N - ((N - i - 1) % 4)
            float rjnarray[4] = {matrix[j][N], matrix[j + 1][N], matrix[j + 2][N], matrix[j + 3][N]};
            t2 = _mm_loadu_ps(matrix[i] + j);
            t3 = _mm_loadu_ps(rjnarray);
            t1 = _mm_mul_ps(t2, t3);
            sum = _mm_add_ps(sum, t1);
            _mm_storeu_ps(matrix[i] + j, _mm_setzero_ps()); // 归0
        }
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        _mm_store_ss(&t, sum);
        matrix[i][N] -= t;

        // 剩下的用串行解决
        for(int j = N - ((N - i - 1) % 4); j < N; j++)
        {
            matrix[i][N] -= matrix[i][j] * matrix[j][N];
            matrix[i][j] = 0;
        }
    }
}
void back_sub_omp(float (*matrix)[N + 1])
{
    #pragma omp parallel num_threads(thread_count)
    for(int i = N - 1; i >= 0; i--)
    {
        #pragma omp for schedule(static)
        for(int j = N - 1; j > i; j--)
        {// 先减去右边的
            matrix[i][N] -= matrix[i][j] * matrix[j][N];
            matrix[i][j] = 0;
        }
    }
}
int main()
{
    init_matrix(equation);
    // 串行高斯
    copy_matrix(equation, copyequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    serial_Gaussian(copyequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "串行 高斯算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;

    // omp 静态 高斯
    copy_matrix(equation, copyequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    omp_Gaussian_static(copyequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "omp静态 高斯算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;


    // omp 动态 高斯
    copy_matrix(equation, copyequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    omp_Gaussian_dynamic(copyequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "omp动态 高斯算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;

    // omp 静态+SSE 高斯
    copy_matrix(equation, copyequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    omp_Gaussian_SSE_static(copyequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "omp静态SSE 高斯算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;

    // omp 动态+SSE 高斯
    copy_matrix(equation, copyequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    omp_Gaussian_SSE_dynamic(copyequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "omp动态SSE 高斯算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;

    // 串行SSE 高斯
    copy_matrix(equation, copyequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    SSE_Gaussian(copyequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "串行SSE 高斯算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;

    // 串行回代
    copy_matrix(copyequation, resequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    back_sub(resequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "串行回代算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    // print_matrix(resequation);
    // SSE回代
    copy_matrix(copyequation, resequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    back_sub_SSE(resequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "SSE回代算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;

    // openmp回代
    copy_matrix(copyequation, resequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    back_sub_omp(resequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "openmp回代算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    // print_matrix(resequation);
    return 0;
}
