#include <iostream>
#include <nmmintrin.h>
#include <windows.h>
#include <string.h>
#include <ctime>
#include <cstdlib>
using namespace std;
const int N = 500; // 矩阵规模
float equation[N][N + 1]; // 原矩阵
float A[N][N];
float copyequation[N][N + 1]; // 拷贝矩阵 ==> 最终转化为上三角矩阵
float resequation[N][N + 1]; // 结果矩阵 ==> 用于回代过程
// 计时模块
LARGE_INTEGER Freq, beginTime, endTime;
double runtime;
// 高斯消元法 串行无优化
void init_matrix(float (*matrix)[N + 1]) // 二维数组形参还可以用float matrix[][N+1]
{
    // 初始化矩阵
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j <= N; j++)
        {
            matrix[i][j] = rand() % (N * (N + 1) - 1) + 1; // 保证a[n][n]不为0
        }
    }
}
void copy_matrix(float (*copymatrix)[N + 1], float (*matrix)[N + 1])
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j <= N; j++)
        {
            copymatrix[i][j] = matrix[i][j];
        }
    }
}
void print_matrix(float (*matrix)[N + 1]) // 二位数组打印
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j <= N; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl << endl;
}
void Gaussian_serial(float (*matrix)[N + 1])
{
    // 要一边求A矩阵一边求结果
    // 每次大循环都会导致矩阵的更新，A如果预先全部算出，那么后面更新矩阵之后A原本的参数就不适用了
    memset(A, 0, sizeof(A));
    for(int k = 0; k < N; k++)
    {
        // 更新A矩阵
        for(int i = k + 1; i < N; i++)
        {
            A[k][i] = matrix[i][k] / matrix[k][k]; // A[k][i]是为了将A改为按行遍历，进行cache优化
        }
        // print_matrix(A);
        // 矩阵化简  matrix[i][k] = 0  i从k+1开始，j从k+1开始
        for(int i = k + 1; i < N; i++)
        {
            for(int j = k + 1; j <= N; j++)
            {
                matrix[i][j] -= A[k][i] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}
// 进行了一定改进的高斯算法，方便进行并行化
void Gaussian_serial_optimize(float (*matrix)[N + 1])
{
    for(int k = 0; k < N; k++)
    {
        // 将每一行的项都除以matrix[k][k],这样就不用特意去求A矩阵了，第matrix[i][k]项就对应A[k][i]
        for(int j = k + 1; j <= N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
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
// SSE高斯算法
void Gaussian_SSE(float (*matrix)[N + 1])
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
            float mikarray[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
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
// 回代算法 串行
void back_substitution_serial(float (*matrix)[N + 1])
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
// 回代算法 SSE
void back_substitution_SSE(float (*matrix)[N + 1])
{
    __m128 t1, t2, t3, sum;
    float t;
    // SSE并行化回代算法
    for(int i = N - 1; i >= 0; i--)
    {
        sum = _mm_setzero_ps();
        int flag = N;
        for(int j = N - 4; j > i ; j -= 4)
        {// 先减去右边的
            float tmp[4] = {matrix[j][N], matrix[j + 1][N], matrix[j + 2][N], matrix[j + 3][N]};
            t1 = _mm_loadu_ps(matrix[i] + j);
            t2 = _mm_loadu_ps(tmp);
            t3 = _mm_mul_ps(t1, t2);
            sum = _mm_add_ps(sum, t3);
            // matrix[i][N] -= matrix[i][j] * matrix[j][N];
            matrix[i][j] = matrix[i][j + 1] = matrix[i][j + 2] = matrix[i][j + 3] = 0;
            flag = j;
        }
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        _mm_store_ss(&t, sum);
        matrix[i][N] -= t;

        // 不足四个的用串行算法补全
        if((N - i - 1) % 4 != 0)
        {
            for(int j = i + 1; j < flag; j++)
            {
                matrix[i][N] -= matrix[i][j] * matrix[j][N];
                matrix[i][j] = 0;
            }
        }
    }
}
int main()
{
    srand((int)time(NULL));
    memset(equation, 0, sizeof(equation));
    // 矩阵初始化
    init_matrix(equation);
    copy_matrix(copyequation, equation);
    // 矩阵规模较小时用于检测运算结果是否正确
    // print_matrix(copyequation);

    // 高斯串行算法一
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    Gaussian_serial(copyequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "高斯第一种串行算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;

    // print_matrix(copyequation);

    // 高斯串行算法二  能改进为并行化版本
    copy_matrix(copyequation, equation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    Gaussian_serial_optimize(copyequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "高斯第二种串行算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    // print_matrix(copyequation);

    // 高斯并行SSE改进版本
    copy_matrix(copyequation, equation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    Gaussian_SSE(copyequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "高斯SSE算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    // print_matrix(copyequation);

    // 回代算法 串行
    copy_matrix(resequation, copyequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    back_substitution_serial(resequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "回代串行算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    // print_matrix(resequation);

    // 回代算法 SSE
    copy_matrix(resequation, copyequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    back_substitution_SSE(resequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "回代SSE算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    // print_matrix(resequation);
    return 0;
}
