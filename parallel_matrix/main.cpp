#include <iostream>
#include <nmmintrin.h>
#include <windows.h>
#include <string.h>

using namespace std;
const int N = 1280; // 矩阵规模
const int test_n = 10; // 输出前n * n的矩阵查看数据运算是否正确
const int T = N / 5; // 分片大小
const int loop = 1; /* 循环次数 ===> 当N很小时，运算速度太快，计时可能没那么准，
                        此时需要多次重复计算过程来增加时间，从而增加计时精准度*/
float a[N][N], b[N][N], c[N][N];
LARGE_INTEGER Freq, beginTime, endTime;
double runtime;

// 矩阵初始化
void initmatrix()
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            a[i][j] = b[i][j] = i * 10 + j;
        }
    }
}

// 矩阵转置
void transpose()
{
    for(int i = 0; i < N; i++)
    {
        for(int j = i + 1; j < N; j++)
        {
            swap(b[i][j], b[j][i]);
        }
    }
}

// 输出检测
void test_c()
{
    for(int i = 0; i < test_n; i++)
    {
        for(int j = 0; j <test_n; j++)
        {
            cout << c[i][j] << "\t";
        }
        cout << endl;
    }
}

// 串行算法
void serial()
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            for(int k = 0; k < N; k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    // 输出10 * 10检测下
    // test_c();
}

// cache优化
void cache_optimize()
{
    transpose();
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            for(int k = 0; k < N; k++)
            {
                c[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    transpose();
    // 输出10 * 10检测下
    // test_c();
}

// SSE版本
void SSE()
{
    __m128 t1, t2, sum;
    transpose();
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            // 一定要记得将sum置为0!!
            sum = _mm_setzero_ps();
            for(int k = N - 4; k >= 0; k -= 4)
            {
                t1 = _mm_loadu_ps(a[i] + k);
                t2 = _mm_loadu_ps(b[j] + k);
                t1 = _mm_mul_ps(t1, t2);
                sum = _mm_add_ps(sum, t1); // A B C D
            }
            sum = _mm_hadd_ps(sum, sum); // A+B C+D A+B C+D
            sum = _mm_hadd_ps(sum, sum); // A+B+C+D A+B+C+D A+B+C+D A+B+C+D ==> c[i][j] += a[i][k] * b[k][j]
            _mm_store_ss(c[i] + j, sum);

            // 计算剩下的项
            for(int k = (N % 4) - 1; k >= 0; k--)
            {
                c[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    transpose();
    // 输出10 * 10检测下
    // test_c();
}

// 分片策略
void tile_matrix()
{
    __m128 t1, t2, sum;
    float t;
    transpose();
    // 分片  为了追求分片效率最大化，N / T最好为整数
    for(int r = 0; r < N / T; r++)
    {
        for(int q = 0; q < N / T; q++)
        {
            for(int s = 0; s < N / T; s++)
            { // 上三层为分片外循环
                // 下三层为分片内循环
                for(int i = 0; i < T; i++)
                {
                    for(int j = 0; j < T; j++)
                    {
                        sum = _mm_setzero_ps();
                        for(int k = 0; k < T - (T % 4); k += 4)
                        { // 四个四个元素运算
                            t1 = _mm_loadu_ps(a[r * T + i] + s * T + k);
                            t2 = _mm_loadu_ps(b[q * T + j] + s * T + k);
                            t1 = _mm_mul_ps(t1, t2);
                            sum = _mm_add_ps(sum, t1);
                        }
                        sum = _mm_hadd_ps(sum, sum);
                        sum = _mm_hadd_ps(sum, sum);
                        _mm_store_ss(&t, sum);
                        c[r * T + i][q * T + j] += t;
                        // 处理k循环没有算完的部分
                        for(int k = T - (T % 4); k < T; k++)
                        {
                            c[r * T + i][q * T + j] += a[r * T + i][s * T + k] * b[q * T + j][s * T + k];
                        }
                    }
                }
            }
        }
    }
    transpose();
    // 输出矩阵看输出是否正确
    // test_c();
}

int main()
{
    initmatrix();
    // 串行算法
    cout << "start serial algorithm!" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(int i = 0; i < loop; i++)
    {
        serial();
    }
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << loop << "次串行算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;

    // 缓存优化
    cout << "start cache optimization algorithm!" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(int i = 0; i < loop; i++)
    {
        cache_optimize();
    }
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << loop << "次缓存优化算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;

    // SSE优化
    cout << "SSE algorithm!" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(int i = 0; i < loop; i++)
    {
        SSE();
    }
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << loop << "次SSE算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;

    // 分片优化
    cout << "tile algorithm!" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(int i = 0; i < loop; i++)
    {
        tile_matrix();
    }
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << loop << "次分片优化算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
    return 0;
}
