#include <iostream>
#include <nmmintrin.h>
#include <windows.h>
#include <string.h>

using namespace std;
const int N = 1280; // �����ģ
const int test_n = 10; // ���ǰn * n�ľ���鿴���������Ƿ���ȷ
const int T = N / 5; // ��Ƭ��С
const int loop = 1; /* ѭ������ ===> ��N��Сʱ�������ٶ�̫�죬��ʱ����û��ô׼��
                        ��ʱ��Ҫ����ظ��������������ʱ�䣬�Ӷ����Ӽ�ʱ��׼��*/
float a[N][N], b[N][N], c[N][N];
LARGE_INTEGER Freq, beginTime, endTime;
double runtime;

// �����ʼ��
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

// ����ת��
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

// ������
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

// �����㷨
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
    // ���10 * 10�����
    // test_c();
}

// cache�Ż�
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
    // ���10 * 10�����
    // test_c();
}

// SSE�汾
void SSE()
{
    __m128 t1, t2, sum;
    transpose();
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            // һ��Ҫ�ǵý�sum��Ϊ0!!
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

            // ����ʣ�µ���
            for(int k = (N % 4) - 1; k >= 0; k--)
            {
                c[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    transpose();
    // ���10 * 10�����
    // test_c();
}

// ��Ƭ����
void tile_matrix()
{
    __m128 t1, t2, sum;
    float t;
    transpose();
    // ��Ƭ  Ϊ��׷���ƬЧ����󻯣�N / T���Ϊ����
    for(int r = 0; r < N / T; r++)
    {
        for(int q = 0; q < N / T; q++)
        {
            for(int s = 0; s < N / T; s++)
            { // ������Ϊ��Ƭ��ѭ��
                // ������Ϊ��Ƭ��ѭ��
                for(int i = 0; i < T; i++)
                {
                    for(int j = 0; j < T; j++)
                    {
                        sum = _mm_setzero_ps();
                        for(int k = 0; k < T - (T % 4); k += 4)
                        { // �ĸ��ĸ�Ԫ������
                            t1 = _mm_loadu_ps(a[r * T + i] + s * T + k);
                            t2 = _mm_loadu_ps(b[q * T + j] + s * T + k);
                            t1 = _mm_mul_ps(t1, t2);
                            sum = _mm_add_ps(sum, t1);
                        }
                        sum = _mm_hadd_ps(sum, sum);
                        sum = _mm_hadd_ps(sum, sum);
                        _mm_store_ss(&t, sum);
                        c[r * T + i][q * T + j] += t;
                        // ����kѭ��û������Ĳ���
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
    // �����������Ƿ���ȷ
    // test_c();
}

int main()
{
    initmatrix();
    // �����㷨
    cout << "start serial algorithm!" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(int i = 0; i < loop; i++)
    {
        serial();
    }
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << loop << "�δ����㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;

    // �����Ż�
    cout << "start cache optimization algorithm!" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(int i = 0; i < loop; i++)
    {
        cache_optimize();
    }
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << loop << "�λ����Ż��㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;

    // SSE�Ż�
    cout << "SSE algorithm!" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(int i = 0; i < loop; i++)
    {
        SSE();
    }
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << loop << "��SSE�㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;

    // ��Ƭ�Ż�
    cout << "tile algorithm!" << endl;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(int i = 0; i < loop; i++)
    {
        tile_matrix();
    }
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << loop << "�η�Ƭ�Ż��㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;
    return 0;
}
