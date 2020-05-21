#include <iostream>
#include <nmmintrin.h>
#include <windows.h>
#include <string.h>
#include <ctime>
#include <cstdlib>
using namespace std;
const int N = 500; // �����ģ
float equation[N][N + 1]; // ԭ����
float A[N][N];
float copyequation[N][N + 1]; // �������� ==> ����ת��Ϊ�����Ǿ���
float resequation[N][N + 1]; // ������� ==> ���ڻش�����
// ��ʱģ��
LARGE_INTEGER Freq, beginTime, endTime;
double runtime;
// ��˹��Ԫ�� �������Ż�
void init_matrix(float (*matrix)[N + 1]) // ��ά�����βλ�������float matrix[][N+1]
{
    // ��ʼ������
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j <= N; j++)
        {
            matrix[i][j] = rand() % (N * (N + 1) - 1) + 1; // ��֤a[n][n]��Ϊ0
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
void print_matrix(float (*matrix)[N + 1]) // ��λ�����ӡ
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
    // Ҫһ����A����һ������
    // ÿ�δ�ѭ�����ᵼ�¾���ĸ��£�A���Ԥ��ȫ���������ô������¾���֮��Aԭ���Ĳ����Ͳ�������
    memset(A, 0, sizeof(A));
    for(int k = 0; k < N; k++)
    {
        // ����A����
        for(int i = k + 1; i < N; i++)
        {
            A[k][i] = matrix[i][k] / matrix[k][k]; // A[k][i]��Ϊ�˽�A��Ϊ���б���������cache�Ż�
        }
        // print_matrix(A);
        // ���󻯼�  matrix[i][k] = 0  i��k+1��ʼ��j��k+1��ʼ
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
// ������һ���Ľ��ĸ�˹�㷨��������в��л�
void Gaussian_serial_optimize(float (*matrix)[N + 1])
{
    for(int k = 0; k < N; k++)
    {
        // ��ÿһ�е������matrix[k][k],�����Ͳ�������ȥ��A�����ˣ���matrix[i][k]��Ͷ�ӦA[k][i]
        for(int j = k + 1; j <= N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;

        // ����
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
// SSE��˹�㷨
void Gaussian_SSE(float (*matrix)[N + 1])
{
    // ���л�
    __m128 t1, t2, t3, t4;
    for(int k = 0; k < N; k++)
    {
        float mkk = matrix[k][k];
        float mkkarray[4] = {mkk, mkk, mkk, mkk};
        t1 = _mm_loadu_ps(mkkarray); // t1���ܱ䣬���ܱ����ǵ���
        for(int j = k; j <= N - ((N - k + 1) % 4); j += 4)
        {
            t2= _mm_loadu_ps(matrix[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(matrix[k] + j, t3);
        }
        // ʣ�µ�Ԫ���ô����㷨����
        for(int j = N - ((N - k + 1) % 4) + 1; j <= N; j++)
        {
            matrix[k][j] /= mkk;
        }

        // �������
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
            // �����㷨���ʣ�µ���
            for(int j = N - ((N - k) % 4) + 1; j <= N; j++)
            {
                matrix[i][j] -= mikarray[0] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}
// �ش��㷨 ����
void back_substitution_serial(float (*matrix)[N + 1])
{
    for(int i = N - 1; i >= 0; i--)
    {
        for(int j = N - 1; j > i; j--)
        {// �ȼ�ȥ�ұߵ�
            matrix[i][N] -= matrix[i][j] * matrix[j][N];
            matrix[i][j] = 0;
        }
    }
}
// �ش��㷨 SSE
void back_substitution_SSE(float (*matrix)[N + 1])
{
    __m128 t1, t2, t3, sum;
    float t;
    // SSE���л��ش��㷨
    for(int i = N - 1; i >= 0; i--)
    {
        sum = _mm_setzero_ps();
        int flag = N;
        for(int j = N - 4; j > i ; j -= 4)
        {// �ȼ�ȥ�ұߵ�
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

        // �����ĸ����ô����㷨��ȫ
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
    // �����ʼ��
    init_matrix(equation);
    copy_matrix(copyequation, equation);
    // �����ģ��Сʱ���ڼ���������Ƿ���ȷ
    // print_matrix(copyequation);

    // ��˹�����㷨һ
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    Gaussian_serial(copyequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "��˹��һ�ִ����㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;

    // print_matrix(copyequation);

    // ��˹�����㷨��  �ܸĽ�Ϊ���л��汾
    copy_matrix(copyequation, equation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    Gaussian_serial_optimize(copyequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "��˹�ڶ��ִ����㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;
    // print_matrix(copyequation);

    // ��˹����SSE�Ľ��汾
    copy_matrix(copyequation, equation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    Gaussian_SSE(copyequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "��˹SSE�㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;
    // print_matrix(copyequation);

    // �ش��㷨 ����
    copy_matrix(resequation, copyequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    back_substitution_serial(resequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "�ش������㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;
    // print_matrix(resequation);

    // �ش��㷨 SSE
    copy_matrix(resequation, copyequation);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    back_substitution_SSE(resequation);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "�ش�SSE�㷨(N=" << N << ")������ʱ��Ϊ:" << runtime << "ms" << endl;
    // print_matrix(resequation);
    return 0;
}
