#include <iostream>
#include <omp.h>
#include <pthread.h>
#include <math.h>
#include <windows.h>
using namespace std;
double global_res = 0.0;
int thread_count = 10;
int next_n = 50; // 动态划分用
pthread_mutex_t mutex;
LARGE_INTEGER Freq, beginTime, endTime;
double runtime = 0.0;
struct Parameter
{
    double a, b;
    int n, thread_id;
    Parameter(double a, double b, int n, int thread_id):a(a), b(b), n(n), thread_id(thread_id)
    {}
    Parameter() {}
    Parameter& operator=(Parameter& parameter) // 重载=，c=a ==> c.operator=(a)
    {
        this->a = parameter.a;
        this->b = parameter.b;
        this->n = parameter.n;
        this->thread_id = parameter.thread_id;
        return *this;
    }
};
double f(double x)
{
    return x * x;
}
void *pthread_trap(void *arg)
{
    Parameter* p = (Parameter*)arg;
    // cout << p->a << " " << p->b << " " << p->n << " " << p->thread_id << endl;
    double h = (p->b - p->a) / p->n;
    int local_n = p->n / thread_count;
    double local_a = p->a + p->thread_id * local_n * h;
    double local_b = (p->thread_id == thread_count - 1)? p->b: local_a + local_n * h;
    double my_result = (f(local_a) + f(local_b)) / 2.0;
    local_n = round((local_b - local_a) / h); // 有可能出现n不能整除thread_count的情况，此时需要对最后的local_n重算
    for(int i = 1; i < local_n; i++)
    {
        my_result += f(local_a + i * h);
    }
    pthread_mutex_lock(&mutex);
    global_res += my_result;
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}
double omp_trap(double a, double b, int n)
{
    double h = (b - a) / n;
    int thread_id = omp_get_thread_num();  // 线程id
    int tc = omp_get_num_threads(); // 线程数
    int local_n = n / tc;
    double local_a = a + thread_id * local_n * h;
    double local_b = (thread_id == tc - 1)? b: local_a + local_n * h;
    local_n = round((local_b - local_a) / h); // 有可能出现n不能整除tc的情况，此时需要对最后的local_n重算
    double my_result = (f(local_a) + f(local_b)) / 2.0;
    for(int i = 1; i < local_n; i++)
    {
        my_result += f(local_a + i * h);
    }
    return my_result;
}
double serial_trap(double a, double b, int n)
{
    double h = (b - a) / n;
    double my_result = (f(a) + f(b)) / 2.0;
    for(int i = 1; i < n; i++)
    {
        my_result += f(a + i * h);
    }
    return my_result * h;
}
int main()
{
    double a = 0, b = 1;
    int n = 1000;
    double h = (b - a) / n;
    pthread_t thread[thread_count];
    Parameter *parameter[thread_count];
    for(int i = 0; i < thread_count; i++)
    {
        Parameter *p = new Parameter(a, b, n, i);
        parameter[i] = p;
    }
    // pthread编程
    pthread_mutex_init(&mutex, NULL);
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    for(int i = 0; i < thread_count; i++)
    {
        pthread_create(&thread[i], NULL, pthread_trap, (void*)parameter[i]);
    }
    for(int i = 0; i < thread_count; i++)
    {
        pthread_join(thread[i], NULL);
    }
    global_res *= h;
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "pthread算法时间(N=" << n << "): " << runtime << "ms" << endl;
    cout << "pthread version result: " << global_res << endl;
    pthread_mutex_destroy(&mutex);

    // destroy the pointer
    for(int i = 0; i < thread_count; i++)
    {
        delete parameter[i];
    }

    // omp编程
    global_res = 0.0;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    #pragma omp parallel num_threads(thread_count) reduction(+: global_res)
    {
        global_res += omp_trap(a, b, n);
    }
    global_res *= h;
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "omp算法时间(N=" << n << "): " << runtime << "ms" << endl;
    cout << "openmp version result: " << global_res << endl;

    // 串行编程
    global_res  = 0.0;
    QueryPerformanceFrequency(&Freq);
    QueryPerformanceCounter(&beginTime);
    global_res = serial_trap(a, b, n);
    QueryPerformanceCounter(&endTime);
    runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
    cout << "串行算法时间(N=" << n << "): " << runtime << "ms" << endl;
    cout << "serial version result: " << global_res << endl;
    return 0;
}
