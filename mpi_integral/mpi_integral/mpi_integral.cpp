// 基于MPI的梯形积分法实现
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <Windows.h>
#include<omp.h>
int Process_Count;
LARGE_INTEGER Freq, beginTime, endTime;
double f(double x)
{
	return x * x;
}
void mpi_integral(int a, int b, int n)
{
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	double h = double((b - a)) / n;
	int local_n = n / Process_Count;
	double border[2]; // 包括local_a local_b两种
	double my_sum = 0, total_sum = 0;
	MPI_Status status;
	// 计时
	QueryPerformanceFrequency(&Freq);
	QueryPerformanceCounter(&beginTime);
	// 分支线程只负责运算
	if (myid != 0)
	{
		MPI_Recv(&border[0], 2, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD, &status);
		double local_a = border[0];
		double local_b = border[1];
		local_n = round((local_b - local_a) / h);
		my_sum = (f(local_a) + f(local_b)) / 2.0;
		for (int i = 1; i < local_n; i++)
			my_sum += f(local_a + i * h);
		// 发送数据给主线程
		MPI_Send(&my_sum, 1, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
	}
	else // 主线程负责运算和分配任务
	{
		for (int i = 1; i < Process_Count; i++)
		{
			border[0] = a + i * local_n * h;
			border[1] = (myid == Process_Count - 1) ? b : border[0] + local_n * h;
			MPI_Send(&border[0], 2, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
		}
		for (int i = 1; i < Process_Count; i++)
		{
			MPI_Recv(&my_sum, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			total_sum += my_sum;
		}

		// 主进程运算自己的那一部分
		double local_a = a;
		double local_b = local_a + local_n * h;
		my_sum = (f(local_a) + f(local_b)) / 2.0;
		for (int i = 1; i < local_n; i++)
			my_sum += f(local_a + i * h);
		total_sum += my_sum;
		QueryPerformanceCounter(&endTime);
		printf("(static)The time of the program use: %f ms\n", (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart);
		printf("(static)The result of integral is: %f\n", total_sum * h);
	}
}
void omp_integral(int a, int b, int n)
{
	QueryPerformanceFrequency(&Freq);
	QueryPerformanceCounter(&beginTime);
	double h = double((b - a)) / n;
	double result = (f(a) + f(b)) / 2.0;
	#pragma omp parallel for num_threads(Process_Count)
	for (int i = 1; i < n; i++)
		result += f(a + i * h);
	result *= h;
	QueryPerformanceCounter(&endTime);
	printf("(openmp)The result of integral is: %f\n", result);
	printf("(openmp)The time of the program use: %f ms\n", (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart);
}
double serial_integral(int a, int b, int n)
{
	double h = double((b - a)) / n;
	double result = (f(a) + f(b)) / 2.0;
	for (int i = 1; i < n; i++)
		result += f(a + i * h);
	return result * h;
}
int main(int argc, char *argv[])
{
	int myid; // 进程号
	int a = 0, b = 1, n = 10000;

	// MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid); // 获取进程号
	MPI_Comm_size(MPI_COMM_WORLD, &Process_Count); // 获取进程数量

	// 运行MPI算法
	mpi_integral(a, b, n);

	MPI_Finalize();

	if (myid == 0)
	{
		// 运行串行算法
		QueryPerformanceFrequency(&Freq);
		QueryPerformanceCounter(&beginTime);
		double serial_result = serial_integral(a, b, n);
		QueryPerformanceCounter(&endTime);
		printf("(serial)The result of integral is: %f\n", serial_result);
		printf("(serial)The time of the program use: %f ms\n", (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart);
	}

	if (myid == 0)
	{
		// 运行openmp算法
		omp_integral(a, b, n);
	}

	return 0;
}