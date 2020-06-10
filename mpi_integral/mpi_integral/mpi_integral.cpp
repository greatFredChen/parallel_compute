// 基于MPI的梯形积分法实现
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <Windows.h>
#include<omp.h>
#include<string>
int Process_Count;
LARGE_INTEGER Freq, beginTime, endTime;
enum way
{
	serial,
	openmp,
	openmp2,
	mpi
};
void start_time()
{
	QueryPerformanceFrequency(&Freq);
	QueryPerformanceCounter(&beginTime);
}
void end_time(way p)
{
	std::string str;
	switch (p)
	{
	case serial: str = "serial"; break;
	case openmp: str = "openmp"; break;
	case mpi: str = "mpi"; break;
	case openmp2: str = "openmp2"; break;
	default: str = "unknown type"; break;
	}
	QueryPerformanceCounter(&endTime);
	printf("(%s)The time of the program use: %f ms\n", str, (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart);
}
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
	start_time();
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
			border[1] = (i == Process_Count - 1) ? b : border[0] + local_n * h;
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
		end_time(way::mpi);
		printf("(mpi)The result of integral is: %f\n", total_sum * h);
	}
}
void omp_integral(int a, int b, int n)
{
	start_time();
	double h = double((b - a)) / n;
	double result = (f(a) + f(b)) / 2.0;
	#pragma omp parallel for num_threads(Process_Count) reduction(+: result)
	for (int i = 1; i < n; i++)
		result += f(a + i * h);
	result *= h;
	end_time(way::openmp);
	printf("(openmp)The result of integral is: %f\n", result);
}
double omp_integral2(int a, int b, int n)
{
	int thread_id = omp_get_thread_num();
	int thread_count = omp_get_num_threads();
	double h = (double(b - a)) / n;
	int local_n = n / thread_count;
	double local_a = a + thread_id * local_n * h;
	double local_b = (thread_id == Process_Count - 1) ? b : local_a + local_n * h;
	local_n = round((local_b - local_a) / h);
	double myres = (f(local_a) + f(local_b)) / 2.0;
	for (int i = 1; i < local_n; i++)
		myres += f(local_a + i * h);
	myres *= h;
	return myres;
}
void serial_integral(int a, int b, int n)
{
	start_time();
	double h = double((b - a)) / n;
	double result = (f(a) + f(b)) / 2.0;
	for (int i = 1; i < n; i++)
		result += f(a + i * h);
	end_time(way::serial);
	printf("(serial)The result of integral is: %f\n", result * h);
}
int main(int argc, char *argv[])
{
	int myid, provided; // 进程号
	int a = 0, b = 1, n = 30;

	// MPI
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
	if (provided < MPI_THREAD_SERIALIZED)
		MPI_Abort(MPI_COMM_WORLD, 1);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid); // 获取进程号
	MPI_Comm_size(MPI_COMM_WORLD, &Process_Count); // 获取进程数量

	// 运行MPI算法
	mpi_integral(a, b, n);

	if (myid == 0)
		// 运行串行算法
		serial_integral(a, b, n);

	if (myid == 0)
	{
		start_time();
		// openmp 2
		double result = 0.0;
		#pragma omp parallel num_threads(Process_Count) reduction(+: result)
		result += omp_integral2(a, b, n);
		end_time(way::openmp2);
		printf("(openmp2)The result of integral is: %f\n", result);
	}

	if (myid == 0)
		// 运行openmp算法
		omp_integral(a, b, n);

	MPI_Finalize();

	return 0;
}