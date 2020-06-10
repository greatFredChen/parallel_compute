#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<Windows.h>
#include<algorithm>
#include<time.h>
#include<string>
#include<vector>
#include<math.h>
using namespace std;
LARGE_INTEGER beginTime, endTime, Freq;
const int N = 10000; // 行数
const int arrlen = 10000; // 列数
int Process_Count; // 进程数
float matrix[N][arrlen]; // 原始矩阵
float copy_matrix[N][arrlen]; // 排序矩阵
float temp_matrix[N][arrlen]; // 临时存储矩阵
const int dynamic_size = 10; // 粒度
vector<double> static_time, dynamic_time; // 静态/动态时间数组
// 划分方式
enum way
{
	static_partition,
	dynamic_partition
};
// 开始时间
void start_time()
{
	QueryPerformanceFrequency(&Freq);
	QueryPerformanceCounter(&beginTime);
}
// 结束时间
double end_time(way p, int myid)
{
	QueryPerformanceCounter(&endTime);
	double runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
	string str;
	switch (p)
	{
	case static_partition: str = "static"; break;
	case dynamic_partition: str = "dynamic"; break;
	default: str = "unknown type"; break;
	}
	// printf("(%s)The runtime of process %d is %f ms\n", str, myid, runtime);
	return runtime;
}
// 计算时间方差
double time_standard_deviation(const vector<double>& v)
{
	double sum = 0.0;
	for (int i = 0; i < v.size(); i++)
		sum += v[i];
	double average = sum / v.size();
	double variance = 0.0;
	for (int i = 0; i < v.size(); i++)
		variance += (v[i] - average) * (v[i] - average);
	variance /= v.size();
	return sqrt(variance);
}
// 把时间数组的数据传递给vector
void time_array_to_vector(vector<double> &v, double* time_array, int n)
{
	for (int i = 0; i < n; i++)
		v.push_back(time_array[i]);
}
// 矩阵初始化
void initialize_matrix(float (*matrix)[arrlen])
{
	srand(unsigned(time(NULL)));
	int seg = N / 4, ratio;
	// 四种类型 完全升序 1/4逆序+3/4升序 1/2逆序+1/2升序 全逆序
	for (int i = 0; i < N; i++)
	{
		if (i < seg)
		{
			ratio = 0;
		}
		else if (i < seg * 2)
		{
			ratio = 32;
		}
		else if (i < seg * 3)
		{
			ratio = 64;
		}
		else
		{
			ratio = 128;
		}
		// 初始化每一行
		if ((rand() & 127) < ratio) // 逆序
		{
			for (int j = 0; j < arrlen; j++)
			{
				matrix[i][j] = arrlen - j;
			}
		}
		else // 升序
		{
			for (int j = 0; j < arrlen; j++)
			{
				matrix[i][j] = j;
			}
		}
	}
}
// 打印矩阵
void print_matrix(float(*matrix)[arrlen], int line_start = 0, int line_end = N, 
	int column_start = 0, int column_end = arrlen)
{
	int line_s = (line_start < 0) ? 0 : line_start;
	int column_s = (column_start < 0) ? 0 : column_start;
	int line_e = (line_end > N) ? N : line_end;
	int column_e = (column_end > arrlen) ? arrlen : column_end;
	for (int i = line_s; i < line_e; i++)
	{
		for (int j = column_s; j < column_e; j++)
			printf("%f ", matrix[i][j]);
		printf("\n");
	}
}
// 拷贝矩阵
void duplicate_matrix(float(*copy_matrix)[arrlen], float(*matrix)[arrlen])
{
	for(int i = 0; i < N; i++)
		for (int j = 0; j < arrlen; j++)
		{
			copy_matrix[i][j] = matrix[i][j];
		}
}
// 指定起始行和终点行的串行排序 [s, e)
void serial_sort(float(*matrix)[arrlen], int s, int e)
{
	for (int i = s; i < e; i++)
		stable_sort(matrix[i], matrix[i] + arrlen);
}
void mpi_static_sort(float(*matrix)[arrlen]) // 静态划分
{
	start_time();
	// 线程id
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	// 获取所有线程的运行时间 动态数组
	double *runtime_array = new double[Process_Count];

	int localN = N / Process_Count; // 当前线程的N
	// Scatter分发数据
	MPI_Scatter(matrix[0], localN * arrlen, MPI_FLOAT, temp_matrix[0], localN * arrlen, MPI_FLOAT,
		0, MPI_COMM_WORLD);
	// 排序
	serial_sort(temp_matrix, 0, localN);
	// Gather收集数据
	MPI_Gather(temp_matrix[0], localN * arrlen, MPI_FLOAT, matrix[0], localN * arrlen, MPI_FLOAT,
		0, MPI_COMM_WORLD);
	// 未完成的串行完成 挑一个线程就行..
	if (N % Process_Count && myid == 0)
		serial_sort(matrix, N - (N % Process_Count), N);
	double runtime = end_time(way::static_partition, myid);
	MPI_Gather(&runtime, 1, MPI_DOUBLE, runtime_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (myid == 0)
		time_array_to_vector(static_time, runtime_array, Process_Count);
}
void mpi_dynamic_sort(float(*matrix)[arrlen])
{
	// 主线程负责任务分发  其余线程负责排序
	MPI_Status status;
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	double *time_array = new double[Process_Count];
	double runtime;
	// 计时
	start_time();

	// 只有一个进程 直接串行排序
	if (Process_Count == 1)
	{
		serial_sort(matrix, 0, N);
		time_array[0] = end_time(way::dynamic_partition, myid);
		time_array_to_vector(dynamic_time, time_array, 1);
		return;
	}

	if (myid != 0)
	{
		while (1)
		{
			// 接收的时候，TAG为行号
			MPI_Recv(&temp_matrix[0][0], dynamic_size * arrlen, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if (status.MPI_TAG >= N)
			{
				runtime = end_time(way::dynamic_partition, myid);
				break;
			}
			serial_sort(temp_matrix, 0, dynamic_size);
			MPI_Send(&temp_matrix[0][0], dynamic_size * arrlen, MPI_FLOAT, 0, status.MPI_TAG, MPI_COMM_WORLD);
		}
	}
	else// 主线程, 且此时进程数大于1
	{
		int i, finished = 1; // 主进程设置为完成，其它进程未完成
		// 发送阶段
		for (i = 1; i < Process_Count; i++)
		{
			if ((i - 1) * dynamic_size < N)
				MPI_Send(&matrix[(i - 1) * dynamic_size][0], dynamic_size * arrlen, MPI_FLOAT, i,
				(i - 1) * dynamic_size, MPI_COMM_WORLD);
			else // 越界则结束
			{
				MPI_Send(&matrix[0][0], dynamic_size * arrlen, MPI_FLOAT, i,
					N + 1, MPI_COMM_WORLD);
				finished++;
			}
		}
		i--; // 前面发送了0 - process_count - 2,那么后面从process_count - 1开始发送
		// 接收阶段
		while (finished < Process_Count) // 是否所有分支线程都已经完成
		{
			MPI_Recv(&temp_matrix[0][0], dynamic_size * arrlen, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG,
				MPI_COMM_WORLD, &status);
			// 复制到matrix中
			if (status.MPI_TAG + dynamic_size <= N)
				memcpy(&matrix[status.MPI_TAG][0], &temp_matrix[0][0], sizeof(float) * dynamic_size * arrlen);
			else
				memcpy(&matrix[status.MPI_TAG][0], &temp_matrix[0][0], sizeof(float) * (N - status.MPI_TAG) * arrlen);
			// 假如还没有完成 把未完成的部分继续重新发送和接收
			if (i * dynamic_size < N)
			{
				MPI_Send(&matrix[i * dynamic_size][0], dynamic_size * arrlen, MPI_FLOAT, status.MPI_SOURCE, i * dynamic_size, MPI_COMM_WORLD);
				i++;
			}
			else
			{
				// 向其它进程发送结束信号
				MPI_Send(&matrix[0][0], dynamic_size * arrlen, MPI_FLOAT, status.MPI_SOURCE, N + 1, MPI_COMM_WORLD);
				finished++;
			}
		}
		runtime = end_time(way::dynamic_partition, myid);
	}
	MPI_Gather(&runtime, 1, MPI_DOUBLE, time_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (myid == 0)
		time_array_to_vector(dynamic_time, time_array, Process_Count);
}
int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &Process_Count); // 获取总的进程数

	// 静态划分
	if (myid == 0)
	{
		initialize_matrix(matrix);
		duplicate_matrix(copy_matrix, matrix);
	}
	mpi_static_sort(copy_matrix);
	// if (myid == 0)
		// print_matrix(copy_matrix, N - 10, N, arrlen - 10, arrlen);

	// 动态划分
	if (myid == 0)
	{
		duplicate_matrix(copy_matrix, matrix);
	}
	mpi_dynamic_sort(copy_matrix);
	// if (myid == 0)
		// print_matrix(copy_matrix, N - 10, N, arrlen - 10, arrlen);

	MPI_Barrier(MPI_COMM_WORLD);
	if (myid == 0)
	{
		printf("(static) time standard deviation is %f\n", time_standard_deviation(static_time));
		printf("(dynamic) time standard deviation is %f\n", time_standard_deviation(dynamic_time));
	}
	// 终止
	MPI_Finalize();
	return 0;
}
