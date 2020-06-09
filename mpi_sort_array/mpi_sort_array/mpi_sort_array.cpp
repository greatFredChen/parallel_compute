#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<Windows.h>
#include<algorithm>
#include<time.h>
LARGE_INTEGER beginTime, endTime, Freq;
const int N = 10000; // 行数
const int arrlen = 10000; // 列数
int Process_Count; // 进程数
float matrix[N][arrlen]; // 原始矩阵
float copy_matrix[N][arrlen]; // 拷贝矩阵
const int dynamic_size = 100; // 粒度
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
		if (rand() & 127 < ratio) // 逆序
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
void mpi_static_sort() // 静态划分
{
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid); // 当前线程id
	MPI_Status status; // 状态变量
	int localN = N / Process_Count; // 当前线程的N
	QueryPerformanceFrequency(&Freq);
	QueryPerformanceCounter(&beginTime);
	if (myid != 0)
	{
		// 非主进程则接收传过来的数据
		MPI_Recv(&copy_matrix[0][0], localN * arrlen, MPI_FLOAT, 0, myid, MPI_COMM_WORLD, &status); // 接收任务分配
		for (int i = 0; i < localN; i++)
			std::stable_sort(copy_matrix[i], copy_matrix[i] + arrlen);
		// 发送排序结果给主进程
		MPI_Send(&copy_matrix[0][0], localN * arrlen, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
		QueryPerformanceCounter(&endTime);
		printf("(static)The time of Process %d is %f ms.\n", myid, (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart);
	}
	else // 主线程分配任务
	{
		// 发送数据给各个进程
		for (int i = 1; i < Process_Count; i++)
			MPI_Send(&matrix[i * localN][0], localN * arrlen, MPI_FLOAT, i, i, MPI_COMM_WORLD);
		// 主线程接收其它线程传过来的数据
		for (int i = 1; i < Process_Count; i++)
		{
			// 先复制到copy_matrix，然后再复制到原来的matrix
			MPI_Recv(&copy_matrix[0][0], localN * arrlen, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
				&status);
			memcpy(&matrix[status.MPI_TAG * localN][0], &copy_matrix[0][0], sizeof(float) * localN * arrlen);
		}
		// 主线程排序
		for (int i = 0; i < localN; i++)
			std::stable_sort(matrix[i], matrix[i] + arrlen);
		QueryPerformanceCounter(&endTime);
		printf("(static)The time of Process %d is %f ms.\n", myid, (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart);
	}
}
void mpi_dynamic_sort()
{
	// 主线程负责任务分发  其余线程负责排序
	MPI_Status status;
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	// 计时
	QueryPerformanceFrequency(&Freq);
	QueryPerformanceCounter(&beginTime);
	// 先把0 - 1 * dynamic_size的部分排序
	for (int i = 0; i < dynamic_size; i++)
		std::stable_sort(matrix[i], matrix[i] + arrlen);

	if (myid != 0)
	{
		while (1)
		{
			// 接收的时候，TAG为行号
			MPI_Recv(&copy_matrix[0][0], dynamic_size * arrlen, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if (status.MPI_TAG >= N) // 结束条件
			{
				QueryPerformanceCounter(&endTime);
				printf("(dynamic)The time of Process %d is %f ms.\n", myid, (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart);
				return;
			}
			// 排序
			for (int i = 0; i < dynamic_size; i++)
				std::stable_sort(copy_matrix[i], copy_matrix[i] + arrlen);
			// 发送回去
			MPI_Send(&copy_matrix[0][0], dynamic_size * arrlen, MPI_FLOAT, 0, status.MPI_TAG, MPI_COMM_WORLD);
		}
		QueryPerformanceCounter(&endTime);
		printf("(dynamic)The time of Process %d is %f ms.\n", myid, (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart);
	}
	else if (Process_Count > 1)// 主线程, 且此时进程数大于1
	{
		int i, finished = 1; // 主进程设置为完成，其它进程未完成
		// 发送阶段
		for (i = 1; i < Process_Count; i++)
			MPI_Send(&matrix[i * dynamic_size][0], dynamic_size * arrlen, MPI_FLOAT, i,
				i * dynamic_size, MPI_COMM_WORLD);
		// 接收阶段
		while (finished < Process_Count) // 是否所有分支线程都已经完成
		{
			MPI_Recv(&copy_matrix[0][0], dynamic_size * arrlen, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG,
				MPI_COMM_WORLD, &status);
			// 复制到matrix中
			memcpy(&matrix[status.MPI_TAG][0], &copy_matrix[0][0], sizeof(float) * dynamic_size * arrlen);
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
		QueryPerformanceCounter(&endTime);
		printf("(dynamic)The time of Process %d is %f ms.\n", myid, (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart);
	}
	else // 只有一个线程
	{
		for (int i = dynamic_size; i < N; i++)
			std::stable_sort(matrix[i], matrix[i] + arrlen);
		QueryPerformanceCounter(&endTime);
		printf("(dynamic)The time of Process %d is %f ms.\n", myid, (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart);
	}
}
void print_matrix(float(*matrix)[arrlen])
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < arrlen; j++)
			printf("%f ", matrix[i][j]);
		printf("\n");
	}
}
int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &Process_Count); // 获取总的进程数

	// 静态划分
	if (myid == 0)
		initialize_matrix(matrix);
	mpi_static_sort();

	// 动态划分
	if (myid == 0)
		initialize_matrix(matrix);
	mpi_dynamic_sort();

	// 终止
	MPI_Finalize();
	return 0;
}
