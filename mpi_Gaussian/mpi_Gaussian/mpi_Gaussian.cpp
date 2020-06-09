#include<iostream>
#include<Windows.h>
#include<pthread.h>
#include<nmmintrin.h>
#include<omp.h>
#include<mpi.h>
#include<time.h>
#include<algorithm>
#include<stdlib.h>
#include<memory.h>
#include<string>
using namespace std;
const int N = 2000; // 矩阵规模
const int dynamic_size = 50; // 动态划分粒度
float matrix[N][N + 1]; // 原始矩阵
float copy_matrix[N][N + 1]; // 拷贝矩阵
LARGE_INTEGER Freq, beginTime, endTime; // 用于计时
int process_thread_count; // 进程/线程总数
pthread_mutex_t mutex; // pthread 动态划分必须的锁
pthread_barrier_t barrier; // pthread同步
int next_n[N]; // pthread 动态划分的统计行数数组
// 不同的并行方式 枚举
enum  parallel_way
{
	serial,
	pthread,
	openmp,
	mpi,
	mpi_openmp
};
// 时间开始
void time_start()
{
	QueryPerformanceFrequency(&Freq);
	QueryPerformanceCounter(&beginTime);
}
// 时间结束
void time_end(parallel_way p)
{
	string way_str;
	switch (p)
	{
		case serial: way_str = "串行"; break;
		case pthread: way_str = "pthread"; break;
		case openmp: way_str = "openmp"; break;
		case mpi: way_str = "MPI"; break;
		case mpi_openmp: way_str = "MPI+openmp混合模式"; break;
		default: way_str = "未知类型"; break;

	}
	double runtime;
	QueryPerformanceCounter(&endTime);
	runtime = (double)(endTime.QuadPart - beginTime.QuadPart) * 1000 / (double)Freq.QuadPart;
	cout << way_str << "高斯算法(N=" << N << ")的运行时间为:" << runtime << "ms" << endl;
}
// 初始化next_n
void initNext_n(int *next_n)
{
	for (int i = 0; i < N; i++)
		next_n[i] = i + 1;
}
// 初始化矩阵
void initMatrix(float(*matrix)[N + 1])
{
	srand(unsigned(time(NULL)));
	for (int i = 0; i < N; i++)
		for (int j = 0; j <= N; j++)
		{
			// 保证初始化的时候所有项不为0
			matrix[i][j] = (rand() % 4000) + 1; // 1 - 4000的随机数
		}
}
// 拷贝矩阵
void copyMatrix(float(*copy_matrix)[N + 1], float(*matrix)[N + 1])
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j <= N; j++)
		{
			copy_matrix[i][j] = matrix[i][j];
		}
}
// 打印矩阵
void printMatrix(float(*matrix)[N + 1])
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j <= N; j++)
		{
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
}
// 列并行SSE startline为开始行，为mpi方法做的兼容
void column_SSE(float(*matrix)[N + 1], int i, int k, int startline)
{
	// matrix[i][j] -= matrix[i][k] * matrix[k][j]
	__m128 t1, t2, t3, t4;
	float temp[4];
	temp[0] = temp[1] = temp[2] = temp[3] = matrix[i][k];
	t1 = _mm_loadu_ps(temp);
	for (int j = k + 1; j <= (N - (N - k) % 4); j += 4)
	{
		t2 = _mm_loadu_ps(matrix[startline] + j);
		t3 = _mm_loadu_ps(matrix[i] + j);
		t4 = _mm_sub_ps(t3, _mm_mul_ps(t1, t2));
		_mm_storeu_ps(matrix[i] + j, t4);
	}
	// 剩下的串行解决
	for (int j = (N - (N - k) % 4) + 1; j <= N; j++)
	{
		matrix[i][j] -= temp[0] * matrix[startline][j];
	}
}
void serial_Gaussian(float(*matrix)[N + 1], parallel_way p)
{
	time_start();
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j <= N; j++)
			matrix[k][j] /= matrix[k][k];

		matrix[k][k] = 1;

		// 化简
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j <= N; j++)
			{
				matrix[i][j] -= matrix[i][k] * matrix[k][j];
			}
			matrix[i][k] = 0;
		}
	}
	time_end(p);
}
// MPI + SSE 主线程负责任务分发+任务执行  其它线程只负责任务执行
// TODO: debug  加上k + 1之后matrix全局数组被改动，原因不明，猜测为数组越界.. 
// 找到bug了，k + 1导致tag不对齐，使用memcpy的时候数组越界...针对这里进行改动
void mpi_SSE(float(*matrix)[N + 1])
{
	// 线程id
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	// 临时存储矩阵
	float temp_matrix[dynamic_size * 2][N + 1];
	// 状态
	MPI_Status status;
	if (process_thread_count == 1)
	{
		serial_Gaussian(matrix, parallel_way::mpi);
		return;
	}
	if (myid == 0)
	{
		// 主线程计时
		time_start();
		// 主线程只负责分发任务
		// 一共执行k次，每次都发送k的行号 第k行数据 k+1到N行的矩阵(每次发送dynamic_size行)
		for (int k = 0; k < N; k++)
		{
			// k的行号 第k行数据每个线程都需要
			for (int thread_id = 1; thread_id < process_thread_count; thread_id++)
			{
				MPI_Send(&k, 1, MPI_INT, thread_id, N + 200, MPI_COMM_WORLD); // tag = N + 200表示发送行号k
				MPI_Send(&matrix[k][0], N + 1, MPI_FLOAT, thread_id, N + 100, MPI_COMM_WORLD); // tag = N + 100 发送第k行数据
			}
			// 动态划分发送k + 1 到 N 行的矩阵  以当前行号作为tag，识别对应的行
			// 初始发送阶段
			int i, finished = 1; // 主线程默认完成
			for (i = 1; i < process_thread_count; i++)
			{
				if (k + 1 + (i - 1) * dynamic_size < N)
				{
					MPI_Send(&matrix[k + 1 + (i - 1) * dynamic_size][0], dynamic_size * (N + 1), MPI_FLOAT,
						i, k + 1 + (i - 1) * dynamic_size, MPI_COMM_WORLD);
				}
				else // 若是越界，则直接结束接下来的进程
				{
					MPI_Send(&matrix[0][0], dynamic_size * (N + 1), MPI_FLOAT, i, N + 1, MPI_COMM_WORLD);
					finished++;
				}
			}
			i--; // 之前算了0-process_thread_count - 2对应的行标签，此时的i为process_thread_count，因此要自减1

			// 接收阶段
			while (finished < process_thread_count) // 是否所有的分支线程都已经完成
			{
				MPI_Recv(&temp_matrix[0][0], dynamic_size * (N + 1), MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG,
					MPI_COMM_WORLD, &status);
				// 拷贝到原来矩阵 注意memcpy拷贝是否越界！！！！！
				if (status.MPI_TAG + dynamic_size <= N)
					memcpy(&matrix[status.MPI_TAG][0], &temp_matrix[0][0], sizeof(float) * dynamic_size * (N + 1));
				else
					memcpy(&matrix[status.MPI_TAG][0], &temp_matrix[0][0], sizeof(float) *
					(N - status.MPI_TAG) * (N + 1));
				// 假如复制还没有完成，则继续进行
				if (k + 1 + i * dynamic_size < N)
				{
					MPI_Send(&matrix[k + 1 + i * dynamic_size][0], dynamic_size * (N + 1), MPI_FLOAT, status.MPI_SOURCE, k + 1 + i * dynamic_size, MPI_COMM_WORLD);
					i++;
				}
				else // 复制已经完成 结束分支线程的内层循环
				{
					MPI_Send(&matrix[0][0], dynamic_size * (N + 1), MPI_FLOAT, status.MPI_SOURCE, N + 1, MPI_COMM_WORLD);
					finished++;
				}
			}
		}
		// k次循环结束后，结束分支线程的外层循环
		int k = N + 1;
		for (int i = 1; i < process_thread_count; i++)
			MPI_Send(&k, 1, MPI_INT, i, N + 200, MPI_COMM_WORLD);
		time_end(parallel_way::mpi);
	}
	else
	{
		// 次线程负责运算执行任务
		while (1)
		{
			int k; // 第k行行号
			MPI_Recv(&k, 1, MPI_INT, 0, N + 200, MPI_COMM_WORLD, &status);
			if (k >= N)
				break;
			// 接收第k行
			MPI_Recv(&temp_matrix[0][0], N + 1, MPI_FLOAT, 0, N + 100, MPI_COMM_WORLD, &status);
			// 内层循环接收原矩阵并进行运算
			while (1)
			{
				// 接收第k + 1 到 k + 1 + dynamic_size行
				MPI_Recv(&temp_matrix[1][0], dynamic_size * (N + 1), MPI_FLOAT, 0, MPI_ANY_TAG,
					MPI_COMM_WORLD, &status);
				if (status.MPI_TAG >= N)
					break;
				// 从1 到 dynamic_size
				for (int i = 1; i <= dynamic_size; i++)
				{
					temp_matrix[i][k] /= temp_matrix[0][k];
					column_SSE(temp_matrix, i, k, 0);
					temp_matrix[i][k] = 0;
				}
				// 运算之后返回结果
				MPI_Send(&temp_matrix[1][0], dynamic_size * (N + 1), MPI_FLOAT, 0, status.MPI_TAG, MPI_COMM_WORLD);
			}
		}
	}
}
// pthread 动态划分
void *pthread_SSE(void *arg)
{
	long thread_id = (long)arg;
	int task = 0;
	for (int k = 0; k < N; k++)
	{
		while (1)
		{
			pthread_mutex_lock(&mutex);
			task = next_n[k];
			next_n[k] += dynamic_size;
			pthread_mutex_unlock(&mutex);
			if (task >= N) break;
			int task_end = (task + dynamic_size >= N) ? N : task + dynamic_size;
			for (int i = task; i < task_end; i++)
			{
				copy_matrix[i][k] /= copy_matrix[k][k];
				column_SSE(copy_matrix, i, k, k);
				copy_matrix[i][k] = 0;
			}
		}
		// 进入下一个k循环之前同步
		pthread_barrier_wait(&barrier);
	}
	pthread_exit(NULL);
	return NULL;
}
// openmp 动态划分
void openmp_SSE(float(*matrix)[N + 1])
{
	time_start();
	#pragma omp parallel num_threads(process_thread_count)
	for (int k = 0; k < N; k++)
	{
		#pragma omp for schedule(dynamic, dynamic_size)
		for (int j = k + 1; j <= N; j++)
			matrix[k][j] /= matrix[k][k];
		#pragma omp barrier
		matrix[k][k] = 1;

		#pragma omp for schedule(dynamic, dynamic_size)
		for (int i = k + 1; i < N; i++)
		{
			column_SSE(matrix, i, k, k);
			matrix[i][k] = 0; // matrix[i][k]已经被存储，可以放心归0
		}
		// 防止上一个k未完成时进入下一个k循环造成冲突
		#pragma omp barrier
	}
	time_end(parallel_way::openmp);
}
// MPI + openmp 混合模式
void mpi_openmp_SSE(float(*matrix)[N + 1])
{
	// 线程id
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	// 临时存储矩阵
	float temp_matrix[dynamic_size * 2][N + 1];
	// 状态
	MPI_Status status;
	if (process_thread_count == 1)
	{
		serial_Gaussian(matrix, parallel_way::mpi_openmp);
		return;
	}
	if (myid == 0)
	{
		// 主线程计时
		time_start();
		// 主线程只负责分发任务
		// 一共执行k次，每次都发送k的行号 第k行数据 k+1到N行的矩阵(每次发送dynamic_size行)
		for (int k = 0; k < N; k++)
		{
			// k的行号 第k行数据每个线程都需要
			#pragma omp parallel for num_threads(process_thread_count)
			for (int thread_id = 1; thread_id < process_thread_count; thread_id++)
			{
				MPI_Send(&k, 1, MPI_INT, thread_id, N + 200, MPI_COMM_WORLD); // tag = N + 200表示发送行号k
				MPI_Send(&matrix[k][0], N + 1, MPI_FLOAT, thread_id, N + 100, MPI_COMM_WORLD); // tag = N + 100 发送第k行数据
			}
			// 动态划分发送k + 1 到 N 行的矩阵  以当前行号作为tag，识别对应的行
			// 初始发送阶段
			int i, finished = 1; // 主线程默认完成
			#pragma omp parallel for num_threads(process_thread_count)
			for (i = 1; i < process_thread_count; i++)
			{
				if (k + 1 + (i - 1) * dynamic_size < N)
				{
					MPI_Send(&matrix[k + 1 + (i - 1) * dynamic_size][0], dynamic_size * (N + 1), MPI_FLOAT,
						i, k + 1 + (i - 1) * dynamic_size, MPI_COMM_WORLD);
				}
				else // 若是越界，则直接结束接下来的进程
				{
					MPI_Send(&matrix[0][0], dynamic_size * (N + 1), MPI_FLOAT, i, N + 1, MPI_COMM_WORLD);
					finished++;
				}
			}
			i--; // 之前算了0-process_thread_count - 2对应的行标签，此时的i为process_thread_count，因此要自减1

			// 接收阶段
			while (finished < process_thread_count) // 是否所有的分支线程都已经完成
			{
				MPI_Recv(&temp_matrix[0][0], dynamic_size * (N + 1), MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG,
					MPI_COMM_WORLD, &status);
				// 拷贝到原来矩阵 注意memcpy拷贝是否越界！！！！！
				if (status.MPI_TAG + dynamic_size <= N)
					memcpy(&matrix[status.MPI_TAG][0], &temp_matrix[0][0], sizeof(float) * dynamic_size * (N + 1));
				else
					memcpy(&matrix[status.MPI_TAG][0], &temp_matrix[0][0], sizeof(float) *
					(N - status.MPI_TAG) * (N + 1));
				// 假如复制还没有完成，则继续进行
				if (k + 1 + i * dynamic_size < N)
				{
					MPI_Send(&matrix[k + 1 + i * dynamic_size][0], dynamic_size * (N + 1), MPI_FLOAT, status.MPI_SOURCE, k + 1 + i * dynamic_size, MPI_COMM_WORLD);
					i++;
				}
				else // 复制已经完成 结束分支线程的内层循环
				{
					MPI_Send(&matrix[0][0], dynamic_size * (N + 1), MPI_FLOAT, status.MPI_SOURCE, N + 1, MPI_COMM_WORLD);
					finished++;
				}
			}
		}
		// k次循环结束后，结束分支线程的外层循环
		int k = N + 1;
		#pragma omp parallel for num_threads(process_thread_count)
		for (int i = 1; i < process_thread_count; i++)
			MPI_Send(&k, 1, MPI_INT, i, N + 200, MPI_COMM_WORLD);
		time_end(parallel_way::mpi_openmp);
	}
	else
	{
		// 次线程负责运算执行任务
		while (1)
		{
			int k; // 第k行行号
			MPI_Recv(&k, 1, MPI_INT, 0, N + 200, MPI_COMM_WORLD, &status);
			if (k >= N)
				break;
			// 接收第k行
			MPI_Recv(&temp_matrix[0][0], N + 1, MPI_FLOAT, 0, N + 100, MPI_COMM_WORLD, &status);
			// 内层循环接收原矩阵并进行运算
			while (1)
			{
				// 接收第k + 1 到 k + 1 + dynamic_size行
				MPI_Recv(&temp_matrix[1][0], dynamic_size * (N + 1), MPI_FLOAT, 0, MPI_ANY_TAG,
					MPI_COMM_WORLD, &status);
				if (status.MPI_TAG >= N)
					break;
				// 从1 到 dynamic_size
				for (int i = 1; i <= dynamic_size; i++)
				{
					temp_matrix[i][k] /= temp_matrix[0][k];
					column_SSE(temp_matrix, i, k, 0);
					temp_matrix[i][k] = 0;
				}
				// 运算之后返回结果
				MPI_Send(&temp_matrix[1][0], dynamic_size * (N + 1), MPI_FLOAT, 0, status.MPI_TAG, MPI_COMM_WORLD);
			}
		}
	}
}
int main(int argc, char *argv[])
{
	// 初始化
	int myid, provided;
	// 使用multiple等级的MPI调用
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	// mpi + SSE
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &process_thread_count);
	if (myid == 0)
	{
		initMatrix(matrix);
		copyMatrix(copy_matrix, matrix);
	}
	mpi_SSE(copy_matrix);

	// 串行算法
	if (myid == 0)
	{
		copyMatrix(copy_matrix, matrix);
		serial_Gaussian(copy_matrix, parallel_way::serial);
	}

	// pthread + SSE
	if (myid == 0)
	{
		copyMatrix(copy_matrix, matrix);
		initNext_n(next_n); // 初始化next_n
		time_start();
		// 动态划分，无需传入线程id
		pthread_mutex_init(&mutex, NULL);
		pthread_barrier_init(&barrier, NULL, process_thread_count);
		pthread_t *thread = new pthread_t[process_thread_count];
		for (long i = 0; i < process_thread_count; i++)
			pthread_create(&thread[i], NULL, pthread_SSE, (void*)i);
		for (int i = 0; i < process_thread_count; i++)
			pthread_join(thread[i], NULL);
		pthread_barrier_destroy(&barrier);
		pthread_mutex_destroy(&mutex);
		time_end(parallel_way::pthread);
	}

	// openmp + SSE
	if (myid == 0)
	{
		copyMatrix(copy_matrix, matrix);
		openmp_SSE(copy_matrix);
	}

	// 开启多线程集群模式 混合编程模式版本 
	// MPI + openmp	
	if (myid == 0)
		copyMatrix(copy_matrix, matrix);
	if (provided < MPI_THREAD_MULTIPLE)
		MPI_Abort(MPI_COMM_WORLD, 1);
	mpi_openmp_SSE(copy_matrix);
	MPI_Finalize();
	return 0;
}
