#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[])
{
	int rank, size;
	int arr[200];
	int arr2[200];
	int i;
	srand(time(0));
	int n;
	double start_time, end_time;

	printf("1.\n");

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	n = size;

	if (rank == 0)
	{
		for (i = 0; i < 200; ++i)
		{
			arr[i] = rand();
			arr2[i] = arr[i];
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	printf("2.\n");
	MPI_Barrier(MPI_COMM_WORLD);

	start_time = MPI_Wtime();
	for (i = 2; i <= n; i = i * 2)
	{
		// printf("3. %d\t%d\n", i, rank);
		printf("%d\n", (int) round(log(i) / log(2)));
		if (rank < i / 2)
		{
			MPI_Send(arr, 200, MPI_INT, rank + (int) round(log(i) / log(2)), 0, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Recv(arr, 200, MPI_INT, rank - (int) round(log(i) / log(2)), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("4. %d\t%d\n", i, rank);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	end_time = MPI_Wtime();
	printf("4.\n");
	if (rank == 0)
	{
		printf("My Broadcast time: %f\n", end_time - start_time);
	}
	MPI_Barrier(MPI_COMM_WORLD);	
	
	start_time = MPI_Wtime();
	MPI_Bcast(arr2, 200, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
		end_time = MPI_Wtime();
		printf("Standard Broadcast time: %f\n", end_time - start_time);
	}

	MPI_Finalize();
}