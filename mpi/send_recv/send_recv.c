#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        fprintf(stderr, "World Size must be greater than 1 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    int number;
    if (world_rank == 0) {
        number = -1;
        MPI_Send(&number,1,MPI_INT,1,0,MPI_COMM_WORLD);
        printf("Sending from rank %d\n", world_rank);
    } else if (world_rank == 1) {
        MPI_Recv(&number,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        printf("Receiving on rank %d\n", world_rank);
    }

    MPI_Finalize();
}