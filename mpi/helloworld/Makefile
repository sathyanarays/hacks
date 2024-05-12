EXECS=mpi_hello_world
MPICC?=mpicc

all: ${EXECS}

mpi_hello_world: mpihelloworld.c
	${MPICC} -o mpi_hello_world mpihelloworld.c

clean:
	rm -f ${EXECS}