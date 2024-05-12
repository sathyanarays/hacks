EXECS=mpi_send_recv
MPICC?=mpicc

all: ${EXECS}

mpi_send_recv: send_recv.c
	${MPICC} -o mpi_send_recv send_recv.c

clean:
	rm -f ${EXECS}