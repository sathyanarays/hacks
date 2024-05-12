EXECS=pingpong
MPICC?=mpicc

all: ${EXECS}

pingpong: ping_pong.c
	${MPICC} -o pingpong ping_pong.c

clean:
	rm -f ${EXECS}