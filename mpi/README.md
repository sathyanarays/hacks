### Create network for MPI

```
docker network create mpinet
```

### Build docker image

```
docker build . -t mpi:1
```

### Create containers

```
docker run --network mpinet --name mpi-master -d mpi:1 sleep 7200
docker run --network mpinet --name mpi-worker1 -d mpi:1 sleep 7200
docker run --network mpinet --name mpi-worker2 -d mpi:1 sleep 7200
docker run --network mpinet --name mpi-worker3 -d mpi:1 sleep 7200
```

### Stop and remove containers

```
docker stop mpi-master mpi-worker1 mpi-worker2 mpi-worker3 && docker rm mpi-master mpi-worker1 mpi-worker2 mpi-worker3
```

### MPI Run command

```
mpirun -n 3 -H mpi-worker1:1,mpi-worker2:1,mpi-worker3:1 ./mpi_hello_world
```

