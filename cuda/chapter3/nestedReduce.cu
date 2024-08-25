#include "common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#define LOG 0

int cpuRecursiveReduce(int *data, int const size)
{
    if (size == 1) return data[0];

    int const stride = size / 2;

    for (int i=0; i<stride; i++)
    {
        data[i] += data[i+stride];
    }

    return cpuRecursiveReduce(data, stride);
}

__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int isize)
{
    unsigned int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    if (isize == 2 && tid == 0)
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int istride = isize >> 1;

    if(istride > 1 && tid < istride)
    {
        idata[tid] += idata[tid + istride];
    }

    __syncthreads();

    if(tid == 0)
    {
        gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);

        cudaDeviceSynchronize();
    }

    __syncthreads();
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0, gpu_sum;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // set up execution configuration
    int nblock  = 2048;
    int nthread = 512;   // initial block size

    if(argc > 1)
    {
        nblock = atoi(argv[1]);   // block size from command line argument
    }

    if(argc > 2)
    {
        nthread = atoi(argv[2]);   // block size from command line argument
    }

    int size = nblock * nthread; // total number of elements to reduceNeighbored

    dim3 block (nthread, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("array %d grid %d block %d\n", size, grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)( rand() & 0xFF );
        h_idata[i] = 1;
    }

    memcpy (tmp, h_idata, bytes);

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    double iStart, iElaps;

    // cpu recursive reduction
    iStart = seconds();
    int cpu_sum = cpuRecursiveReduce (tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce\t\telapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // gpu reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    //reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored\t\telapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // gpu nested reduce kernel
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    gpuRecursiveReduce<<<grid, block>>>(d_idata, d_odata, block.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu nested\t\telapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}