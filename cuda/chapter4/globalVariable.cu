#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__device__ float devData;

__global__ void checkGlobalVariable()
{
    printf("Device: the value of the global variable is %f\n", devData);
    devData += 2.0f;
}

int main(void)
{
    float value = 3.14f;
    CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float)));
    printf("Host: copied %f to the global variable\n", value);

    // invoke the kernel
    checkGlobalVariable<<<1,1>>>();

    CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
    printf("Host: the value changed by the kernel to %f\n", value);

    CHECK(cudaDeviceReset());
    return 0;
}