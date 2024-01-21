#include <stdio.h>

__global__ void helloFromGPU(void)
{
    printf("Hello World from GPU Hell\n");
}

int main(void)
{
    printf("Hello World from CPU\n");

    helloFromGPU <<<1,10>>>();
    cudaDeviceReset();

    printf("Exiting");
    return 0;
}

