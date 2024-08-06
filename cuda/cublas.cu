#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <unistd.h>


int main (void){
    cudaError_t cudaStat;    
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    float* v1;
    float* v2;
    float* out;
    float* buffer1;
    float* buffer2;
    float* buffer3;
    
    buffer1 = (float *)malloc(5 * sizeof(float));    
    buffer2 = (float *)malloc(5 * sizeof(float));    
    buffer3 = (float *)malloc(5 * sizeof(float));    
    cudaMalloc ((void**)&v1, 5 * sizeof (*v1));
    cudaMalloc ((void**)&v2, 5 * sizeof (*v1));
    cudaMalloc ((void**)&out, 5 * sizeof (*v1));
    
    buffer1[0] = 2.0;
    buffer1[1] = 1.0;
    buffer1[2] = 1.0;
    buffer1[3] = 1.0;
    buffer1[4] = 1.0;

    cudaMemcpy(v1, buffer1, 5 * sizeof(float), cudaMemcpyHostToDevice);

    buffer2[0] = 2.0;
    buffer2[1] = 2.0;
    buffer2[2] = 3.0;
    buffer2[3] = 0.5;
    buffer2[4] = 1.0;

    cudaMemcpy(v2, buffer2, 5 * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Here\n");
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    cublasSdot(handle,5,v1,1,v2,1,out);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Dot product failed\n");
        return EXIT_FAILURE;
    }

    printf("Here\n");
    cudaMemcpy(buffer3, out, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    
    for(i=0;i<5;i++){
        printf("%f ", buffer3[i]);
    }
    
    sleep(1000);

    cublasDestroy(handle);
}