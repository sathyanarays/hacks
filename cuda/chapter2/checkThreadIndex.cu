#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call) { alert((call), __FILE__, __LINE__); }
inline void alert(cudaError_t code, const char *file, int line) {
    const cudaError_t error = code;
    if (error != cudaSuccess)
    {
        printf("Error : %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(-10*error);
    }
}

void initialInt(int *ip, int size) {
    for(int i=0; i<size; i++) {
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny) {
    int *ic = C;
    printf("\nMatrix: (%d,%d)\n", nx, ny);
    for (int iy = 0; iy < ny; iy++) {
        for (int ix=0; ix<nx; ix++) {
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix;

    printf("thread_id (%d, %d) block_id (%d, %d), coordinate (%d,%d) global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);    
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);    

    // set matrix dimension
    int nx = 8;
    int ny = 6;
    int nxy = nx*ny;
    int nBytes = nxy * sizeof(float);

    // malloc host memory
    int *h_A;
    h_A = (int *)malloc(nBytes);

    // initialize host matrix with integer
    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    // malloc device memory
    int *d_MatA;
    cudaMalloc((void **)&d_MatA, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);

    // set up execution configuration
    dim3 block(4,2);
    int gridx = (nx + block.x - 1) / block.x;
    int gridy = (ny + block.y - 1) / block.y;
    printf("gridx, gridy: %d, %d\n", gridx, gridy);
    dim3 grid(gridx, gridy);

    // invoke the kernel
    printThreadIndex <<<grid, block>>>(d_MatA, nx, ny);
    cudaDeviceSynchronize();

    // free host and device memory
    cudaFree(d_MatA);
    free(h_A);

    // reset device
    cudaDeviceReset();

    return (0);
}