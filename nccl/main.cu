#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    ncclUniqueId Id;
    ncclGetUniqueId(&Id);    
    printf("%s\n", Id.internal);
}