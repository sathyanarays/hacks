#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    ncclUniqueId Id;

    if(argc > 1){
        ncclGetUniqueId(&Id);    
        FILE *fp;
        fp = fopen("init_token.txt","w");
        fwrite(&Id, 1, sizeof(ncclUniqueId), fp);
        fclose(fp);
        printf("Written Init token to file\n");
    } else {
        FILE *fp;
        fp = fopen("init_token.txt", "r");
        fread(&Id, 1, sizeof(ncclUniqueId), fp);
        fclose(fp);
        printf("Read Init token from file\n");
    }

    cudaSetDevice(0);
    ncclComm_t comm;
    if(argc > 1){
        printf("Joining as rank 0\n");
        ncclCommInitRank(&comm, 2, Id, 0);
    } else {        
        printf("Joining as rank 1\n");
        ncclCommInitRank(&comm, 2, Id, 1);
    }
}