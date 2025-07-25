#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void)
{
    printf("threadIdx: (%d, %d, %d), blockIdx: (%d, %d, %d), blockDim: (%d, %d, %d), gridDim: (%d, %d, %d)\n", 
    threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, 
    blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z); 
}

int main(int argc, char **argv)
{
    int nElem = 6; 

    // define grid and block structure 
    dim3 block (3); 
    dim3 grid ((nElem + block.x - 1) / block.x); 

    // check grid and block dimension from host side 
    printf("grid size from CPU: (%d, %d, %d)\n", grid.x, grid.y, grid.z); 
    printf("block size from CPU: (%d, %d, %d)\n", block.x, block.y, block.z); 

    // check grid and block dimension from device side 
    checkIndex <<<grid, block>>> (); 

    cudaDeviceReset();

    return 0;
}