#include <cuda_runtime.h>
#include <stdio.h>

void initialInt(int *ip, int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny)
{
    int *ic = C; 
    printf("\n Matrix: %d, %d \n", nx, ny); 
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix ++)
        {
            printf("%3d", ic[ix]); 
        }
        ic += nx; 
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x; 
    int iy = threadIdx.y + blockIdx.y * blockDim.y; 
    unsigned int idx = ix + iy * nx; 

    printf("thread_id: (%d, %d) block_id: (%d, %d) coordinate: (%d, %d), global index: %2d, ival: %2d\n",
     threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]); 
}

int main(int argc, char **argv)
{
    int dev = 0; 
    cudaSetDevice(dev); 

    // set matrix dimension 
    int nx = 8; 
    int ny = 6; 
    int nxy = nx * ny; 
    int nBytes = nxy * sizeof(float); 

    // malloc host memory 
    int *h_A; 
    h_A = (int *) malloc(nBytes); 

    // initialize host matrix with integer 
    initialInt(h_A, nxy); 
    printMatrix(h_A, nx, ny); 

    // malloc device memory 
    int *d_A; 
    cudaMalloc((int **)&d_A, nBytes); 

    // transfer data from host side to device side 
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice); 

    // set up execution configuration 
    dim3 block (4, 2); 
    dim3 grid ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y); 

    // invoke kernel 
    printThreadIndex <<<grid, block>>> (d_A, nx, ny); 
    cudaDeviceSynchronize(); 

    // free memory 
    free(h_A); 
    cudaFree(d_A); 

    cudaDeviceReset(); 

    return 0;
}