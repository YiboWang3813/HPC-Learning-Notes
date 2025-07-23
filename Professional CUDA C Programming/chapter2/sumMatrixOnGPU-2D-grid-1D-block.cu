#include<cuda_runtime.h>
#include<stdio.h>
#include <chrono>

double cpuSecond()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

void initialData(float *ip, int size)
{
    // generate different seed for random number 
    time_t t; 
    srand((unsigned int) time(&t)); 

    for (int i=0; i < size; i++)
    {
        ip[i] = (float) (rand() & 0xFF) / 10.0f; 
    }
}

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0e-8; 
    bool match = 1; 
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i] > epsilon))
        {
            match = 0; 
            printf("Arrays do not match!\n");
            printf("host: %5.2f, gpu: %5.2f, at current %d\n", hostRef[i], gpuRef[i], i); 
            break;
        }
    }
    if (match) 
    {
        printf("Arrays match!\n");
    }
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A; 
    float *ib = B; 
    float *ic = C; 

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix ++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; 
        ib += nx; 
        ic += nx;
    }
}

__global__ void sumMatrixOnGPU(float *A, float *B, float *C, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x; 
    int iy = blockIdx.y; 
    unsigned int idx = ix + iy * nx; 

    if (ix < nx && iy < ny)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv)
{
    // set up device 
    int dev = 0; 
    cudaSetDevice(dev); 

    // set up size of matrix 
    int nx = 1<<14; 
    int ny = 1<<14;
    int nxy = nx * ny; 
    int nBytes = nxy * sizeof(float); 
    printf("Matrix size: (%d, %d)\n", nx, ny);

    // malloc host memory 
    float *h_A, *h_B, *hostRef, *gpuRef; 
    h_A = (float *) malloc(nBytes); 
    h_B = (float *) malloc(nBytes); 
    hostRef = (float *) malloc(nBytes); 
    gpuRef = (float *) malloc(nBytes); 

    // initialize data at host side 
    initialData(h_A, nxy); 
    initialData(h_B, nxy); 

    memset(hostRef, 0, nBytes); 
    memset(gpuRef, 0, nBytes); 

    // add matrix on host side 
    double iStart, iElaps; 
    iStart = cpuSecond(); 
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny); 
    iElaps = cpuSecond() - iStart; 
    printf("Sum matrix on CPU, time cost %f sec\n", iElaps); 

    // malloc device global memory 
    float *d_A, *d_B, *d_C; 
    cudaMalloc((float **)&d_A, nBytes); 
    cudaMalloc((float **)&d_B, nBytes); 
    cudaMalloc((float **)&d_C, nBytes); 

    // transfer data from host side to device side 
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice); 

    // set execution config 
    int dimx = 32; 
    // int dimy = 32; 
    dim3 block (dimx, 1); 
    dim3 grid ((nx + block.x - 1) / block.x, ny); 

    // invoke the kernel 
    iStart = cpuSecond(); 
    sumMatrixOnGPU <<<grid, block>>> (d_A, d_B, d_C, nx, ny); 
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond() - iStart; 
    printf("Sum matrix on GPU, time cost %f sec\n", iElaps); 

    // transfer results back to CPU 
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost); 

    // compare 
    checkResult(hostRef, gpuRef, nxy); 

    // free global memory 
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C); 

    // free host memory 
    free(h_A); 
    free(h_B); 
    free(gpuRef); 
    free(hostRef);

    cudaDeviceReset(); 

    return 0; 
}