#include <cuda_runtime.h>
#include <stdio.h>
// #include <sys/time.h> 
#include <chrono>

double cpuSecond()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
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

void sumArrayOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx=0; idx < N; idx ++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnGPU(float* A, float* B, float*C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    if (i < N) 
    {
        C[i] = A[i] + B[i]; 
        C[i+N] = A[i+N] + B[i+N]; 
    }
}

int main(int argc, char** argv)
{
    // set up device 
    int dev = 0; 
    cudaSetDevice(dev); 

    // set up data size of vectors 
    int nElem = 1<<24; 
    printf("Vector size: %d \n", nElem); 

    // malloc host memory 
    size_t nBytes = nElem * sizeof(float); 

    float *h_A, *h_B, *hostRef, *gpuRef; 
    h_A = (float*) malloc(nBytes); 
    h_B = (float*) malloc(nBytes); 
    hostRef = (float*) malloc(nBytes); 
    gpuRef = (float*) malloc(nBytes); 

    // initial data at host side 
    initialData(h_A, nElem); 
    initialData(h_B, nElem); 

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes); 

    // malloc device global memory 
    float *d_A, *d_B, *d_C; 
    cudaMalloc((float**)&d_A, nBytes); 
    cudaMalloc((float**)&d_B, nBytes); 
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host side to device 
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice); 

    // invoke kernel at host side 
    int iLen = 256; 
    dim3 block (iLen); 
    dim3 grid ((nElem + block.x - 1) / (block.x * 2));  

    double iStart, iElaps; 
    iStart = cpuSecond(); 
    sumArraysOnGPU <<<grid, block>>> (d_A, d_B, d_C, nElem / 2); 
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond() - iStart; 
    printf("SumArraysOnGPU<<<%d,%d>>>, time elapsed %f sec\n", grid.x, block.x, iElaps); 

    // copy kernel result back to host side 
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost); 

    // add vectors at host side for result check 
    iStart = cpuSecond(); 
    sumArrayOnHost(h_A, h_B, hostRef, nElem); 
    iElaps = cpuSecond() - iStart; 
    printf("SumArraysOnCPU, time elapsed %f sec\n", iElaps); 

    // check results 
    checkResult(hostRef, gpuRef, nElem); 

    // free global device memory 
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C); 

    // free host memory 
    free(h_A); 
    free(h_B); 
    free(hostRef); 
    free(gpuRef); 

    return 0;  
}