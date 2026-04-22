#include <stdio.h>

#include "example/common/book.h"
#include "example/common/cpu_bitmap.h"

#pragma region Chapter 3

__global__ void add (int a, int b, int* c)
{
    *c = a + b;
}

void testAdd()
{

    int c;
    int *dev_c;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );
    
    int a = 3;
    int b = 4;
    
    add<<<1,1>>>(a, b, dev_c);
    
    HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost) );
    
    printf("%d + %d = %d\n", a, b, c);
    cudaFree(dev_c);
}


void viewDeviceProps()
{
    int count;
    HANDLE_ERROR( cudaGetDeviceCount(&count) );
    printf("Device count: %d\n", count);
    
    cudaDeviceProp prop;
    for (int i = 0; i < count; i++)
    {
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i) );
        printf("\n=== Device Base Info ===\n");
        printf("Name: %s\n", prop.name);
        printf("Major/minor: %d.%d\n", prop.major, prop.minor);

        int clockRateKHz;
        cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, i);
        printf("Clock rate: %d\n", clockRateKHz);

        int deviceOverlap;
        cudaDeviceGetAttribute(&deviceOverlap, cudaDevAttrGpuOverlap, i);
        printf("Device Overlap: %d\n", deviceOverlap);


        printf("\n=== Device Memory ===\n");
        printf("Total Global Mem: %ld\n", prop.totalGlobalMem);
        printf("Total Const Mem: %ld\n", prop.totalConstMem);
        printf("Mem Pitch: %ld\n", prop.memPitch);
        printf("Texture Alignment: %ld\n", prop.textureAlignment);
        
        
        printf("\n=== Device Multi Processors ===\n");
        printf("mpCount: %d\n", prop.multiProcessorCount);
        printf("shared mem per block: %ld\n", prop.sharedMemPerBlock);
        printf("regs per block: %d\n", prop.regsPerBlock);
        printf("warp size: %d\n", prop.warpSize);
        printf("max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("max threads dims: (%d %d %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("max grid size: (%d %d %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        
        
        printf("\n=== Device Additional Info ===\n");
        printf("Integrated: %d\n", prop.integrated);
        printf("canMapHostMemory: %d\n", prop.canMapHostMemory);

        int kernelTimeout;
        cudaDeviceGetAttribute(&kernelTimeout, cudaDevAttrKernelExecTimeout , i);
        printf("kernelExecTimeoutEnabled: %d\n", kernelTimeout);
        
        printf("\n");
    }
}

void filterDevices()
{
    
    int dev;
    HANDLE_ERROR( cudaGetDevice(&dev) );
    printf("Current device id: %d\n", dev);
    
    int count;
    HANDLE_ERROR( cudaGetDeviceCount(&count) );
    
    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;
    
    HANDLE_ERROR( cudaChooseDevice(&dev, &prop) );
    printf("Filtered device id: %d\n", dev);

    HANDLE_ERROR( cudaSetDevice(dev));
}

#pragma endregion


#pragma region Chapter 4

#define N 20

__global__ void addVectors(int* a, int* b, int* c)
{
    int tid = blockIdx.x;
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

void parallelExample()
{
    int a[N];
    int b[N];
    int c[N];
    int *dev_a;
    int *dev_b;
    int *dev_c;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N*sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N*sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N*sizeof(int) ) );

    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i*i;
    }
    

    HANDLE_ERROR( cudaMemcpy( dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice) );
    
    
    addVectors<<<N,1>>> (dev_a, dev_b, dev_c);
    
    HANDLE_ERROR( cudaMemcpy( c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost) );

    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }


    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

#define DIM 1000

struct cuComplex
{
    float r;
    float i;

    __device__ cuComplex (float a, float b) : r(a), i(b) {}

    __device__ float magnitude2( void )
    { 
        return r*r + i*i;
    }

    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r*a.r-i*a.i, i*a.r+r*a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r+a.r, i+a.i);
    }
};


__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}


__global__ void kernel(unsigned char* ptr)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    int juliaValue = julia(x, y);
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 20;
    ptr[offset*4 + 2] = 20;
    ptr[offset*4 + 3] = 255;
}

void fractalExample()
{
    CPUBitmap bitmap (DIM, DIM);
    unsigned char *dev_bitmap;
    
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    
    dim3 grid(DIM, DIM);
    kernel<<<grid,1>>>(dev_bitmap);
    
    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(),
        dev_bitmap,
        bitmap.image_size(),
        cudaMemcpyDeviceToHost ) );
    
    bitmap.display_and_exit();
    
    cudaFree(dev_bitmap);
}


#pragma endregion



int main (int argc, char* argv[])
{
    fractalExample();

    return 0;
}