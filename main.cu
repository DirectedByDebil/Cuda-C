#include <stdio.h>

#include "example/common/book.h"
#include "example/common/cpu_bitmap.h"
#include "example/common/cpu_anim.h"

#define N 20
#define DIM 1000

#define DIM 1024
#define PI 3.1415926535897932f

#define INF 2e10f
#define rnd( x ) (x * rand() / RAND_MAX)
#define SPHERES 2000


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

#pragma region Chapter 5

//========== test blocks and threads ================

__global__ void addVectors_threads(int* a, int* b, int* c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

void parallelExample_threads()
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
    
    
    addVectors_threads<<<128,128>>> (dev_a, dev_b, dev_c);
    
    HANDLE_ERROR( cudaMemcpy( c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost) );

    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }


    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

//========== test animation ================

struct DataBlock
{
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
};

void cleanup (DataBlock *d)
{
    cudaFree(d->dev_bitmap);
}

__global__ void animationKernel(unsigned char *ptr, int ticks)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	// now calculate the value at that position
	float fx = x - DIM/2;
	float fy = y - DIM/2;
	float d = sqrtf( fx * fx + fy * fy );
	
	unsigned char grey = (unsigned char) (128.0f + 127.0f * cos(d/10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));
	
	ptr[offset*4 + 0] = grey;
	ptr[offset*4 + 1] = grey;
	ptr[offset*4 + 2] = grey;
	ptr[offset*4 + 3] = 255;
}

void generate_frame(DataBlock *d, int ticks)
{
    dim3 blocks(DIM/16, DIM/16);
    dim3 threads(16, 16);
    
    animationKernel<<<blocks,threads>>> (d->dev_bitmap, ticks);

    HANDLE_ERROR( cudaMemcpy( d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));
}

void testAnimation()
{
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_bitmap, bitmap.image_size()));

    bitmap.anim_and_exit( (void (*)(void*,int))generate_frame, (void (*)(void*))cleanup );
}

//========== test dot product ================

#define imin(a,b) (a<b?a:b)

const int M = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (M+threadsPerBlock-1)/threadsPerBlock);

__global__ void dot (float *a, float *b, float *c)
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (tid < M)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

void testDotProduct()
{
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = (float *)malloc(M*sizeof(float));
    b = (float *)malloc(M*sizeof(float));
    partial_c = (float *)malloc(blocksPerGrid*sizeof(float));

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, M*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, M*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float)));
	
	for(int i=0; i<M; i++) {
		a[i] = i;
		b[i] = i*2;
	}
	
	
	HANDLE_ERROR(cudaMemcpy(dev_a, a, M*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, M*sizeof(float), cudaMemcpyHostToDevice));
	
	dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
	
	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));
	
	c = 0;
	for(int i=0; i<blocksPerGrid; i++) {
		c += partial_c[i];
	}
	
	#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
	printf("Does GPU value %.6g = %.6g?\n", c, 2*sum_squares((float)(M-1)));
	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
	
	free(a);
	free(b);
	free(partial_c);
}

//========== test shared raster ================

__global__ void bitmapKernel( unsigned char *ptr ) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	__shared__ float shared[16][16];
	

	const float period = 128.0f;
	
	shared[threadIdx.x][threadIdx.y] = 
			255 * (sinf(x*2.0f*PI/ period) + 1.0f) *
                (sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;
	
	__syncthreads();
	
	ptr[offset*4 + 0] = 0;
	ptr[offset*4 + 1] = shared[15-threadIdx.x][15-threadIdx.y];
	ptr[offset*4 + 2] = 0;
	ptr[offset*4 + 3] = 255;
}


void testSharedRaster()
{
    CPUBitmap bitmap( DIM, DIM );
	unsigned char *dev_bitmap;

	HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );

	dim3 grids( DIM/16, DIM/16 );
	dim3 threads(16,16);
	
	bitmapKernel<<<grids,threads>>>( dev_bitmap );

	HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), 
                dev_bitmap, 
                bitmap.image_size(), 
                cudaMemcpyDeviceToHost ) );
	bitmap.display_and_exit();

	HANDLE_ERROR( cudaFree( dev_bitmap ) );
}

#pragma endregion

#pragma region Chapter 6

struct Sphere
{
	float r,b,g;
	float radius;
	float x,y,z;
	__device__ float hit(float ox, float oy, float *n)
    {
		float dx = ox - x;
		float dy = oy - y;
		if(dx*dx + dy*dy < radius*radius) {
			float dz = sqrtf( radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf( radius*radius );
			return dz + z;
		}
		return -INF;
	}
};

__constant__ Sphere s[SPHERES];

__global__ void spheresKernel_constant(unsigned char* ptr )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM/2);
	float oy = (y - DIM/2);
	
	float r=0, g=0, b=0;
	float maxz = -INF;
	for(int i=0; i<SPHERES; i++)
    {
		float n;
		float t = s[i].hit(ox, oy, &n);
		if(t > maxz)
        {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}
	}
	
	ptr[offset*4 + 0] = (int)(r * 255);
	ptr[offset*4 + 1] = (int)(g * 255);
	ptr[offset*4 + 2] = (int)(b * 255);
	ptr[offset*4 + 3] = 255;
}
__global__ void spheresKernel(unsigned char* ptr, Sphere *spheres)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM/2);
	float oy = (y - DIM/2);
	
	float r=0, g=0, b=0;
	float maxz = -INF;
	for(int i=0; i<SPHERES; i++)
    {
		float n;
		float t = spheres[i].hit(ox, oy, &n);
		if(t > maxz)
        {
			float fscale = n;
			r = spheres[i].r * fscale;
			g = spheres[i].g * fscale;
			b = spheres[i].b * fscale;
			maxz = t;
		}
	}
	
	ptr[offset*4 + 0] = (int)(r * 255);
	ptr[offset*4 + 1] = (int)(g * 255);
	ptr[offset*4 + 2] = (int)(b * 255);
	ptr[offset*4 + 3] = 255;
}

void testSpheres_constant()
{
	cudaEvent_t start, stop;
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
	
	CPUBitmap bitmap( DIM, DIM );
	unsigned char *dev_bitmap;
	
	HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
	
	Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
	
	for(int i=0; i<SPHERES; i++) {
		temp_s[i].r = rnd( 1.0f );
		temp_s[i].g = rnd( 1.0f );
		temp_s[i].b = rnd( 1.0f );
		temp_s[i].x = rnd( 1000.0f ) - 500;
		temp_s[i].y = rnd( 1000.0f ) - 500;
		temp_s[i].z = rnd( 1000.0f ) - 500;
		temp_s[i].radius = rnd( 100.0f ) + 20;
	}
	
	HANDLE_ERROR( cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES ) );
	free( temp_s );
	
	dim3 grids(DIM/16, DIM/16);
	dim3 threads(16, 16);
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
	spheresKernel_constant<<<grids,threads>>>(dev_bitmap);
	
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	
	float elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	printf( "Time to generate: %3.1f ms\n", elapsedTime);
	
	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );
	
	HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost ) );
	bitmap.display_and_exit();
	
	cudaFree( dev_bitmap );	
}

void testSpheres()
{
	cudaEvent_t start, stop;
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
	
	CPUBitmap bitmap( DIM, DIM );
	unsigned char *dev_bitmap;
    Sphere *spheres;
	
	HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
	HANDLE_ERROR( cudaMalloc( (void**)&spheres, sizeof(Sphere)*SPHERES ) );
	
	Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
	
	for(int i=0; i<SPHERES; i++) {
		temp_s[i].r = rnd( 1.0f );
		temp_s[i].g = rnd( 1.0f );
		temp_s[i].b = rnd( 1.0f );
		temp_s[i].x = rnd( 1000.0f ) - 500;
		temp_s[i].y = rnd( 1000.0f ) - 500;
		temp_s[i].z = rnd( 1000.0f ) - 500;
		temp_s[i].radius = rnd( 100.0f ) + 20;
	}
	
    HANDLE_ERROR( cudaMemcpy(spheres, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice) );
	
	dim3 grids(DIM/16, DIM/16);
	dim3 threads(16, 16);
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
	spheresKernel<<<grids,threads>>>(dev_bitmap, spheres);
	
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	
	float elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	printf( "Time to generate: %3.1f ms\n", elapsedTime);
	
	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );
	
	HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost ) );
	bitmap.display_and_exit();
	
	cudaFree( dev_bitmap );	
    cudaFree( spheres );
}

#pragma endregion


int main (int argc, char* argv[])
{
    //========== laba 3 ================
    
    //parallelExample_threads();
    //testAnimation();
    //testDotProduct();
    //estSharedRaster();
    
    
    //========== laba 4 ================
    testSpheres();
    //testSpheres_constant();


    //========== laba 5 ================

    return 0;
}