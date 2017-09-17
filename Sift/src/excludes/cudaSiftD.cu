#include "cudaSiftD.h"
#include <stdio.h>

__global__ void kernel()
{
	int count = 0;
	for (int i = 0; i < 5000; ++i)
		{count++;}
	printf("hi this is thread:%d\t%d\n", threadIdx.x, count);
}

__global__ void shKernel(float *data)
{
	extern __shared__ float shared[];
	float *shMag, *shOri;
	shMag = &shared[0];
	shOri = &shared[blockDim.x+1];
	int tx = threadIdx.x;
	if (tx < blockDim.x && blockIdx.x == 0)
		shMag[tx] = data[tx];
	if (tx < blockDim.x && blockIdx.x == 1)
		shOri[tx] = data[tx+ blockIdx.x*blockDim.x];
	__syncthreads();
	printf("threadIdx.x\t%d\t:\t%f\n", tx, shared[tx + blockIdx.x*blockDim.x]);

}
