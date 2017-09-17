#include "excludes/cudaSiftD.cu"
#include <stdio.h>
#include "cudaUtils.h"

void testcopyKernel( cudaStream_t &stream )
{
	kernel<<<1, 10, 0, stream>>>();
}

void sharedKernel( cudaStream_t &stream )
{
	float *d_data;
	float *h_data;
	int dx = 10;
	h_data = (float *)malloc( dx * sizeof( float));
	CUDA_SAFECALL( cudaMalloc( (void **)&d_data, (size_t)dx * sizeof( float) ) );

	for (int i = 0; i < dx; ++i)
		h_data[i] = i*i;

	CUDA_SAFECALL( cudaMemcpy((void *)d_data, (void *)h_data, (size_t)(dx * sizeof(float)), cudaMemcpyHostToDevice ) );

	int sx = 8; // if size is incorrect shared memory will still work fine
	dim3 blockDim(4,1,1);
	dim3 gridDim(2,1,1);
	shKernel<<<gridDim, blockDim, sx*sizeof( float ), stream>>>( d_data );

	free( h_data );
	CUDA_SAFECALL( cudaFree( d_data ));
}
