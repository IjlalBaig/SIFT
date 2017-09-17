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
	int w = 9;
	int p = 10;
	int h = 5;
	int gx = 50;
	h_data = (float *)malloc( gx * sizeof( float));
	CUDA_SAFECALL( cudaMalloc( (void **)&d_data, (size_t)gx * sizeof( float) ) );

	for (int i = 0; i < p; ++i)
	{
		for (int j = 0; j < h; ++j)
			h_data[i + j*p] = (i < w) ? (i*i): -1;
	}

	CUDA_SAFECALL( cudaMemcpy((void *)d_data, (void *)h_data, (size_t)(gx * sizeof(float)), cudaMemcpyHostToDevice ) );

	int sx = 5+4; // if size is incorrect shared memory will still work fine
	int sy = 2+8;
	dim3 blockDim(4,2,1);
	dim3 gridDim(1,1,1);
	shKernel<<<gridDim, blockDim, sx*sy*sizeof( float ), stream>>>( d_data, w, p, h, 5, 0, 3, 5 );

	free( h_data );
	CUDA_SAFECALL( cudaFree( d_data ));
}
