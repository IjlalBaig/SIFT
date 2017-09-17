#include "cudaSiftD.h"
#include <stdio.h>

__global__ void kernel()
{
	int count = 0;
	for (int i = 0; i < 5000; ++i)
		{count++;}
	printf("hi this is thread:%d\t%d\n", threadIdx.x, count);
}

__global__ void shKernel(float *data, int w, int p, int h, const int apronLeft, const int apronRight, const int apronUp, const int apronDown )
{
	extern __shared__ float shared[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bDimX = blockDim.x;
	int bDimY = blockDim.y;
	int gx = tx + bDimX * blockIdx.x;
	int gy = ty + bDimY * blockIdx.y;
//	if (gx == 0 && gy == 0)
//	{
//		for (int j = 0; j < h; ++j)
//		{
//			for (int i = 0; i < p; ++i)
//			{
//				printf( "%f  ", data[cuda2DTo1D( i, j, p )] );
//			}
//			printf( "\n" );
//		}
//
//	}
	cudaMemcpyGlobalToShared( shared, data
							, tx, ty, gx, gy
							, bDimX, bDimY, w, p, h
							, apronLeft, apronRight, apronUp, apronDown );
	if (gx == 0 && gy == 0)
	{
		for (int j = 0; j < bDimY + apronUp + apronDown; ++j)
		{
			for (int i = 0; i < bDimX + apronLeft + apronRight; ++i)
			{
				printf( "%f  ", shared[cuda2DTo1D( i, j, bDimX + apronLeft + apronRight )] );
			}
			printf( "\n" );
		}

	}


}
