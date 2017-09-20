#include "cudaSiftD.h"
#include <stdio.h>



__global__ void blurKernel(float *gDst, float *gSrc
						, int w, int p, int h
						, const int nTilesX, const int nTilesY
						, const int apronLeft, const int apronRight, const int apronUp, const int apronDown
						, const int bankOffset)
{


	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bDimX = blockDim.x;
	int bDimY = blockDim.y;
	int bIdxX = blockIdx.x;
	int bIdxY = blockIdx.y;
	int kernelOffset = 0;
	int gx = tx + bDimX * bIdxX * nTilesX;
	int gy = ty + bDimY * bIdxY * nTilesY;
	int gIdx = gx + p * gy;
	int sDimX = (apronLeft + bDimX + apronRight + bankOffset);
	int sDimY = (apronUp + bDimY + apronDown);

	extern __shared__ float shared[];

	// Load data to shared
	cudaMemcpyGlobalToShared(shared, gSrc, tx, ty
							, gx, gy, bDimX, bDimY, w, p, h
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, bankOffset);
	// Copy data to global
	cudaMemcpySharedToGlobal(gDst, shared
							, tx, ty, gx, gy
							, bDimX, bDimY, w, p, h
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, bankOffset);
}




__global__ void kernelGaussianSize()
{
	int tx = threadIdx.x;
	printf("scale %d\t:\t%d\n", tx, c_GaussianBlurSize[tx]);
	if (tx == 0)
		printf("%d\n", c_MaxGaussianBlurSize);
}

__global__ void kernelGaussianVector()
{
	int tx = threadIdx.x;
	printf("thread %d\t:\t%f\n", tx, c_GaussianBlur[tx]);
}

__global__ void kernel()
{
	int count = 0;
	for (int i = 0; i < 5000; ++i)
		{count++;}
	printf("hi this is thread:%d\t%d\n", threadIdx.x, count);
}

__global__ void shKernel(float *data
						, int w, int p, int h
						, const int nTilesX, const int nTilesY
						, const int apronLeft, const int apronRight, const int apronUp, const int apronDown )
{
	extern __shared__ float shared[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bDimX = blockDim.x;
	int bDimY = blockDim.y;
	int bIdxX = blockIdx.x;
	int bIdxY = blockIdx.y;
	int gx = tx + bDimX * bIdxX;
	int gy = ty + bDimY * bIdxY;
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
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, 0 );
	if (gx == 0 && gy == 0)
	{
		for (int j = 0; j < apronUp +  bDimY*(nTilesY) + apronDown; ++j)
		{
			for (int i = 0; i < apronLeft + bDimX*(nTilesX) + apronRight; ++i)
			{
				printf( "%f  ", shared[cuda2DTo1D( i, j, apronLeft + bDimX*nTilesX + apronRight )] );
			}
			printf( "\n" );
		}
	}
	cudaMemcpySharedToGlobal(data, shared
							, tx, ty, gx, gy
							, bDimX, bDimY, w, p, h
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, 0);
	if (gx == 0 && gy == 0)
	{
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				printf( "%f  ", data[cuda2DTo1D( i, j, p)] );
			}
			printf( "\n" );
		}

	}
}
