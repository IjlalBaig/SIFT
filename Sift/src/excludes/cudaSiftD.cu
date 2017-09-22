#include "cudaSiftD.h"
#include <stdio.h>



__global__ void blurKernel( float *gDst, float *gSrc
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
	int sx = 0;
	int sy = 0;
	int gx = tx + bDimX * bIdxX * nTilesX;
	int gx_ = 0;
	int gy = ty + bDimY * bIdxY * nTilesY;
	int dataSizeX = nTilesX*bDimX;
	int sDimX = (apronLeft + dataSizeX + apronRight + bankOffset);
	extern __shared__ float shared[];

	// Load data to shared
	cudaMemcpyGlobalToShared(shared, gSrc, tx, ty
							, gx, gy, bDimX, bDimY, w, p, h
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, bankOffset);

	// Convolve-x
	for (int i = 0; i < N_SCALES + 3; ++i)
	{
//		apronOld += c_GaussianBlurSize[i] - 1;
//		apronActive = apronLeft + apronRight - apronOld;
//		activeTiles = cudaIDivUpNear(dataSizeX + apronActive, bDimX);
//
//		if (gx == 0 && gy == 0){
//			printf("scale \t %d\nfilter size \t %d\n", i, c_GaussianBlurSize[i]);
//			printf("apronActive \t %d\nactiveTiles \t %d\n", apronActive, activeTiles);
//		}

		for (int j = 0; j < nTilesX; ++j)
		{
			sx = tx + j*bDimX;
			sy = ty;
			gx_ = sx + bDimX * bIdxX * nTilesX;

			if (sx < dataSizeX && gx_ < w)
			{
				float sum = 0;
				for (int k = 0; k < B_KERNEL_SIZE; ++k)
					sum = __fmaf_rn(c_GaussianBlur[i * B_KERNEL_SIZE + k], shared[cuda2DTo1D(sx + k, sy, sDimX)], sum);
				gDst[cuda2DTo1D(gx + j*bDimX, gy, p)] = sum;
			}
		}
//		__syncthreads();

//		// Copy data to global
//		cudaMemcpySharedToGlobal(gDst, shared
//								, tx, ty, gx, gy
//								, bDimX, bDimY, w, p, h
//								, nTilesX, nTilesY
//								, apronLeft, apronRight, apronUp, apronDown, bankOffset);
	}
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

	__syncthreads();
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
//	if (gx == 0 && gy == 0)
//	{
//		for (int j = 0; j < h; ++j)
//		{
//			for (int i = 0; i < w; ++i)
//			{
//				printf( "%f  ", data[cuda2DTo1D( i, j, p)] );
//			}
//			printf( "\n" );
//		}
//
//	}
}
