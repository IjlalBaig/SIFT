#include "cudaSiftD.h"
#include <stdio.h>





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





























__global__ void blurKernel(float *dst, float *src,
					int w, int p, int h,
					const int apronLeft, const int apronRight, const int apronUp, const int apronDown )
{


	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bDimX = blockDim.x;
	int bDimY = blockDim.y;
	int bIdxX = blockIdx.x;
	int bIdxY = blockIdx.y;
	// Avoid bank conflict by ensuring shared data words are odd numbered
	int apronRight_ = ( (apronLeft + bDimX + apronRight)%2 != 0 ) ? ( apronRight ):( apronRight + 1 );

	int kernelOffset = 0;
	int gx = tx + bDimX * bIdxX;
	int gy = ty + bDimY * bIdxY;
	int gIdx = gx + p * gy;
	int gDim = p*h;
	int sDimX = (apronLeft + bDimX + apronRight_);
	int sDimY = (apronUp + bDimY + apronDown);
	int sDim = sDimX * sDimY;
	int sx;
	int sy;

	extern __shared__ float shared[];
	float *s = &shared[0];
	float *s_ = &shared[sDim];

	// Load shared data
	cudaMemcpyGlobalToShared(shared, src, tx, ty, gx, gy, bDimX, bDimY, w, p, h, apronLeft, apronRight_, apronUp, apronDown);


	for (int i = 0; i < N_SCALES + 3; ++i)
	{
		// Convolve-X and transpose
		if (gx < w)
		{
			for (int j = 0; j < 0; ++j)	// loop tBlock over sDimX
			{
				sy = ty + j * cudaIDivUpNear(sDimY, bDimY);
				for (int k = 0; k < 0; ++k) // loop tBlock over sDimY
				{
				sx = tx + k * cudaIDivUpNear(sDimX, bDimX);
					if (sx < sDimX && sy < sDimY)
					{
						float sum = 0;
						for (int l = 0; l < c_GaussianBlurSize[i]; ++l)
							sum =__fmaf_rn(c_GaussianBlur[kernelOffset + l], s[cuda2DTo1D(sx + l, sy, bDimX + apronLeft + apronRight_)], sum);
						s_[cuda2DTo1D( sx, sy, bDimX + apronLeft + apronRight_)] = sum;
						__syncthreads();
						dst[cuda2DTo1D( sx + bDimX * bIdxY, sy + bDimY * bIdxY, p )] = sum;
						kernelOffset += c_GaussianBlurSize[i];
					}
				}
			}
			__syncthreads();
		}
	}
}

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
