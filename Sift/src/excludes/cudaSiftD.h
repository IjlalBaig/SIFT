#ifndef CUDA_SIFT_D_H
#define CUDA_SIFT_D_H
#include <stdio.h>
#include "../utils.h"
// Define sift constants
#define SIGMA 1.6f
#define MIN_THRESH 15.0f
#define R_THRESH 10.0f

// Define kernel parameters
//#define B_KERNEL_RADIUS 4
#define B_KERNEL_SIZE  (2*B_KERNEL_RADIUS + 1)


#define WIDTH_CONV_BLOCK 32
#define HEIGHT_CONV_BLOCK 32
#define DEFUALT_BLOCK_DIM 32
#define ORIENT_BLOCK_DIM 16


// Constant memory variables
__constant__ float c_GaussianBlur[(N_SCALES + 3) * B_KERNEL_SIZE];
__constant__ float c_GaussianWnd[N_SCALES * WND_KERNEL_SIZE];
__constant__ float c_EdgeThreshold;
__constant__ float c_ExtremaThreshold;
__constant__ unsigned int c_MaxPointCount;

// static memory variables
__device__ unsigned int d_PointCount[2];
__device__ unsigned int d_PointStartIdx[2];
// Define device free functions
__device__ int cudaIDivUpOdd( int num, int den )
{
	int result = (num%den != 0) ? (num/den + 1) : (num/den);
	return (result%2 != 0) ? (result):(result+1);
}

__device__ int cudaIDivUpNear( int num, int den ){return (num%den != 0) ? (num/den + 1) : (num/den);}
__device__ int cudaIAlignUp( int A, int a ){return (A%a != 0) ? (A + a - A%a) : (A);}
__device__ int cudaAssignBin(float theta, int nBins)
{
	float binRange = 360.0f / nBins;
	theta = (theta < 0) ? (theta + 360) : (theta);
	theta = fmodf(theta, 360.0);
	return __float2int_rd(theta / binRange);
}
__device__ int cuda2DTo1D(int x, int y, int width){return x + y * width;}
__device__ void cudaMemcpyGlobalToShared( float *s, const float *g
					, const int tx, const int ty, const int gx, const int gy
					, const int bDimX, const int bDimY, const int w, const int p, const int h
					, const int nTilesX, const int nTilesY
					, const int apronLeft, const int apronRight, const int apronUp, const int apronDown, const int bankOffset)
{
	int sx = 0;
	int sy = 0;
	int gx_ = 0;
	int gy_ = 0;
	int sDimX = apronLeft + bDimX*nTilesX + apronRight + bankOffset;
	int sDimY = apronUp + bDimY*nTilesY + apronDown;
	int apronBlocksX = cudaIDivUpNear(apronLeft, bDimX) + cudaIDivUpNear(apronRight, bDimX);
	int apronBlocksY = cudaIDivUpNear(apronUp, bDimY) + cudaIDivUpNear(apronDown, bDimY);
	int jLoop = nTilesY + apronBlocksY;
	int iLoop = nTilesX + apronBlocksX;

	for (int j = 0; j < jLoop; ++j)
	{
		sy = ty + j*bDimY;
		for (int i = 0; i < iLoop; ++i)
		{
			sx =  tx + i*bDimX;
			if (sx >= 0 && sx < sDimX - bankOffset && sy >= 0 && sy < sDimY)
			{
				gx_ = gx  - apronLeft + i*bDimX;
				gy_ = gy - apronUp + j*bDimY;
				if(gx_ >= 0 && gx_ < w && gy_ >= 0 && gy_ < h)
					s[cuda2DTo1D( sx, sy, sDimX)] = g[cuda2DTo1D( gx_, gy_, p )];
				else
					s[cuda2DTo1D( sx, sy, sDimX )] = 0;
			}
		}
	}
	__syncthreads();
}
__device__ void shflmax(int* const maxIdx, const float* const vin)
{
	int tIdx = cuda2DTo1D(threadIdx.x, threadIdx.y, blockDim.x);
	float v = vin[tIdx];
	float m = v;
	int idx = 0;
	v = max(v,__shfl_xor(v,16));
	v = max(v,__shfl_xor(v, 8));
	v = max(v,__shfl_xor(v, 4));
	v = max(v,__shfl_xor(v, 2));
	v = max(v,__shfl_xor(v, 1));
	idx = __ffs(__ballot(m == v));

	if (tIdx%32 == 0)
		maxIdx[tIdx/32] = idx-1;
}

__device__ float shflsum(const float* const vin)
{
	int tIdx = cuda2DTo1D(threadIdx.x, threadIdx.y, blockDim.x);
	float v = vin[tIdx];
	v +=__shfl_down(v,16);
	v +=__shfl_down(v, 8);
	v +=__shfl_down(v, 4);
	v +=__shfl_down(v, 2);
	v +=__shfl_down(v, 1);
	return v;
}
__device__ void cudaMemcpySharedToGlobal( float *g, const float *s
					, const int tx, const int ty, const int gx, const int gy
					, const int bDimX, const int bDimY, const int w, const int p, const int h
					, const int nTilesX, const int nTilesY
					, const int apronLeft, const int apronRight, const int apronUp, const int apronDown, const int bankOffset)
{
	int sx = 0;
	int sy = 0;
	int gx_ = 0;
	int gy_ = 0;
	int sDimX = apronLeft + bDimX*nTilesX + apronRight + bankOffset;
	int sDimY = apronUp + bDimY*nTilesY + apronDown;

	for (int j = 0; j < nTilesY; ++j)
	{
		sy = apronUp + ty + j*bDimY;
		for (int i = 0; i < nTilesX; ++i)
		{
			sx =  apronLeft + tx + i*bDimX;

			if (sx >= apronLeft && sx < sDimX - apronRight - bankOffset && sy >= apronUp && sy < sDimY - apronDown)
			{
				gx_ = gx + i*bDimX;
				gy_ = gy + j*bDimY;
				if(gx_ >= 0 && gx_ < w && gy_ >= 0 && gy_ < h)
					g[cuda2DTo1D( gx_, gy_, p )] = s[cuda2DTo1D( sx, sy, sDimX)];
			}
		}
	}
	__syncthreads();
}

#endif
