#include "cudaSiftD.h"
#include "../sift.h"
#include <stdio.h>

__global__ void subtractKernel( float *gDst, float *gSrc1, float *gSrc2
							, int w, int p, int h )
{
	int gx = threadIdx.x + blockDim.x * blockIdx.x;
	int gy = threadIdx.y + blockDim.y * blockIdx.y;
	int gIdx = gx + p * gy;

	// Compute difference
	if (gx < w && gy < h)
		gDst[gIdx] = gSrc1[gIdx] - gSrc2[gIdx];
}

__global__ void findExtremaKernel( SiftPoint *pt, float *gDoG, float *gHessian
								, int w, int p, int h
								, const int nTilesX, const int nTilesY
								, const int apronLeft, const int apronRight, const int apronUp, const int apronDown
								, const int bankOffset, const int scaleIdx )
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int sx = 0;
	int sy = 0;
	int bDimX = blockDim.x;
	int bDimY = blockDim.y;
	int bIdxX = blockIdx.x;
	int bIdxY = blockIdx.y;
	int gx = tx + bDimX * bIdxX * nTilesX;
	int gx_ = 0;
	int gy = ty + bDimY * bIdxY * nTilesY;
	int gIdx = 0;
	int dataSizeX = nTilesX*bDimX;
	int dataSizeY = nTilesY*bDimY;
	int sDimX = (apronLeft + dataSizeX + apronRight + bankOffset);
	int gDim = p*h;
	extern __shared__ float shared[];
	float *sharedScale0 = shared;
	float *sharedScale1 = shared + dataSizeX*dataSizeY;
	float *sharedScale2 = shared + 2*dataSizeX*dataSizeY;
	float pxVal = 0.0;
	float pxRank = 0.0;
	float trace = 0.0;
	float det = 0.0;
	int ptCount = 0;


	// Load data to shared
	cudaMemcpyGlobalToShared( sharedScale0, gDoG, tx, ty
							, gx, gy, bDimX, bDimY, w, p, h
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, bankOffset );
	cudaMemcpyGlobalToShared( sharedScale1, gDoG + gDim, tx, ty
							, gx, gy, bDimX, bDimY, w, p, h
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, bankOffset );
	cudaMemcpyGlobalToShared( sharedScale2, gDoG + 2*gDim, tx, ty
							, gx, gy, bDimX, bDimY, w, p, h
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, bankOffset );

	for (int i = 0; i < nTilesX; ++i)
	{
		gx_ = gx + i*bDimX;
		if(gx_ < w || gy < h)
		{
			sx = apronLeft + tx + i*bDimX;
			sy = apronUp + ty;
			gIdx = cuda2DTo1D( gx_, gy, p );
			pxVal = sharedScale1[cuda2DTo1D( sx, sy, sDimX )];

			//	Compare 3x3 region in all scales (extrema if pxRank == 9.0)
			for (int l = -1; l < 2; ++l)
			{
				for (int m = -1; m < 2; ++m)
				{
					pxRank += (pxVal >= sharedScale0[cuda2DTo1D( sx + l, sy + m, sDimX )]) ? (1.0):(-1.0);
					pxRank += (pxVal >= sharedScale1[cuda2DTo1D( sx + l, sy + m, sDimX )]) ? (1.0):(-1.0);
					pxRank += (pxVal >= sharedScale2[cuda2DTo1D( sx + l, sy + m, sDimX )]) ? (1.0):(-1.0);
				}
			}

			//	Check if extrema has low edge response
			trace = (gHessian + 0*gDim)[gIdx] + (gHessian + 1*gDim)[gIdx];
			det = (gHessian + 0*gDim)[gIdx] * (gHessian + 1*gDim)[gIdx] + pow((gHessian + 2*gDim)[gIdx], 2);
			if ( pxRank == 9.0
				&& trace*trace/det < (pow( R_THRESH * 1, 2 ))/R_THRESH
				&& abs(pxVal) > EXTREMA_THRESH
				&&  d_PointCounter[scaleIdx] < c_MaxPointCount)
			{
				ptCount = atomicAdd(&d_PointCounter[scaleIdx], 1);
				pt[ptCount].xpos = gx_;
				pt[ptCount].ypos = gy;
			}
		}
	}
}
__global__ void gradientKernel( float *gDst, float *gSrc
							, int w, int p, int h
							, const int nTilesX, const int nTilesY
							, const int apronLeft, const int apronRight, const int apronUp, const int apronDown
							, const int bankOffset )
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int sx = 0;
	int sy = 0;
	int bDimX = blockDim.x;
	int bDimY = blockDim.y;
	int bIdxX = blockIdx.x;
	int bIdxY = blockIdx.y;
	int gx = tx + bDimX * bIdxX * nTilesX;
	int gy = ty + bDimY * bIdxY * nTilesY;
	int gx_ = 0;
	int dataSizeX = nTilesX*bDimX;
	int sDimX = (apronLeft + dataSizeX + apronRight + bankOffset);
	int gDim = p*h;
	float magnitude = 0.0;
	float direction = 0.0;
	float lUp = 0.0;
	float lDown = 0.0;
	float lLeft = 0.0;
	float lRight = 0.0;
	extern __shared__ float shared[];

	// Load data to shared
	cudaMemcpyGlobalToShared( shared, gSrc, tx, ty
							, gx, gy, bDimX, bDimY, w, p, h
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, bankOffset );

	for (int i = 0; i < nTilesX; ++i)
	{
		gx_ = gx + i*bDimX;
		if (gx_ < w && gy < h)
		{
			sx = apronLeft + tx + i*bDimX;
			sy = apronUp + ty;

			//	Gather neighbor pixels
			lUp = shared[cuda2DTo1D( sx, sy - 1, sDimX )];
			lDown = shared[cuda2DTo1D( sx, sy + 1, sDimX )];
			lLeft = shared[cuda2DTo1D( sx - 1, sy, sDimX )];
			lRight = shared[cuda2DTo1D( sx + 1, sy, sDimX )];

			//	Compute magnitude + direction
			magnitude = sqrtf(powf( lRight - lLeft, 2 ) + powf( lDown - lUp, 2 ));
			direction = (180/PI) * atan2( lDown - lUp, lRight - lLeft );

			// Copy data to global
			gDst[cuda2DTo1D( gx_, gy, p )] = magnitude;
			gDst[cuda2DTo1D( gx_, gy, p ) + gDim] = direction;
		}
	}

}

__global__ void hessianKernel( float *gDst, float *gSrc
							, int w, int p, int h
							, const int nTilesX, const int nTilesY
							, const int apronLeft, const int apronRight, const int apronUp, const int apronDown
							, const int bankOffset )
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int sx = 0;
	int sy = 0;
	int bDimX = blockDim.x;
	int bDimY = blockDim.y;
	int bIdxX = blockIdx.x;
	int bIdxY = blockIdx.y;
	int gx = tx + bDimX * bIdxX * nTilesX;
	int gy = ty + bDimY * bIdxY * nTilesY;
	int gx_ = 0;
	int dataSizeX = nTilesX*bDimX;
	int sDimX = (apronLeft + dataSizeX + apronRight + bankOffset);
	int gDim = p*h;
	float Dxx = 0;
	float Dxy = 0;
	float Dyy = 0;
	extern __shared__ float shared[];

	// Load data to shared
	cudaMemcpyGlobalToShared( shared, gSrc, tx, ty
							, gx, gy, bDimX, bDimY, w, p, h
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, bankOffset );

	for (int i = 0; i < nTilesX; ++i)
	{
		gx_ = gx + i*bDimX;
		if (gx_ < w && gy < h)
		{
			sx = apronLeft + tx + i*bDimX;
			sy = apronUp + ty;

			//	Compute Dxx
			Dxx = -2*shared[cuda2DTo1D( sx, sy, sDimX )];
			Dxx += shared[cuda2DTo1D( sx - 1, sy, sDimX )];
			Dxx += shared[cuda2DTo1D( sx + 1, sy, sDimX )];

			//	Compute Dyy
			Dyy = -2*shared[cuda2DTo1D( sx, sy, sDimX )];
			Dyy += shared[cuda2DTo1D( sx, sy - 1, sDimX )];
			Dyy += shared[cuda2DTo1D( sx, sy + 1, sDimX )];

			//	Compute Dxy
			Dxy = shared[cuda2DTo1D( sx, sy, sDimX )];
			Dxy += shared[cuda2DTo1D( sx - 1, sy - 1, sDimX )];
			Dxy -= shared[cuda2DTo1D( sx - 1, sy, sDimX )];
			Dxy -= shared[cuda2DTo1D( sx, sy - 1, sDimX )];

			// Copy data to global
			gDst[cuda2DTo1D( gx_, gy, p )] = Dxx;
			gDst[cuda2DTo1D( gx_, gy, p ) + gDim] = Dyy;
			gDst[cuda2DTo1D( gx_, gy, p ) + 2*gDim] = Dxy;
		}
	}

}

__global__ void xblurMultiKernel( float *gDst, float *gSrc
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
	int gDim = p*h;
	extern __shared__ float shared[];

	// Load data to shared
	cudaMemcpyGlobalToShared( shared, gSrc, tx, ty
							, gx, gy, bDimX, bDimY, w, p, h
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, bankOffset );

	// Convolve-x
	for (int i = 0; i < N_SCALES + 3; ++i)
	{
		int kernelStartIdx = i * B_KERNEL_SIZE;
		for (int j = 0; j < nTilesX; ++j)
		{
			sx = tx + j*bDimX;
			sy = ty;
			gx_ = sx + bDimX * bIdxX * nTilesX;

			if (sx < dataSizeX && gx_ < w && gy < h)
			{
				float sum = 0;
				for (int k = 0; k < B_KERNEL_SIZE; ++k)
					sum = __fmaf_rn( c_GaussianBlur[kernelStartIdx + k], shared[cuda2DTo1D( sx + k, sy, sDimX )], sum );
				__syncthreads();
				gDst[cuda2DTo1D( gx + j*bDimX, gy, p ) + i*gDim] = sum;
			}
		}
//		// Copy data to global
//		cudaMemcpySharedToGlobal(gDst, shared
//								, tx, ty, gx, gy
//								, bDimX, bDimY, w, p, h
//								, nTilesX, nTilesY
//								, apronLeft, apronRight, apronUp, apronDown, bankOffset);
	}
}

__global__ void yblurKernel( float *gDst, float *gSrc, const int scaleIdx
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
	int gy = ty + bDimY * bIdxY * nTilesY;
	int gy_ = 0;
	int dataSizeY = nTilesY*bDimY;
	int sDimX = (bDimX + bankOffset);
	int kernelStartIdx = scaleIdx * B_KERNEL_SIZE;
	extern __shared__ float shared[];

	// Load data to shared
	cudaMemcpyGlobalToShared( shared, gSrc, tx, ty
							, gx, gy, bDimX, bDimY, w, p, h
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, bankOffset );

	// Convolve-y
	for (int i = 0; i < nTilesY; ++i)
	{
		sx = tx;
		sy = ty + i*bDimY;
		gy_ = sy + bDimY * bIdxY * nTilesY;

		if (sy < dataSizeY && gy_ < h && gx < w)
		{
				float sum = 0;
				for (int j = 0; j < B_KERNEL_SIZE; ++j)
					sum = __fmaf_rn( c_GaussianBlur[kernelStartIdx + j], shared[cuda2DTo1D(sx, sy + j, sDimX)], sum );
				__syncthreads();
				gDst[cuda2DTo1D( gx, gy + i*bDimY, p )] = sum;
		}
	}

}

__global__ void copyKernel( float *gDst, const float *gSrc, int w, int p, int h )
{
	int gx = threadIdx.x + blockDim.x * blockIdx.x;
	int gy = threadIdx.y + blockDim.y * blockIdx.y;
	int gIdx = gx + p * gy;
	gDst[gIdx] = gSrc[gIdx];
}

__global__ void subtractKernel( float *gDst, const float *gSrc1, const float *gSrc2, int w, int p, int h )
{
	int gx = threadIdx.x + blockDim.x * blockIdx.x;
	int gy = threadIdx.y + blockDim.y * blockIdx.y;
	int gIdx = gx + p * gy;

	// Compute difference
	if (gx < w && gy < h)
		gDst[gIdx] = gSrc1[gIdx] - gSrc2[gIdx];
}

__global__ void resizeKernel( float *gDst, float *gSrc, int w, int p, int h )
{
	int gx = threadIdx.x + blockDim.x * blockIdx.x;
	int gy = threadIdx.y + blockDim.y * blockIdx.y;
	int gIdx = gx + p * gy;

	// Resize to half size
	if(gx < w && gy < h && gx%2 == 0 && gy%2 == 0 )
	{
		int gx_ = gx / 2.0f;
		int gy_ = gy / 2.0f;
		int p_ = cudaIAlignUp( cudaIDivUpNear( w, 2 ), 128 );
		int gIdx_ = gx_ + gy_ * p_;
		gDst[gIdx_] = gSrc[gIdx];
	}
}


//__global__ void kernelGaussianSize()
//{
//	int tx = threadIdx.x;
//	printf( "scale %d\t:\t%d\n", tx, c_GaussianBlurSize[tx] );
//	if (tx == 0)
//		printf( "%d\n", c_MaxGaussianBlurSize );
//}

__global__ void kernelGaussianVector()
{
	int tx = threadIdx.x;
	printf( "thread %d\t:\t%f\n", tx, c_GaussianBlur[tx] );
}

__global__ void kernel()
{
	int count = 0;
	for (int i = 0; i < 5000; ++i)
		count++;
	printf( "hi this is thread:%d\t%d\n", threadIdx.x, count );
}

__global__ void shKernel( float *data
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
		for (int j = 0; j < apronUp +  bDimY*nTilesY + apronDown; ++j)
		{
			for (int i = 0; i < apronLeft + bDimX*nTilesX + apronRight; ++i)
			{
				printf( "%f  ", shared[cuda2DTo1D( i, j, apronLeft + bDimX*nTilesX + apronRight )] );
			}
			printf( "\n" );
		}
	}

	cudaMemcpySharedToGlobal( data, shared
							, tx, ty, gx, gy
							, bDimX, bDimY, w, p, h
							, nTilesX, nTilesY
							, apronLeft, apronRight, apronUp, apronDown, 0 );
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
