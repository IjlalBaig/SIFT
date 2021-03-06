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

__global__ void OrientationKernel( SiftPoint *pt, float *gGradient
								, int w, int p, int h
								, const int scalePtCnt, const int scaleIdx, const float scale
								, const int streamIdx )
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bDimX = blockDim.x;
	int bIdxX = blockIdx.x;
	int tIdx = cuda2DTo1D(tx, ty, bDimX);
	int lPtIdx = tIdx + bIdxX*16;
	int binVal = 0;
	__shared__ float sharedPtPos[ORIENT_BUFFER][2];
	__shared__ float sharedMag[ORIENT_BUFFER][16*16];
	__shared__ float sharedDir[ORIENT_BUFFER][16*16];
	__shared__ float sharedHist[ORIENT_BUFFER*32];
	__shared__ int sharedHistMaxIdx[ORIENT_BUFFER];

	int tPtCnt = ( scalePtCnt - (bIdxX + 1)*ORIENT_BUFFER  > 0)? ORIENT_BUFFER : scalePtCnt%ORIENT_BUFFER;


	//	Load pt positions
	if (tIdx < tPtCnt && d_PointStartIdx[streamIdx] + lPtIdx < MAX_POINTCOUNT)
	{
		sharedPtPos[tIdx][0] =  pt[ d_PointStartIdx[streamIdx] + lPtIdx].xpos;
		sharedPtPos[tIdx][1] =  pt[ d_PointStartIdx[streamIdx] + lPtIdx].ypos;
	}
	__syncthreads();

	// 	Load gradient magnitude and direction regions
	#pragma unroll
	for (int i = 0; i < ORIENT_BUFFER; ++i)
	{
		sharedMag[i][cuda2DTo1D(tx, ty, bDimX)] = gGradient[cuda2DTo1D(sharedPtPos[i][0] - 7 + tx, sharedPtPos[i][1] - 7 + ty, p)];
		sharedDir[i][cuda2DTo1D(tx, ty, bDimX)] = (gGradient + p*h)[cuda2DTo1D(sharedPtPos[i][0] - 7 + tx, sharedPtPos[i][1] - 7 + ty, p)];
	}
	__syncthreads();

	// 	Multiply gaussian wnd
	#pragma unroll
	for (int i = 0; i < ORIENT_BUFFER; ++i)
	{
		sharedMag[i][cuda2DTo1D(tx, ty, bDimX)] *= c_GaussianWnd[ tx + scaleIdx*WND_KERNEL_SIZE];
		sharedMag[i][cuda2DTo1D(ty, tx, bDimX)] *= c_GaussianWnd[ tx + scaleIdx*WND_KERNEL_SIZE];

	}
	__syncthreads();

	//	Initialize histogram
	#pragma unroll
	for (int i = 0; i < 2; ++i)
	{
		sharedHist[tIdx + i*8*32] = 0;
	}
	__syncthreads();
	//	Compute patch histogram
	#pragma unroll
	for (int i = 0; i < ORIENT_BUFFER; ++i)
	{
		binVal = cudaAssignBin(sharedDir[i][cuda2DTo1D(tx, ty, bDimX)], 32);
		atomicAdd(&sharedHist[i*ORIENT_BUFFER + binVal], sharedMag[i][cuda2DTo1D(tx, ty, bDimX)]);
	}
	__syncthreads();

	//	Find histogram max oientation threshold
	#pragma unroll
	for (int i = 0; i < 2; ++i)
	{
		shflmax(&sharedHistMaxIdx[8*i], &sharedHist[i*8*32]);
	}

	//	Save orientation and scale
	if (tIdx < tPtCnt && d_PointStartIdx[streamIdx] + lPtIdx < MAX_POINTCOUNT )
	{
		pt[d_PointStartIdx[streamIdx] + lPtIdx].orientation = 360.0f/32.0f*sharedHistMaxIdx[tIdx];
		pt[d_PointStartIdx[streamIdx] + lPtIdx].scale = scale;
	}
}

__global__ void DescriptorKernel( SiftPoint *pt, cudaTextureObject_t texObjMag, cudaTextureObject_t texObjDir
								, int w, int p, int h
								, const int scalePtCnt, const int scaleIdx, const int streamIdx )
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bDimX = blockDim.x;
	int bIdxX = blockIdx.x;
	int tIdx = cuda2DTo1D(tx, ty, bDimX);
	int lPtIdx = tIdx + bIdxX*16;
	int tPtCnt = ( scalePtCnt - (bIdxX + 1)*16  > 0)? 16 : scalePtCnt%16;
	float u = 0;
	float v = 0;
	float u_ = 0;
	float v_ = 0;
	float uo = 0;
	float vo = 0;
	float sum = 0;
	int sbIdxX = 0;
	int sbIdxY = 0;
	int binVal = 0;

	__shared__ float sharedPtPos[16][2];
	__shared__ float sharedPtOrient[16];
	__shared__ float sharedMag[16][16*16];
	__shared__ float sharedDir[16][16*16];
	__shared__ float sharedHist[16][128];
	__shared__ float sharedHistSum;

	//	Load pt positions
	if (tIdx < tPtCnt && d_PointStartIdx[streamIdx] + lPtIdx < MAX_POINTCOUNT)
	{
		sharedPtPos[tIdx][0] =  pt[ d_PointStartIdx[streamIdx] + lPtIdx].xpos;
		sharedPtPos[tIdx][1] =  pt[ d_PointStartIdx[streamIdx] + lPtIdx].ypos;
		sharedPtOrient[tIdx] = pt[ d_PointStartIdx[streamIdx] + lPtIdx].orientation;
	}
	__syncthreads();

	// 	Load rotated gradient magnitude and direction regions
	for (int i = 0; i < tPtCnt; ++i)
	{
		uo = (sharedPtPos[i][0] - 0.5) / (float)w;
		vo = (sharedPtPos[i][1] - 0.5) / (float)h;
		u_ = uo - 7.5 + tx;
		v_ = vo - 7.5 + ty;
		u = (u_ - uo) * cosf(sharedPtOrient[i]) + (v_ - vo) * sinf(sharedPtOrient[i]) + uo;
		v = (v_ - vo) * cosf(sharedPtOrient[i]) - (u_ - uo) * sinf(sharedPtOrient[i]) + vo;
		sharedMag[i][cuda2DTo1D(tx, ty, bDimX)] = tex2D<float>(texObjMag, u, v);
		sharedDir[i][cuda2DTo1D(tx, ty, bDimX)] = tex2D<float>(texObjDir, u, v);
	}
	__syncthreads();
	// 	Multiply gaussian wnd
	for (int i = 0; i < 16; ++i)
	{
		sharedMag[i][cuda2DTo1D(tx, ty, bDimX)] *= c_DescWnd[ tx ];
		sharedMag[i][cuda2DTo1D(ty, tx, bDimX)] *= c_DescWnd[ tx ];
	}
	__syncthreads();

	//	Initialize histogram
	for (int i = 0; i < 16; ++i)
	{
		if (tIdx < 128)
			sharedHist[i][tIdx] = 0;
	}
	__syncthreads();

	//	Compute patch histogram
	for (int i = 0; i < 16; ++i)
	{
		binVal = cudaAssignBin(sharedDir[i][cuda2DTo1D(tx, ty, bDimX)], 8);
		sbIdxX = tx/4;
		sbIdxY = ty/4;
		atomicAdd(&sharedHist[i][binVal + 8*sbIdxX + 32*sbIdxY], sharedMag[i][cuda2DTo1D(tx, ty, bDimX)]);
	}
	__syncthreads();

	//	Compute descriptor
	for (int i = 0; i < tPtCnt; ++i)
	{
		if (tIdx < 128)
		{
			// 	Normalize histogram
			sum = shflsum(sharedHist[i]);
			__syncthreads();
			if (tIdx%32 == 0)
			{
				atomicAdd(&sharedHistSum, sum);
			}
			__syncthreads();
			sharedHist[i][tIdx] /= sharedHistSum;
			__syncthreads();
			//	Clamp histogram to 0.2
			if (sharedHist[i][tIdx] > 0.2)
				sharedHist[i][tIdx] = 0.2;
			__syncthreads();

			// 	Re-normalize histogram
			sum = shflsum(sharedHist[i]);
			sharedHistSum = 0.0;
			__syncthreads();
			if (tIdx%32 == 0)
				atomicAdd(&sharedHistSum, sum);
			__syncthreads();
			sharedHist[i][tIdx] /= sharedHistSum;
		}
	}

	//	Save descriptor
	for (int i = 0; i < tPtCnt; ++i)
	{
		if (tIdx < 128 )
		{
			pt[d_PointStartIdx[streamIdx] + i + bIdxX*16].data[tIdx] = sharedHist[i][tIdx];
		}
		__syncthreads();
	}

}

__global__ void findExtremaKernel( SiftPoint *pt, float *gDoG, float *gHessian
								, int w, int p, int h
								, const int nTilesX, const int nTilesY
								, const int apronLeft, const int apronRight, const int apronUp, const int apronDown
								, const int bankOffset, const int streamIdx )
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
	float *sharedScale1 = shared + (dataSizeX + bankOffset)*dataSizeY;
	float *sharedScale2 = shared + 2*(dataSizeX + bankOffset)*dataSizeY;
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
			#pragma unroll
			for (int l = -1; l < 2; ++l)
			{
				#pragma unroll
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
			if ( gx_ > 8 && gx_ < w - 8
				&& gy > 8 && gy < h - 8
				&& pxRank == 9.0
				&& trace*trace/det < (pow( R_THRESH * 1, 2 ))/R_THRESH
				&& abs(pxVal) > EXTREMA_THRESH
				&&  d_PointCount[streamIdx] < MAX_POINTCOUNT )
			{
				ptCount = atomicAdd(&d_PointCount[streamIdx], 1);
				if( ptCount < MAX_POINTCOUNT)
				{
					pt[ptCount].xpos = gx_;
					pt[ptCount].ypos = gy;
				}
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
	#pragma unroll
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
				#pragma unroll
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
				#pragma unroll
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
