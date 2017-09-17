#ifndef CUDA_SIFT_D_H
#define CUDA_SIFT_D_H
#include <stdio.h>
// Define sift constants
#define SIGMA 2.0f
#define N_OCTAVES 4
#define N_SCALES 2
#define MIN_THRESH 2.0f
#define R_THRESH 10.0f

#define WIDTH_CONV_BLOCK 32
#define HEIGHT_CONV_BLOCK 8


// Define device free functions
__device__ int cudaIDivUp( int num, int den ){return (num%den != 0) ? (num/den + 1) : (num/den);}
__device__ int cudaIAlignUp( int A, int a ){return (A%a != 0) ? (A + a - A%a) : (A);}
__device__ int cuda2DTo1D(int x, int y, int width){return x + y * width;}
__device__ void cudaMemcpyGlobalToShared( float *s, const float *g
								, const int tx, const int ty, const int gx, const int gy
								, const int bDimX, const int bDimY, const int w, const int p, const int h
								, const int apronLeft, const int apronRight, const int apronUp, const int apronDown )
{
	int sx = 0;
	int sy = 0;
	int gx_ = 0;
	int gy_ = 0;
	int sDimX = bDimX + apronLeft + apronRight;
	int sDimY = bDimY + apronUp + apronDown;
	int apronBlocksX = cudaIDivUp(apronLeft, bDimX) + cudaIDivUp(apronRight, bDimX);
	int apronBlocksY = cudaIDivUp(apronUp, bDimY) + cudaIDivUp(apronDown, bDimY);

	for (int j = 0; j < apronBlocksY + 1; ++j)
	{
		sy = ty + j*bDimY;
		for (int i = 0; i < apronBlocksX + 1; ++i)
		{
			sx = tx + i*bDimX;

			if (sx >= 0 && sx < sDimX && sy >= 0 && sy < sDimY)
			{
				gx_ = gx - apronLeft + i*bDimX;
				gy_ = gy - apronUp + j* bDimY;
				if(gx_ >= 0 && gx_ < w && gy_ >= 0 && gy_ < h)
					s[cuda2DTo1D( sx, sy, sDimX)] = g[cuda2DTo1D( gx_, gy_, p )];
				else
					s[cuda2DTo1D( sx, sy, sDimX )] = 0;
				printf( "sx, sy, gx_, gy_, s\t %d, %d, %d, %d, %f \n", sx, sy, gx_, gy_, s[cuda2DTo1D( sx, sy, sDimX )] );
			}
		}
	}
	__syncthreads();
}

#endif
