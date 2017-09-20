#include "excludes/cudaSiftD.cu"
#include <stdio.h>
#include "utils.h"
#include "cudaUtils.h"



void blurOctave(float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream)
{
	float k = pow(2,0.5);
	// Get max Kernel size
	int maxKernelSize = imfilter::gaussianSize( pow( k, N_SCALES + 1 ) * SIGMA );
	// Get max apron Size
	int maxApronStart = floor( maxKernelSize / 2 );
	int maxApronEnd =  maxKernelSize - maxApronStart - 1;
	// Set bankOffset to 1 for even filter size
	int bankOffset = (maxApronStart == maxApronEnd) ? (1):(0);
	// Set x-convolution kernel parameters
	int nTilesX = 11;
	int nTilesY = 1;
	dim3 blockSize(WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1);
	dim3 gridSize(iDivUp(width, WIDTH_CONV_BLOCK*nTilesX), iDivUp(height, HEIGHT_CONV_BLOCK*nTilesY), 1);
	int sDimX = nTilesX*WIDTH_CONV_BLOCK + maxApronStart + maxApronEnd + bankOffset;
	int sDimY = HEIGHT_CONV_BLOCK;

	printf("sDimX \t%d\nsDimY \t%d\n", sDimX, sDimY);
	blurKernel<<<gridSize, blockSize, sDimX*sDimY*sizeof(float), stream>>>(dst, src
																		, width, pitch, height
																		, nTilesX, nTilesY
																		, maxApronStart, maxApronEnd, 0, 0
																		, bankOffset);
}
















void initDeviceConstant()
{


	// Set c_GaussianBlurSize[] for each scale
	// Set c_MaxGaussianBlurSize
	// Set c_GaussianBlur[] kernel for each scale
	float k = pow(2,0.5);
	int blurSize = 0;
	int maxBlurSize = 0;
	int blurSizeArray[N_SCALES + 3];

	int kernelStartPtr = 0;
	float gaussianBlur[B_KERNEL_SIZE];
	float sigma = 0.0;
	float sigmaOld = 0.0;
	float sigmaNew = 0.0;

	for(int i = 0; i < N_SCALES + 3; ++i)
	{
		sigma = pow( k, i-1 ) * SIGMA;
		sigmaNew  = sigma - sigmaOld;
		sigmaOld = sigma;
		// Push new kernel array to gaussiaBlur[]
		imfilter::gaussian1D( gaussianBlur + kernelStartPtr, sigmaNew );
		// Set blurSize to current kernel size
		blurSize = imfilter::gaussianSize( sigmaNew );
		// Push blurSize to blurSizeArray[]
		blurSizeArray[i] = blurSize;
		// Increment kernelStartPtr to point on top of gaussiaBlur[] stack
		kernelStartPtr += blurSize;
		// Set maxBlurSize
		maxBlurSize = (blurSize > maxBlurSize) ? (blurSize):(maxBlurSize);
	}
	// Copy symbols to constant memory
	cudaMemcpyToSymbol( c_GaussianBlurSize, &blurSizeArray, (N_SCALES + 3)*sizeof( int ) );
	cudaMemcpyToSymbol( c_MaxGaussianBlurSize, &maxBlurSize, sizeof( int ) );
	cudaMemcpyToSymbol( c_GaussianBlur, &gaussianBlur, B_KERNEL_SIZE * sizeof( float ) );
}

void testSetConstants(cudaStream_t &stream)
{
	kernelGaussianSize<<<1, 5, 0, stream>>>();
	kernelGaussianVector<<<1, B_KERNEL_SIZE, 0, stream>>>();
}

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
			h_data[i + j*p] = (i < w) ? ((i+1)*(i+1)): -1;
	}

	CUDA_SAFECALL( cudaMemcpy((void *)d_data, (void *)h_data, (size_t)(gx * sizeof(float)), cudaMemcpyHostToDevice ) );
	int nTilesX = 2;
	int nTilesY = 1;

	int apronLeft = 2;
	int apronRight = 5;
	int apronUp = 2;
	int apronDown = 5;
	dim3 blockDim(3,2,1);
	dim3 gridDim(1,1,1);
	int sx = apronLeft + apronRight + blockDim.x*nTilesX;
	int sy = apronUp + apronDown + blockDim.y*nTilesY;
	shKernel<<<gridDim, blockDim, sx*sy*sizeof( float ), stream>>>( d_data
																	, w, p, h
																	, nTilesX, nTilesY
																	, apronLeft, apronRight, apronUp, apronDown);

	free( h_data );
	CUDA_SAFECALL( cudaFree( d_data ));
}
