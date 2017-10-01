#include "excludes/cudaSiftD.cu"
#include <stdio.h>
#include "utils.h"
#include "cudaUtils.h"
#include "sift.h"

void copyDeviceData( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream )
{
	dim3 blockSize( WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
	dim3 gridSize( iDivUp( width, WIDTH_CONV_BLOCK ), iDivUp( height, HEIGHT_CONV_BLOCK ), 1 );
	copyKernel<<<gridSize, blockSize, 0, stream>>>( dst, src, width, pitch, height );
}

void computeExtrema()
{
//	// Get max apron Size
//	int maxApron = 1;
//	// Set bankOffset so sDimX is odd and no bank conlicts
//	int bankOffset = 1;
//	// Set hessian kernel parameters
//	int nTilesX = 11 ;
//	int nTilesY = 1;
//	dim3 blockSize( WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
//	dim3 gridSize( iDivUp( width, WIDTH_CONV_BLOCK*nTilesX ), iDivUp( height, HEIGHT_CONV_BLOCK*nTilesY ), 1 );
//	int sDimX = nTilesX*WIDTH_CONV_BLOCK + 2*maxApron + bankOffset;
//	int sDimY = nTilesY*HEIGHT_CONV_BLOCK + 2*maxApron;
//	int gDim = pitch*height;
}

void computeDiffOfGauss( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream )
{

	// Set DoG kernel parameters
	dim3 blockSize(WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
	dim3 gridSize( iDivUp( width, WIDTH_CONV_BLOCK ), iDivUp( height, HEIGHT_CONV_BLOCK ), 1 );
	int gDim = pitch*height;

	//	Launch subtract kernel
		for (int i = 0; i < N_SCALES + 2; ++i)
			subtractKernel<<<gridSize, blockSize>>>(dst + i*gDim, src + i*gDim, src + (i + 1)*gDim, width, pitch, height);
}

void computeGradient( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream )
{
	// Get max apron Size
	int maxApron = 1;
	// Set bankOffset so sDimX is odd and no bank conlicts
	int bankOffset = 3;
	// Set hessian kernel parameters
	int nTilesX = 11 ;
	int nTilesY = 1;
	dim3 blockSize( WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
	dim3 gridSize( iDivUp( width, WIDTH_CONV_BLOCK*nTilesX ), iDivUp( height, HEIGHT_CONV_BLOCK*nTilesY ), 1 );
	int sDimX = nTilesX*WIDTH_CONV_BLOCK + 2*maxApron + bankOffset;
	int sDimY = nTilesY*HEIGHT_CONV_BLOCK + 2*maxApron;
	int gDim = pitch*height;
	int nUnitsPerGradient = 2;
	for (int i = 0; i < N_SCALES; ++i)
		gradientKernel<<<gridSize, blockSize, sDimX*sDimY*sizeof( float ), stream>>>( dst + i*gDim*nUnitsPerGradient, src + (i + 1)*gDim
																					, width, pitch, height
																					, nTilesX, nTilesY
																					, maxApron, maxApron, maxApron, maxApron
																					, bankOffset );
}

void computeHessian( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream )
{
	// Get max apron Size
	int maxApron = 1;
	// Set bankOffset so sDimX is odd and no bank conlicts
	int bankOffset = 1;
	// Set hessian kernel parameters
	int nTilesX = 11 ;
	int nTilesY = 1;
	dim3 blockSize( WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
	dim3 gridSize( iDivUp( width, WIDTH_CONV_BLOCK*nTilesX ), iDivUp( height, HEIGHT_CONV_BLOCK*nTilesY ), 1 );
	int sDimX = nTilesX*WIDTH_CONV_BLOCK + 2*maxApron + bankOffset;
	int sDimY = nTilesY*HEIGHT_CONV_BLOCK + 2*maxApron;
	int gDim = pitch*height;
	int nUnitsPerHessian = 3;
	for (int i = 0; i < N_SCALES; ++i)
		hessianKernel<<<gridSize, blockSize, sDimX*sDimY*sizeof( float ), stream>>>( dst + i*gDim*nUnitsPerHessian, src + (i + 1)*gDim
																					, width, pitch, height
																					, nTilesX, nTilesY
																					, maxApron, maxApron, maxApron, maxApron
																					, bankOffset );
}


void blurOctave( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream )
{
	float k = pow( 2, 0.5 );
	int nBlurScales = N_SCALES + 3;
	int gDim = pitch * height;

	// 	Allocate tmp data pointer
	float *tmpDst;
	CUDA_SAFECALL( cudaMalloc( (void **) &tmpDst,(size_t)(pitch * height * nBlurScales * sizeof( float )) ) );

	// Get max apron Size
	int maxApronStart = B_KERNEL_RADIUS;
	int maxApronEnd = B_KERNEL_RADIUS;
	// Set bankOffset so sDimX is odd and no bank conlicts
	int bankOffset = 1;
	// Set x-convolution kernel parameters
	int nTilesX = 11 ;//- iDivUp((maxApronStart + maxApronEnd), WIDTH_CONV_BLOCK) ;
	int nTilesY = 1;
	dim3 blockSizeX( WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
	dim3 gridSizeX( iDivUp( width, WIDTH_CONV_BLOCK*nTilesX ), iDivUp( height, HEIGHT_CONV_BLOCK*nTilesY ), 1 );
	int sDimX = nTilesX*WIDTH_CONV_BLOCK + maxApronStart + maxApronEnd + bankOffset;
	int sDimY = nTilesY*HEIGHT_CONV_BLOCK;
	xblurMultiKernel<<<gridSizeX, blockSizeX, sDimX*sDimY*sizeof( float ), stream>>>( tmpDst, src
																					, width, pitch, height
																					, nTilesX, nTilesY
																					, maxApronStart, maxApronEnd, 0, 0
																					, bankOffset );
	// Set y-convolution kernel parameters
	nTilesX = 1;
	nTilesY = 11;
	dim3 blockSizeY( WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
	dim3 gridSizeY( iDivUp( width, WIDTH_CONV_BLOCK*nTilesX ), iDivUp( height, HEIGHT_CONV_BLOCK*nTilesY ), 1 );
	sDimX = nTilesX*WIDTH_CONV_BLOCK + bankOffset;
	sDimY = nTilesY*HEIGHT_CONV_BLOCK + maxApronStart + maxApronEnd;

	for (int i = 0; i < nBlurScales; ++i)
	{
		yblurKernel<<<gridSizeY, blockSizeY, sDimX*sDimY*sizeof( float ), stream>>>( dst + i*gDim, tmpDst + i*gDim, i
																					, width, pitch, height
																					, nTilesX, nTilesY
																					, 0, 0, maxApronStart, maxApronEnd
																					, bankOffset );
	}
	// 	Free tmp data
	CUDA_SAFECALL( cudaFree( tmpDst) );
}


void allocateOctave( float *&multiBlur, float *&multiDoG
					, float *&multiHessian, float *&multiGradient
					, int w, int p, int h )
{
	int nBlurScales = N_SCALES + 3;
	int nDoGScales = N_SCALES + 2;
	CUDA_SAFECALL( cudaMalloc( (void **) &multiBlur,(size_t)(p * h * nBlurScales * sizeof( float )) ) );
	CUDA_SAFECALL( cudaMalloc( (void **) &multiDoG,(size_t)(p * h * nDoGScales  * sizeof( float )) ) );
	CUDA_SAFECALL( cudaMalloc( (void **) &(multiHessian),(size_t)(p * h * N_SCALES * sizeof( float )) ) );
	CUDA_SAFECALL( cudaMalloc( (void **) &multiGradient,(size_t)(p * h * 2 *N_SCALES * sizeof( float )) ) );


//	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
//	safeCall( cudaMallocArray( &m, &channelDesc, w, h ) );
//	safeCall( cudaMallocArray( &theta, &channelDesc, w, h ) );
}

void freeOctave( float *&multiBlur, float *&multiDoG
				, float *&multiHessian, float *&multiGradient )
{
//	Free device memory
	CUDA_SAFECALL( cudaFree( multiBlur ) );
	CUDA_SAFECALL( cudaFree( multiDoG ) );
	CUDA_SAFECALL( cudaFree( multiHessian ) );
	CUDA_SAFECALL( cudaFree( multiGradient ) );

//	Free texture memory
	//safeCall( cudaDestroyTextureObject( texObj ) );
//	safeCall( cudaFreeArray( m ) );
//	safeCall( cudaFreeArray( theta ) );
}

int getPointCount(int streamIdx)
{
	int h_pointCounter[BATCH_SIZE];
	CUDA_SAFECALL( cudaMemcpyFromSymbol( &h_pointCounter, d_PointCounter, sizeof( int ) ) );
	return h_pointCounter[streamIdx];
}

void extractSift( float *d_res, int resOctave, float *d_src, int width, int pitch, int height, int octaveIdx, cudaStream_t &stream, int streamIdx )
{
	int ptCount = getPointCount(streamIdx);
	if (ptCount < MAX_POINTCOUNT)
	{
		//	Allocate octave pointers
		float *d_multiBlur;
		float *d_multiDoG;
		float *d_multiHessian;
		float *d_multiGradient;

		// 	Allocate Octave
		allocateOctave( d_multiBlur, d_multiDoG
					, d_multiHessian, d_multiGradient
					, width, pitch, height );
		//	Compute octave scale space
		blurOctave( d_multiBlur, d_src, width, pitch, height, stream );

		//	Compute difference of gaussian
		computeDiffOfGauss( d_multiDoG, d_multiBlur, width, pitch, height, stream );

		//	Compute octave hessian
		computeHessian( d_multiHessian, d_multiBlur, width, pitch, height, stream );

		//	Compute octave gradient
		computeGradient( d_multiGradient, d_multiBlur, width, pitch, height, stream );

		//	Copy back result
		if (resOctave == octaveIdx)
		{
			int scaleIdx = 0;
			copyDeviceData(d_res, scaleIdx*pitch*height + d_multiGradient, width, pitch, height, stream );
		}
		//	Check if numPts == maxPts
		octaveIdx += 1;
		if (octaveIdx < N_OCTAVES /*&& numPts < maxPts*/)
		{
			float *d_src_;
			int width_ = iDivUp(width, 2);
			int height_ = iDivUp(height, 2);
			int pitch_ = iAlignUp(width_, 128);
			CUDA_SAFECALL( cudaMalloc( (void **) &d_src_,(size_t)(pitch_ * height_ * sizeof( float )) ) );

			//	Set resize kernel parameters
			dim3 blockSize( WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
			dim3 gridSize( iDivUp( width, WIDTH_CONV_BLOCK ), iDivUp( height, HEIGHT_CONV_BLOCK ), 1 );
			resizeKernel<<<gridSize, blockSize, 0, stream>>>( d_src_, d_multiBlur + N_SCALES*pitch*height, width, pitch, height );

			extractSift( d_res, resOctave, d_src_, width_, pitch_, height_, octaveIdx, stream, streamIdx );
			CUDA_SAFECALL( cudaFree( d_src_ ) );
		}
		//	Free octave
		freeOctave(d_multiBlur, d_multiDoG, d_multiHessian, d_multiGradient );
	}
}


void initDeviceVariables()
{
	//	Allocate d_PointCounter Variables
	int h_pointCounter[BATCH_SIZE];
	for (int i = 0; i < BATCH_SIZE; ++i)
		h_pointCounter[i] = 0;
	CUDA_SAFECALL( cudaMemcpyToSymbol( d_PointCounter, &h_pointCounter, BATCH_SIZE*sizeof( int ) ) );
}

void initDeviceConstant()
{
	// 	Set c_GaussianBlur[] kernel for each scale
	float k = pow(2,0.5);
	float gaussianBlur[(N_SCALES + 3) * B_KERNEL_SIZE];
	float sigmaNew = 0.0;

	for(int i = 0; i < N_SCALES + 3; ++i)
	{
		sigmaNew = pow( k, i-1 ) * SIGMA;

		// 	Push new kernel array to gaussiaBlur[]
		imfilter::gaussian1D( gaussianBlur + i * (2*B_KERNEL_RADIUS + 1), sigmaNew );
	}
	// 	Copy gaussian kernel to constant memory
	CUDA_SAFECALL( cudaMemcpyToSymbol( c_GaussianBlur, &gaussianBlur, (N_SCALES + 3) * B_KERNEL_SIZE * sizeof( float ) ) );

	//	Copy extrema threshold to constant memory
	float extremaThreshold = EXTREMA_THRESH;
	CUDA_SAFECALL( cudaMemcpyToSymbol( c_ExtremaThreshold, &extremaThreshold, sizeof( float ) ) );

	//	Copy eigenvalue ratio threshold to constant memory
	float eigenRatio = pow( R_THRESH + 1, 2 ) / 2;
	CUDA_SAFECALL( cudaMemcpyToSymbol( c_EdgeThreshold, &eigenRatio, sizeof( float ) ) );

}

//void testSetConstants(cudaStream_t &stream)
//{
//	kernelGaussianSize<<<1, 5, 0, stream>>>();
//	kernelGaussianVector<<<1, (N_SCALES + 3) * B_KERNEL_SIZE, 0, stream>>>();
//}

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
