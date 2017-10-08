#include "excludes/cudaSiftD.cu"
#include <stdio.h>
#include "cudaSiftH.h"
#include "utils.h"
#include "cudaUtils.h"
#include "sift.h"


void extractSift(SiftPoint *siftPt, float *d_res, float *d_src, int width, int pitch, int height, cudaStream_t &stream, int streamIdx, int octaveIdx )
{

	int ptCount = getPointCount(streamIdx);
	/*
	 *
	 */printf("%d\n", ptCount);
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

		//	Compute octave SiftData
		computeOctaveSift( siftPt, d_multiDoG, d_multiHessian, d_multiGradient, width, pitch, height, stream, streamIdx );

		//	Copy back result
		/*
		 *
		 */
		if (RESULT_OCTAVE == octaveIdx)
		{
			int scaleIdx = 0;
			copyDeviceData(d_res, scaleIdx*pitch*height + d_multiGradient, width, pitch, height, stream );
		}
		/*
		 *
		 */
		octaveIdx += 1;
		if (octaveIdx < N_OCTAVES)
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
			extractSift( siftPt, d_res, d_src_, width_, pitch_, height_, stream, streamIdx, octaveIdx );
			CUDA_SAFECALL( cudaFree( d_src_ ) );
		}

		//	Free octave
		freeOctave(d_multiBlur, d_multiDoG, d_multiHessian, d_multiGradient );
	}
}
void computeOctaveSift( SiftPoint *pt, float *src_DoG, float *src_Hessian, float *src_Gradient, int width, int pitch, int height, cudaStream_t &stream, int streamIdx )
{
	// 	Get max apron Size
	int maxApron = 1;
	// 	Set bankOffset so sDimX is odd and no bank conflicts occur
	int bankOffset = 3;
	// 	Set hessian kernel parameters
	int nTilesX = 3;
	int nTilesY = 1;
	int nScales = 3;
	dim3 blockSizeXtr( WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
	dim3 gridSizeXtr( iDivUp( width, WIDTH_CONV_BLOCK*nTilesX ), iDivUp( height, HEIGHT_CONV_BLOCK*nTilesY ), 1 );
	int sDimX = nScales*nTilesX*WIDTH_CONV_BLOCK + 2*maxApron + bankOffset;
	int sDimY = nTilesY*HEIGHT_CONV_BLOCK + 2*maxApron;
	int gDim = pitch*height;
	int nUnitsPerHessian = 3;
	int nUnitsPerGradient = 2;
	int ptCountStart = 0;
	int scalePointCount = 0;
	for (int i = 0; i < N_SCALES; ++i)
	{
		ptCountStart = updatePtStartIdx( streamIdx );
		findExtremaKernel<<< gridSizeXtr, blockSizeXtr, sDimX*sDimY*sizeof( float ), stream >>>( pt, src_DoG + i*gDim, src_Hessian + i*nUnitsPerHessian*gDim
																					, width, pitch, height
																					, nTilesX, nTilesY
																					, maxApron, maxApron, maxApron, maxApron
																					, bankOffset, streamIdx );
		scalePointCount =  getPointCount( streamIdx ) - ptCountStart;
		dim3 blockSizeOrient(16,16,1);
		dim3 gridSizeOrient( iDivUp(scalePointCount, ORIENT_BUFFER ), 1, 1 );
		OrientationKernel<<< gridSizeOrient, blockSizeOrient, 0, stream >>>( pt, src_Gradient + i*nUnitsPerGradient*gDim
																		, width, pitch, height
																		, scalePointCount, i, streamIdx);
		scalePointCount =  getPointCount( streamIdx ) - ptCountStart;

		cudaArray *cuArrayMag;
		cudaArray *cuArrayDir;
		cudaTextureObject_t texObjMag;
		cudaTextureObject_t texObjDir;

		setTexture( cuArrayMag, texObjMag, src_Gradient + i*nUnitsPerGradient*gDim, width, pitch, height );
		setTexture( cuArrayDir, texObjDir, src_Gradient + (i*nUnitsPerGradient + 1)*gDim, width, pitch, height );


		DescriptorKernel<<< gridSizeOrient, blockSizeOrient, 0, stream >>>(pt, texObjMag, texObjDir
																		, width, pitch, height
																		, scalePointCount, i, streamIdx);
		freeTexture(cuArrayMag, texObjMag);
		freeTexture(cuArrayDir, texObjDir);
	}
}


void initDeviceVariables()
{
	//	Reset point count
	int h_PointCount[2];
	for (int i = 0; i < 2; ++i)
		h_PointCount[i] = 0;
	CUDA_SAFECALL( cudaMemcpyToSymbol( d_PointCount, &h_PointCount, 2*sizeof( int ) ) );
}
void initDeviceConstant()
{
	//	Set blur kernel
	float k = pow(2,0.5);
	float gaussianBlur[(N_SCALES + 3) * B_KERNEL_SIZE];
	float sigmaNew = 0.0;

	for(int i = 0; i < N_SCALES + 3; ++i)
	{
		sigmaNew = pow( k, i-1 ) * SIGMA;
		imfilter::gaussian1D( gaussianBlur + i * (2*B_KERNEL_RADIUS + 1), sigmaNew, 2*B_KERNEL_RADIUS + 1 );
	}
	CUDA_SAFECALL( cudaMemcpyToSymbol( c_GaussianBlur, &gaussianBlur, (N_SCALES + 3) * B_KERNEL_SIZE * sizeof( float ) ) );

	//	Set window mask
	float gaussianWnd[(N_SCALES) * WND_KERNEL_SIZE];

	for(int i = 0; i < N_SCALES; ++i)
	{
		sigmaNew = pow( k, i ) * SIGMA * 1.5;
		imfilter::gaussian1D( gaussianWnd + i * WND_KERNEL_SIZE, sigmaNew, WND_KERNEL_SIZE);
	}
	CUDA_SAFECALL( cudaMemcpyToSymbol( c_GaussianWnd, &gaussianWnd, N_SCALES * WND_KERNEL_SIZE * sizeof( float ) ) );
}
void allocateOctave( float *&multiBlur, float *&multiDoG
		, float *&multiHessian, float *&multiGradient
		, int w, int p, int h )
{
	int nBlurScales = N_SCALES + 3;
	int nDoGScales = N_SCALES + 2;
	int nUnitsPerHessian = 3;
	CUDA_SAFECALL( cudaMalloc( (void **) &multiBlur,(size_t)(p * h * nBlurScales * sizeof( float )) ) );
	CUDA_SAFECALL( cudaMalloc( (void **) &multiDoG,(size_t)(p * h * nDoGScales  * sizeof( float )) ) );
	CUDA_SAFECALL( cudaMalloc( (void **) &(multiHessian),(size_t)(p * h * nUnitsPerHessian * N_SCALES * sizeof( float )) ) );
	CUDA_SAFECALL( cudaMalloc( (void **) &multiGradient,(size_t)(p * h * 2 * N_SCALES * sizeof( float )) ) );
}
void freeOctave( float *&multiBlur, float *&multiDoG
		, float *&multiHessian, float *&multiGradient )
{
	CUDA_SAFECALL( cudaFree( multiBlur ) );
	CUDA_SAFECALL( cudaFree( multiDoG ) );
	CUDA_SAFECALL( cudaFree( multiHessian ) );
	CUDA_SAFECALL( cudaFree( multiGradient ) );
}
void setTexture(cudaArray *&cuArray, cudaTextureObject_t &texObj , float *src, int w, int p, int h)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	CUDA_SAFECALL(cudaMallocArray(&cuArray, &channelDesc, w, h));
// 	Copy d_data to cuArray
	for (int i = 0; i < N_SCALES; ++i)
		cudaMemcpy2DToArray(cuArray, 0, 0, src, p*sizeof(float), w*sizeof(float), h, cudaMemcpyDeviceToDevice);

// 	Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

// 	Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

// 	Create texture object
	texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
}
void freeTexture(cudaArray *&cuArray, cudaTextureObject_t &texObj)
{
	CUDA_SAFECALL(cudaDestroyTextureObject(texObj));
	CUDA_SAFECALL(cudaFreeArray(cuArray));
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

	// Set bankOffset so sDimX is odd and no bank conflicts
	int bankOffset = 1;

	// Set x-convolution kernel parameters
	int nTilesX = 11 ;
	int nTilesY = 1;
	dim3 blockSizeX( WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
	dim3 gridSizeX( iDivUp( width, WIDTH_CONV_BLOCK*nTilesX ), iDivUp( height, HEIGHT_CONV_BLOCK*nTilesY ), 1 );
	int sDimX = nTilesX*WIDTH_CONV_BLOCK + maxApronStart + maxApronEnd + bankOffset;
	int sDimY = nTilesY*HEIGHT_CONV_BLOCK;
	xblurMultiKernel<<< gridSizeX, blockSizeX, sDimX*sDimY*sizeof( float ), stream >>>( tmpDst, src
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
		yblurKernel<<< gridSizeY, blockSizeY, sDimX*sDimY*sizeof( float ), stream >>>( dst + i*gDim, tmpDst + i*gDim, i
																					, width, pitch, height
																					, nTilesX, nTilesY
																					, 0, 0, maxApronStart, maxApronEnd
																					, bankOffset );
	}

	// 	Free tmp data
	CUDA_SAFECALL( cudaFree( tmpDst) );
}
void computeHessian( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream )
{
	// Get max apron Size
	int maxApron = 1;

	// Set bankOffset so sDimX is odd and no bank conflicts
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
		hessianKernel<<< gridSize, blockSize, sDimX*sDimY*sizeof( float ), stream >>>( dst + i*gDim*nUnitsPerHessian, src + (i + 1)*gDim
																					, width, pitch, height
																					, nTilesX, nTilesY
																					, maxApron, maxApron, maxApron, maxApron
																					, bankOffset );
}
void computeGradient( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream )
{
	// Get max apron Size
	int maxApron = 1;
	// Set bankOffset so sDimX is odd and no bank conflicts
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
void computeDiffOfGauss( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream )
{
	// 	Set DoG kernel parameters
	dim3 blockSize(WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
	dim3 gridSize( iDivUp( width, WIDTH_CONV_BLOCK ), iDivUp( height, HEIGHT_CONV_BLOCK ), 1 );
	int gDim = pitch*height;

	//	Launch subtract kernel
		for (int i = 0; i < N_SCALES + 2; ++i)
			subtractKernel<<< gridSize, blockSize, 0, stream >>>(dst + i*gDim, src + i*gDim, src + (i + 1)*gDim, width, pitch, height);
}
void copyDeviceData( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream )
{
	dim3 blockSize( WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
	dim3 gridSize( iDivUp( width, WIDTH_CONV_BLOCK ), iDivUp( height, HEIGHT_CONV_BLOCK ), 1 );
	copyKernel<<< gridSize, blockSize, 0, stream >>>( dst, src, width, pitch, height );
}
int getPointCount( int streamIdx )
{
	int h_pointCount[2];
	CUDA_SAFECALL( cudaMemcpyFromSymbol( &h_pointCount, d_PointCount, 2*sizeof( int ) ) );
	return h_pointCount[streamIdx];
}
int updatePtStartIdx( int streamIdx )
{
	unsigned int *d_ptrStartIdx;
	unsigned int *d_ptrPointCount;
	CUDA_SAFECALL( cudaGetSymbolAddress( (void **) &d_ptrStartIdx, d_PointStartIdx ) );
	CUDA_SAFECALL( cudaGetSymbolAddress( (void **) &d_ptrPointCount, d_PointCount ) );
	CUDA_SAFECALL( cudaMemcpy(&d_ptrStartIdx[streamIdx], &d_ptrPointCount[streamIdx], sizeof( unsigned int ), cudaMemcpyDeviceToDevice ) );
	int result = -1;
	CUDA_SAFECALL( cudaMemcpy(&result, &d_ptrPointCount[streamIdx], sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );
	return result;
}
