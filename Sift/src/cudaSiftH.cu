#include "excludes/cudaSiftD.cu"
#include <stdio.h>
#include "utils.h"
#include "cudaUtils.h"
#include "sift.h"

int updatePtStartIdx( int streamIdx )
{
	unsigned int *d_ptrStartIdx;
	unsigned int *d_ptrPointCount;
	CUDA_SAFECALL( cudaGetSymbolAddress( (void **) &d_ptrStartIdx, d_PointStartIdx ) );
	CUDA_SAFECALL( cudaGetSymbolAddress( (void **) &d_ptrPointCount, d_PointCount ) );
	CUDA_SAFECALL( cudaMemcpy(&d_ptrStartIdx[streamIdx], &d_ptrPointCount[streamIdx], sizeof( unsigned int ), cudaMemcpyDeviceToDevice ) );
	int result = -1;
	CUDA_SAFECALL( cudaMemcpy(&result, &d_ptrPointCount[streamIdx], sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );
	printf("result\t%d\n", result);
	return result;
}

int getPointCount(int streamIdx)
{
	int h_pointCount[BATCH_SIZE];
	CUDA_SAFECALL( cudaMemcpyFromSymbol( &h_pointCount, d_PointCount, sizeof( int ) ) );
	return h_pointCount[streamIdx];
}

void copyDeviceData( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream )
{
	dim3 blockSize( WIDTH_CONV_BLOCK, HEIGHT_CONV_BLOCK, 1 );
	dim3 gridSize( iDivUp( width, WIDTH_CONV_BLOCK ), iDivUp( height, HEIGHT_CONV_BLOCK ), 1 );
	copyKernel<<<gridSize, blockSize, 0, stream>>>( dst, src, width, pitch, height );
}

void computeOctaveSift( SiftPoint *pt, float *src_DoG, float *src_Hessian, float *src_Gradient, int width, int pitch, int height, cudaStream_t &stream, int streamIdx )
{

	/*
	 *
	 *
	 *
	 */
	//	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
	//	safeCall( cudaMallocArray( &m, &channelDesc, w, h ) );
	//	safeCall( cudaMallocArray( &theta, &channelDesc, w, h ) );
	/*
	 *
	 *
	 *
	 */


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
		findExtremaKernel<<<gridSizeXtr, blockSizeXtr, sDimX*sDimY*sizeof( float ), stream>>>( pt, src_DoG + i*gDim, src_Hessian + i*nUnitsPerHessian*gDim
																					, width, pitch, height
																					, nTilesX, nTilesY
																					, maxApron, maxApron, maxApron, maxApron
																					, bankOffset, streamIdx );
		scalePointCount =  getPointCount( streamIdx ) - ptCountStart;
		dim3 blockSizeOrient(16,16,1);
		dim3 gridSizeOrient( iDivUp(scalePointCount, ORIENT_BUFFER ), 1, 1 );
		OrientationKernel<<<gridSizeOrient, blockSizeOrient, 0, stream>>>( pt, src_Gradient + i*nUnitsPerGradient*gDim
																		, width, pitch, height
																		, scalePointCount, i, streamIdx);
		scalePointCount =  getPointCount( streamIdx ) - ptCountStart;
//		DescriptorKernel<<<gridSizeOrient, blockSizeOrient, 0, stream>>>(pt, src_Gradient_tex + i*nUnitsPerGradient*gDim
//																		, width, pitch, height
//																		, scalePointCount, i, streamIdx);
	}
}

void computeDiffOfGauss( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream )
{

	// 	Set DoG kernel parameters
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
	int nUnitsPerHessian = 3;
	CUDA_SAFECALL( cudaMalloc( (void **) &multiBlur,(size_t)(p * h * nBlurScales * sizeof( float )) ) );
	CUDA_SAFECALL( cudaMalloc( (void **) &multiDoG,(size_t)(p * h * nDoGScales  * sizeof( float )) ) );
	CUDA_SAFECALL( cudaMalloc( (void **) &(multiHessian),(size_t)(p * h * nUnitsPerHessian * N_SCALES * sizeof( float )) ) );
	CUDA_SAFECALL( cudaMalloc( (void **) &multiGradient,(size_t)(p * h * 2 * N_SCALES * sizeof( float )) ) );



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

void testWarpMax( cudaStream_t &stream )
{
	//	Define pointers
	float *h_data;
	float *d_data;

	//	Allocate memory
	h_data = (float *)malloc( 64*sizeof( float ) );
	CUDA_SAFECALL( cudaMalloc( (void **)&d_data, (size_t)(64*sizeof( float ))) );

	//	Set h_data
	for (int i = 0; i < 64; ++i)
	{
		h_data[i] = rand() % 32;
		printf("random value is \t%f\n ", h_data[i]);
	}

	//	Copy to d_data
	cudaMemcpy( d_data, h_data, 64*sizeof ( float ), cudaMemcpyHostToDevice );

	warpMaxKernel<<<1 , 64, 0, stream>>>(d_data);

	//	Readback results
	cudaMemcpy( h_data, d_data, 64*sizeof ( float ), cudaMemcpyDeviceToHost );

	//	Display results
	printf("max value is \t%f\n ", h_data[0]);
	printf("max value is \t%f\n ", h_data[1]);
	printf("max value is \t%f\n ", h_data[2]);
	printf("max value is \t%f\n ", h_data[3]);
	printf("max value is \t%f\n ", h_data[4]);
	printf("max value is \t%f\n ", h_data[16]);
	printf("max value is \t%f\n ", h_data[32]);
}

void linear2cuArray(cudaArray *cuArray, float *src, cudaTextureObject_t &texObj, int w, int p, int h)
{
// 	Copy d_data to cuArray
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


void extractSift(SiftPoint *siftPt, float *d_res, int resOctave, float *d_src, int width, int pitch, int height, int octaveIdx, cudaStream_t &stream, int streamIdx )
{
	int ptCount = getPointCount(streamIdx);
	printf("%d\n", ptCount);
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

		/*
		 *
		 *
		 *
		 */
		cudaArray *cuArray;
		// 	Allocate CUDA array in device memory
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

		CUDA_SAFECALL(cudaMallocArray(&cuArray, &channelDesc, width, height));

		cudaTextureObject_t texObj;
		linear2cuArray(cuArray, d_multiBlur, texObj, width, pitch, height);
		// 	Invoke kernel
		dim3 dimBlock(32, 32);
		dim3 dimGrid(iDivUp(pitch, 32), iDivUp(height, 32), 1);
		transformKernel<<<dimGrid, dimBlock>>>(d_multiGradient, texObj, width, pitch, height, 45);
		// Destroy texture object
		CUDA_SAFECALL(cudaDestroyTextureObject(texObj));
		// Free device memory
		CUDA_SAFECALL(cudaFreeArray(cuArray));
		/*
		 *
		 *
		 *
		 */
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

			extractSift( siftPt, d_res, resOctave, d_src_, width_, pitch_, height_, octaveIdx, stream, streamIdx );
			CUDA_SAFECALL( cudaFree( d_src_ ) );
		}

		//	Free octave
		freeOctave(d_multiBlur, d_multiDoG, d_multiHessian, d_multiGradient );
	}
}

void initDeviceVariables()
{
	//	Allocate d_globalPtCount Variables
	int h_PointCount[BATCH_SIZE];
	for (int i = 0; i < BATCH_SIZE; ++i)
		h_PointCount[i] = 0;
	CUDA_SAFECALL( cudaMemcpyToSymbol( d_PointCount, &h_PointCount, BATCH_SIZE*sizeof( int ) ) );

	//	Allocate d_scalePtCount Variables
//	int h_sPtCount[BATCH_SIZE];
//	for (int i = 0; i < BATCH_SIZE; ++i)
//		h_sPtCount[i] = 0;
//	CUDA_SAFECALL( cudaMemcpyToSymbol( d_scalePtCount, &h_sPtCount, BATCH_SIZE*sizeof( int ) ) );
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
		imfilter::gaussian1D( gaussianBlur + i * (2*B_KERNEL_RADIUS + 1), sigmaNew, 2*B_KERNEL_RADIUS + 1 );
	}
	// 	Copy gaussian kernel to constant memory
	CUDA_SAFECALL( cudaMemcpyToSymbol( c_GaussianBlur, &gaussianBlur, (N_SCALES + 3) * B_KERNEL_SIZE * sizeof( float ) ) );

	// 	Set c_GaussianWnd[] Mask for each scale
	float gaussianWnd[(N_SCALES) * WND_KERNEL_SIZE];

	for(int i = 0; i < N_SCALES; ++i)
	{
		sigmaNew = pow( k, i ) * SIGMA * 1.5;

		// 	Push new kernel array to gaussiaWnd[]
		imfilter::gaussian1D( gaussianWnd + i * WND_KERNEL_SIZE, sigmaNew, WND_KERNEL_SIZE);
	}
	// 	Copy gaussian kernel to constant memory
	CUDA_SAFECALL( cudaMemcpyToSymbol( c_GaussianWnd, &gaussianWnd, N_SCALES * WND_KERNEL_SIZE * sizeof( float ) ) );

	//	Copy extrema threshold to constant memory
	float extremaThreshold = EXTREMA_THRESH;
	CUDA_SAFECALL( cudaMemcpyToSymbol( c_ExtremaThreshold, &extremaThreshold, sizeof( float ) ) );

	//	Copy eigenvalue ratio threshold to constant memory
	float eigenRatio = pow( R_THRESH + 1, 2 ) / 2;
	CUDA_SAFECALL( cudaMemcpyToSymbol( c_EdgeThreshold, &eigenRatio, sizeof( float ) ) );

	//	Copy max point count to constant memory
	unsigned int maxPointCount = MAX_POINTCOUNT;
	CUDA_SAFECALL( cudaMemcpyToSymbol( c_MaxPointCount, &maxPointCount, sizeof( unsigned int ) ) );

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
