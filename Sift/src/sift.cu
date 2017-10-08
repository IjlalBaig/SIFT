#include "sift.h"
#include "utils.h"
#include "cudaUtils.h"
#include "cudaSiftH.h"
#include "cudaImage.h"

SiftData::SiftData():
	numPts( 0 ), maxPts( 0 ), h_data( NULL ), d_data( NULL ),
	d_internalAlloc( false ), h_internalAlloc( false )
{

}
SiftData::~SiftData()
{
	if(d_internalAlloc && d_data!=NULL)
		CUDA_SAFECALL( cudaFree( d_data ) );
	d_data = NULL;
	if(h_internalAlloc && h_data!=NULL)
		free( h_data );
	h_data = NULL;
}
void SiftData::Allocate(int max, SiftPoint *d_ptr, SiftPoint *h_ptr)
{
	numPts = 0;
	maxPts = max;
	d_data = d_ptr;
	h_data = h_ptr;
	if (d_ptr==NULL)
	{
		CUDA_SAFECALL( cudaMalloc( (void **)&d_data, (size_t)(maxPts * sizeof( SiftPoint )) ) );
		if (d_data==NULL)
			printf( "Failed to allocate Sift device data\n" );
		d_internalAlloc = true;
	}
	if (h_ptr==NULL)
	{
		h_data = (SiftPoint *)malloc( maxPts * sizeof( SiftPoint ) );
		if (h_data==NULL)
			printf( "Failed to allocate Sift host data\n" );
		h_internalAlloc = true;
	}
}
double SiftData::Upload(cudaStream_t stream)
{
	if (d_data!=NULL && h_data!=NULL)
		CUDA_SAFECALL( cudaMemcpyAsync( (void *)d_data, (const void *)h_data, maxPts * sizeof( SiftPoint ), cudaMemcpyHostToDevice, stream ) );
	return 0.0;
}
double SiftData::Readback(cudaStream_t stream)
{
	CUDA_SAFECALL( cudaMemcpyAsync( (void *)h_data, (const void *)d_data, maxPts * sizeof( SiftPoint ), cudaMemcpyDeviceToHost, stream ) );
	return 0.0;
}

int sift( std::string dstPath, std::string *srcPath, int nImgs)
{
//	cudaStream_t stream;
//	CUDA_SAFECALL( cudaStreamCreate(&stream) );
//	sharedKernel( stream );
//	CUDA_SAFECALL( cudaStreamDestroy(stream));

	//	Load image batch to Mat object
	cv::Mat matImg[BATCH_SIZE];
	int width[BATCH_SIZE];
	int height[BATCH_SIZE];
	for (int i = 0; i < BATCH_SIZE; ++i)
	{
		image::imload( matImg[i], srcPath[i], false );
		width[i] = matImg[i].cols;
		height[i] = matImg[i].rows;
	}

	//	Allocate Cuda Objects
	CudaImage cuImg[BATCH_SIZE];
	SiftData siftData[BATCH_SIZE];
	for (int i = 0; i < BATCH_SIZE; ++i)
	{
		cuImg[i].Allocate( width[i], height[i], NULL, (float *)matImg[i].data );
		siftData[i].Allocate(MAX_POINTCOUNT, NULL, NULL);
	}

	//	Create batch streams
	cudaStream_t stream[BATCH_SIZE];
	for (int i = 0; i < BATCH_SIZE; ++i)
		CUDA_SAFECALL( cudaStreamCreate( &stream[i] ) );

	// 	Set device constants
	initDeviceConstant();

	//	Set device variables
	initDeviceVariables();

	//	Execute sift on streams
	for (int i = 0; i < BATCH_SIZE; ++i)
	{
		//	Upload CudaImage to GPU
		cuImg[i].Upload(stream[i]);
		siftData[i].Upload(stream[i]);
		//	Launch Kernels
//		testcopyKernel(stream[i]);
//		testSetConstants(stream[i]);
		//	Allocate result pointer
		int resOctave = 0;
		int w = iDivUp( cuImg[i].width, pow(2, resOctave ) );
		int p = iAlignUp( w, 128);
		int h = iDivUp( cuImg[i].height, pow(2, resOctave ) );
		float *d_res;
		cv::Mat1f matRes(h, w);
		CUDA_SAFECALL( cudaMalloc( (void **) &d_res,(size_t)( p*h*sizeof( float ) ) ) );

		extractSift(siftData[i].d_data, d_res, resOctave, cuImg[i].d_data, cuImg[i].width, cuImg[i].pitch, cuImg[i].height, 0, stream[i], i );

		//	Copy data to result image
		CUDA_SAFECALL( cudaMemcpy2DAsync( (void *)matRes.data, (size_t)(w*sizeof( float )), (const void *)d_res, (size_t)(p* sizeof( float )), (size_t)(w*sizeof( float )), (size_t)h, cudaMemcpyDeviceToHost, stream[i]) );
//		image::imshow( matRes );
//			int gDim = cuRes.pitch*cuRes.height;
//		copyDeviceData(cuRes.d_data, 4*gDim + d_multiBlur[i], cuRes.width, cuRes.pitch, cuRes.height, stream[i] );
		//	Free pointers
		CUDA_SAFECALL( cudaFree( d_res ) );

	}

	for (int i = 0; i < BATCH_SIZE; ++i)
	{
		//	Download results to CPU
		cuImg[i].Readback(stream[i]);
		siftData[i].Readback(stream[i]);
	}

	//	Show result
	for (int i = 0; i < 1200; ++i)
		image::drawPoint( matImg[0], siftData[0].h_data[i].xpos, siftData[0].h_data[i].ypos, siftData[0].h_data[i].scale, siftData[0].h_data[i].orientation);
	image::imshow( matImg[0] );
	//	Destroy cuda streams for batchSize
	for (int i = 0; i < BATCH_SIZE; ++i)
		CUDA_SAFECALL( cudaStreamDestroy(stream[i]));
	return 0;
}


