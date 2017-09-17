#include "sift.h"
#include "utils.h"
#include "cudaUtils.h"
#include "cudaSiftH.h"
#include "cudaImage.h"


int sift( std::string dstPath, std::string *srcPath)
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
		siftData[i].Allocate(MAX_POINTS, NULL, NULL);
	}

	//	Create batch streams
	cudaStream_t stream[BATCH_SIZE];
	for (int i = 0; i < BATCH_SIZE; ++i)
		CUDA_SAFECALL( cudaStreamCreate( &stream[i] ) );

	//	Execute sift on streams
	for (int i = 0; i < BATCH_SIZE; ++i)
	{
		//	Upload CudaImage to GPU
		cuImg[i].Upload(stream[i]);
		siftData[i].Upload(stream[i]);

		//	Launch Kernels
		testcopyKernel(stream[i]);
	}

	for (int i = 0; i < BATCH_SIZE; ++i)
	{
		//	Download results to CPU
		cuImg[i].Readback(stream[i]);
		siftData[i].Readback(stream[i]);
	}

	//	Show result
	image::imshow( matImg[0] );

	//	Destroy cuda streams for batchSize
	for (int i = 0; i < BATCH_SIZE; ++i)
		CUDA_SAFECALL( cudaStreamDestroy(stream[i]));
	return 0;
}

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
