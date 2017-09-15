#include "sift.h"
#include "utils.h"
#include "cudaUtils.h"
#include "cudaSiftH.h"


int sift( char *dstPath, char *srcPath ){
	/* Load image */
	cv::Mat imgMat;
	image::imload( imgMat, srcPath, false );

	/* Define host and device data */
	float *h_data;
	float *d_data;
	int width = imgMat.cols;
	int height = imgMat.rows;
	int pitch;

	/* Allocate host and device Memory */
	h_data = (float *)malloc( width*height*sizeof( float ) );
	CUDA_SAFECALL( cudaMallocPitch( (void **)&d_data, (size_t *)(&pitch), (size_t)(width*sizeof(float)), (size_t)(height) ) );
	pitch/=(sizeof(int));

	/* Set host and device data */
	h_data = (float *)imgMat.data;
	CUDA_SAFECALL( cudaMemcpy2D( (void *) d_data, (size_t)(pitch*sizeof(float)), (const void *) h_data, (size_t)(width*sizeof(float)), (size_t)(width*sizeof(float)), (size_t)height, cudaMemcpyHostToDevice ) );


	/* Create stream */
	cudaStream_t stream1;
	CUDA_SAFECALL( cudaStreamCreate( &stream1 ) );

	/* Compute sift */
//		exclusiveSift();
	/* Show result */
	image::imshow( imgMat );

	/* Save result */

	/* Free resources (stream, h_pointer, d_pointer) */
	CUDA_SAFECALL( cudaStreamDestroy( stream1 ) );
	CUDA_SAFECALL( cudaFree( d_data ) );
	return 0;
}
