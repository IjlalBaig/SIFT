#include <stdio.h>

#include "cudaImage.h"

#include "cudaUtils.h"
#include "utils.h"


CudaImage::CudaImage(): width( 0 ), height( 0 ),pitch( 0 ), h_data( NULL ), d_data( NULL ),
						d_internalAlloc( false ), h_internalAlloc( false )
{

}
CudaImage::~CudaImage()
{
	if(d_internalAlloc && d_data!=NULL)
		CUDA_SAFECALL( cudaFree( d_data ) );
	d_data = NULL;
	if(h_internalAlloc && h_data!=NULL)
		free( h_data );
	h_data = NULL;
}
void CudaImage::Allocate( int w, int h, float *d_ptr, float *h_ptr )
{
	width = w;
	height = h;
	d_data = d_ptr;
	h_data = h_ptr;
	if (d_ptr==NULL)
	{
		CUDA_SAFECALL( cudaMallocPitch( (void **)&d_data, (size_t *)&pitch, (size_t)(width*sizeof( float )), (size_t)height ) );
		pitch /= sizeof( float );
		if(d_data==NULL)
			printf( "Failed to allocate device data\n" );
		d_internalAlloc = true;
	}
	if(h_ptr==NULL)
	{
		h_data = (float *)malloc( width*height*sizeof( float ) );
		if(h_data==NULL)
			printf( "Failed to allocate host data\n" );
		h_internalAlloc = true;
	}
}
double CudaImage::Clone( CudaImage &src, cudaStream_t stream)
{

	if(width!=src.width || height!=src.height)
	{
		printf("Clone Failed: dimension mismatch\n");
		return 0.0;
	}

	if(src.d_data!=NULL)
		CUDA_SAFECALL( cudaMemcpy2DAsync( (void *)d_data, (size_t)(pitch * sizeof( float )), (const void *)src.d_data, (size_t)(pitch * sizeof( float )), (size_t)(width*sizeof( float )), (size_t)height, cudaMemcpyDeviceToDevice, 0 ) );
	return 0;
}
double CudaImage::Upload( cudaStream_t stream )
{
	if(d_data!=NULL && h_data!=NULL)
		CUDA_SAFECALL( cudaMemcpy2DAsync( (void *)d_data, (size_t)(pitch * sizeof( float )), (const void *)h_data, (size_t)(width*sizeof( float )), (size_t)(width*sizeof( float )), (size_t)height, cudaMemcpyHostToDevice, stream ) );
	return 0;
}
double CudaImage::Readback( cudaStream_t stream )
{
	CUDA_SAFECALL( cudaMemcpy2DAsync( (void *)h_data, (size_t)(width*sizeof( float )), (const void *)d_data, (size_t)(pitch* sizeof( float )), (size_t)(width*sizeof( float )), (size_t)height, cudaMemcpyDeviceToHost, stream ) );
	return 0;
}
