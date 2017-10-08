
#include <iostream>
#include <fstream>
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

int sift( std::string dstPath, std::string *srcPath, const int nImgs)
{
	int nImgProcessed = 0;
	int nImgQueue = 0;
	while (nImgs > nImgProcessed)
	{
		//	Update image queue count
		nImgQueue = (nImgs - nImgProcessed > 1) ? (2):(1);

		//	Load image queued to Mat object
		cv::Mat matImg[nImgQueue];
		int width[nImgQueue];
		int height[nImgQueue];

		//	Read in image dimensions
		for (int i = 0; i < nImgQueue; ++i)
		{
			image::imload( matImg[i], srcPath[nImgProcessed + i], false );
			width[i] = matImg[i].cols;
			height[i] = matImg[i].rows;
		}
		//	Allocate Cuda Objects
		CudaImage cuImg[nImgQueue];
		SiftData siftData[nImgQueue];
		for (int i = 0; i < nImgQueue; ++i)
		{
			cuImg[i].Allocate( width[i], height[i], NULL, (float *)matImg[i].data );
			siftData[i].Allocate(MAX_POINTCOUNT, NULL, NULL);
		}

		//	Create queue streams
		cudaStream_t stream[nImgQueue];
		for (int i = 0; i < nImgQueue; ++i)
			CUDA_SAFECALL( cudaStreamCreate( &stream[i] ) );

		// 	Set device constants
		initDeviceConstant();

		//	Set device variables
		initDeviceVariables();

		//	Execute sift on streams
		for (int i = 0; i < nImgQueue; ++i)
		{
			//	Upload CudaImage to GPU
			cuImg[i].Upload(stream[i]);
			siftData[i].Upload(stream[i]);

			//	Allocate result pointer
			/*	remove later
			 *
			 */
			int w = iDivUp( cuImg[i].width, pow(2, RESULT_OCTAVE ) );
			int p = iAlignUp( w, 128);
			int h = iDivUp( cuImg[i].height, pow(2, RESULT_OCTAVE ) );
			float *d_res;
			cv::Mat1f matRes;
			matRes.create(h, w);
			CUDA_SAFECALL( cudaMalloc( (void **) &d_res,(size_t)( p*h*sizeof( float ) ) ) );
			/*
			 *
			 *
			 *
			 *
			 *
			 */
			extractSift(siftData[i].d_data, d_res, cuImg[i].d_data, cuImg[i].width, cuImg[i].pitch, cuImg[i].height, stream[i], i, 0 );
			//	display result
			/*
			 *
			 */
			CUDA_SAFECALL( cudaMemcpy2DAsync( (void *)matRes.data, (size_t)(w*sizeof( float )), (const void *)d_res, (size_t)(p* sizeof( float )), (size_t)(w*sizeof( float )), (size_t)h, cudaMemcpyDeviceToHost, stream[i]) );
			//	imshow halts kernel execution
//			image::imshow( matRes );
			CUDA_SAFECALL( cudaFree( d_res ) );
			/*
			 *
			 *
			 *
			 *
			 */
		}
		//	Download results to CPU
		for (int i = 0; i < nImgQueue; ++i)
		{
			cuImg[i].Readback(stream[i]);
			siftData[i].Readback(stream[i]);
		}
			/*
			 *
			 */
		//	Save result
		for (int i = 0; i < nImgQueue; ++i)
		{
			std::string srcFile( srcPath[nImgProcessed + i].substr( srcPath[nImgProcessed + i].find_last_of("\\/") + 1, srcPath[nImgProcessed + i].size() ) );
			std::string dstFile(dstPath + "/" + srcFile + ".sift.txt");
			std::ofstream outFile(dstFile.c_str());

			for (int j = 0; j < MAX_POINTCOUNT; ++j)
			{
				outFile << siftData[i].h_data[j].xpos << "\t";
				outFile << siftData[i].h_data[j].ypos << "\t";
				outFile << siftData[i].h_data[j].scale << "\t";
				outFile << siftData[i].h_data[j].orientation << "\t";
				for (int k = 0; k < 128; ++k)
					outFile << siftData[i].h_data[j].data[k] << " ";
				outFile << std::endl;
				image::drawPoint( matImg[i], siftData[i].h_data[j].xpos, siftData[i].h_data[j].ypos, siftData[i].h_data[j].scale, siftData[i].h_data[j].orientation);
//
			}
			for (int i = 0; i < nImgQueue; ++i)
				image::imshow( matImg[i] );
			outFile.close();
		}

		//	Destroy cuda streams for queue
		for (int i = 0; i < nImgQueue; ++i)
			CUDA_SAFECALL( cudaStreamDestroy(stream[i]));

		//	Update images processed count
		nImgProcessed += nImgQueue;
	}
	return 0;
}


