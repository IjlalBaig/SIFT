
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
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

std::mutex mu_save;
int sift( std::string dstPath, std::string *srcPath, int nImgs)
{
	int nImgProcessed = 0;
	SiftPoint *siftResult;
	while (nImgs > nImgProcessed)
	{
		//	Update image queue count
		int nImgQueue = (nImgs - nImgProcessed > 1) ? (2):(1);
		//	Load image queued to Mat object
		cv::Mat matImg[2];
		int width[2];
		int height[2];

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

		//	Upload CudaImage to GPU
		for (int i = 0; i < nImgQueue; ++i)
		{
			cuImg[i].Upload(stream[i]);
			siftData[i].Upload(stream[i]);
		}
			//	Execute sift on streams
		for (int i = 0; i < nImgQueue; ++i)
			extractSift(siftData[i].d_data, cuImg[i].d_data, cuImg[i].width, cuImg[i].pitch, cuImg[i].height, stream[i], i, 0 );
		//	Download results to CPU
		for (int i = 0; i < nImgQueue; ++i)
		{
			cuImg[i].Readback(stream[i]);
			siftData[i].Readback(stream[i]);
		}


		for (int i = 0; i < nImgQueue; ++i)
		{
			siftResult = siftData[i].h_data;
			siftData[i].h_data = nullptr;
			std::thread t1(saveSift, dstPath, srcPath[nImgProcessed + i], siftResult, getPointCount( i ) - 1);
			t1.detach();
		}
		//	Destroy cuda streams for queue
		for (int i = 0; i < nImgQueue; ++i)
			CUDA_SAFECALL( cudaStreamDestroy(stream[i]));

		//	Update images processed count
		nImgProcessed += nImgQueue;
	}
	bool success = mu_save.try_lock();
	while (!success)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(2));
		success = mu_save.try_lock();
	}
	mu_save.unlock();
	return 0;
}
void saveSift(std::string dstPath, std::string srcPath, SiftPoint *h_data, int ptCount)
{
	std::unique_lock<std::mutex> locker(mu_save, std::defer_lock);
	locker.lock();
	std::string srcFile( srcPath.substr( srcPath.find_last_of("\\/") + 1, srcPath.size() ) );
	std::string dstFile(dstPath + "/" + srcFile + ".sift.txt");
	std::ofstream outFile(dstFile.c_str());
	for (int j = 0; j < ptCount ; ++j)
	{
		outFile << std::fixed << std::setprecision(1) << float(h_data[j].xpos) << "\t";
		outFile << std::fixed << std::setprecision(1) << float(h_data[j].ypos) << "\t";
		outFile << std::fixed << std::setprecision(3) << float(h_data[j].scale) << "\t";
		outFile << std::fixed << std::setprecision(3) << float(h_data[j].orientation) << "\t";
		for (int k = 0; k < 128; ++k)
			outFile << std::fixed << std::setprecision(6)<< h_data[j].data[k] << " ";
		outFile << std::endl;
	}
	outFile.close();
	locker.unlock();
}


