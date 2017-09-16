#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H


class CudaImage {
	public:
		int width;
		int height;
		int pitch;
		float *h_data;
		float *d_data;
		bool d_internalAlloc;
		bool h_internalAlloc;

	public:
		CudaImage();
		~CudaImage();
		void Allocate( int width, int height, float *d_ptr, float *h_ptr );
        double Clone( CudaImage &src, cudaStream_t stream = 0  );
		double Upload( cudaStream_t stream = 0 );
		double Readback( cudaStream_t stream = 0 );
};

#endif
