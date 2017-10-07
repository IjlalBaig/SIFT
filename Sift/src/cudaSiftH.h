#ifndef CUDA_SIFT_H_H
#define CUDA_SIFT_H_H

// Prerequisites for Sift
void initDeviceConstant();
void initDeviceVariables();

int exclusiveSift( float *d );
void testSetConstants( cudaStream_t &stream );
void testcopyKernel( cudaStream_t &stream );
void sharedKernel( cudaStream_t &stream );
void testMax( cudaStream_t &stream );
void blurOctave( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream );
void allocateOctave( float *&multiBlur, float *&multiDoG
					, float *&multiHessian, float *&multiMagnitude, float *&multiDirection
					, int width, int pitch, int height );
void freeOctave( float *&multiBlur, float *&multiDoG
				, float *&multiHessian, float *&multiMagnitude, float *&multiDirection );
void extractSift( SiftPoint *siftPt, float *d_res, int resOctave, float *d_src, int width, int pitch, int height, int octaveIdx, cudaStream_t &stream, int streamIdx );
void copyDeviceData( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream );
#endif
