#ifndef CUDA_SIFT_H_H
#define CUDA_SIFT_H_H

#define RESULT_OCTAVE 0

void initDeviceConstant();
void initDeviceVariables();
void allocateOctave( float *&multiBlur, float *&multiDoG
		, float *&multiHessian, float *&multiGradient
		, int width, int pitch, int height );
void freeOctave( float *&multiBlur, float *&multiDoG
		, float *&multiHessian, float *&multiGradient );
void setTexture(cudaArray *&cuArray, cudaTextureObject_t &texObj , float *linearSrc
		, int width, int pitch, int height);
void freeTexture(cudaArray *&cuArray, cudaTextureObject_t &texObj);

void blurOctave( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream );
void computeHessian( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream );
void computeGradient( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream );
void computeDiffOfGauss( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream );
void computeOctaveSift( SiftPoint *pt, float *src_DoG
		, float *src_Hessian, float *src_Gradient
		, int width, int pitch, int height
		, cudaStream_t &stream, int streamIdx, int octaveIdx );
void extractSift( SiftPoint *siftPt, float *d_res, float *d_src
		, int width, int pitch, int height
		, cudaStream_t &stream, int streamIdx, int octaveIdx );


int getPointCount( int streamIdx );
int clampPtCount( int streamIdx );
int updatePtStartIdx( int streamIdx );
/*
 *
 *
 */
void copyDeviceData( float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream );
/*
 *
 *
 */
#endif
