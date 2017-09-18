#ifndef CUDA_SIFT_H_H
#define CUDA_SIFT_H_H

// Prerequisites for Sift
void initDeviceConstant();

int exclusiveSift(float *d);
void testSetConstants(cudaStream_t &stream);
void testcopyKernel(cudaStream_t &stream);
void blurOctave(float *dst, float *src, int width, int pitch, int height, cudaStream_t &stream);
void sharedKernel( cudaStream_t &stream );
#endif
