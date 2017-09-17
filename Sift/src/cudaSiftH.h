#ifndef CUDA_SIFT_H_H
#define CUDA_SIFT_H_H

int exclusiveSift(float *d);
void testcopyKernel(cudaStream_t &stream);
void sharedKernel( cudaStream_t &stream );
#endif
