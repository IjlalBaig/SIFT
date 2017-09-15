#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
#define CUDA_SAFECALL(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#endif
