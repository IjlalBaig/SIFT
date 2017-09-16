#include "excludes/cudaSiftD.cu"
#include <stdio.h>

void testcopyKernel(cudaStream_t &stream)
{
	kernel<<<1, 10, 0, stream>>>();
}
