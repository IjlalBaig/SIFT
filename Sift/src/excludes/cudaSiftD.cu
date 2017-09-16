#include "cudaSiftD.h"
#include <stdio.h>

__global__ void kernel()
{
	int count = 0;
	for (int i = 0; i < 5000; ++i)
		{count++;}
	printf("hi this is thread:%d\t%d\n", threadIdx.x, count);
}
