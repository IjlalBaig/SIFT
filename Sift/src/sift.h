#ifndef SIFT_H
#define SIFT_H

#include <string>
#include <cuda_runtime.h>

#define BATCH_SIZE 1
#define MAX_POINTS 1500


typedef struct {
  float xpos;
  float ypos;
  float scale;
  float orientation;
  float data[128];
} SiftPoint;

class SiftData {
	public:
	  int numPts;         // Number of available Sift points
	  int maxPts;         // Number of allocated Sift points
	  SiftPoint *h_data;  // Host (CPU) data
	  SiftPoint *d_data;  // Device (GPU) data
	  bool d_internalAlloc;
	  bool h_internalAlloc;

	public:
		SiftData();
		~SiftData();
		void Allocate( int maxPts, SiftPoint *d_ptr, SiftPoint *h_ptr);
		double Upload(cudaStream_t stream = 0 );
		double Readback(cudaStream_t stream = 0 );
};

int sift( std::string dstPath, std::string *srcPath);
#endif
