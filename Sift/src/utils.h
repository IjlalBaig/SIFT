#ifndef UTILS_H
#define UTILS_H

#define PI 3.14159265
#define B_KERNEL_RADIUS 9
#define N_OCTAVES 4
#define N_SCALES 2
#define BATCH_SIZE 1
#define MAX_POINTCOUNT 1500

#define EXTREMA_THRESH 5.0f
#define R_THRESH 10.0f

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <cmath>

namespace image{
int imload( cv::Mat &img, const std::string path, const bool color );
int imshow( const cv::Mat &img );
int imsave( const std::string path, cv::Mat &img );
int drawPoint( cv:: Mat &img, float x, float y, float scale, float orientation );
}

namespace imfilter{
int gaussian1D(float *kernelPtr, float sigma);
int gaussian2D(float *kernelPtr, float sigma);
}

int iDivUp( int num, int den );
int iAlignUp( int dataSize, int alignSize );
cv::Point cylindrical2Catesian( cv::Point &ptCylin, float rho, float theta );

#endif
