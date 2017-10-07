#include "utils.h"

/*
 *
 *
 * Image manipulation
 */
int image::imload( cv::Mat &img, const std::string path, const bool color )
{
	if (color)
		cv::imread( path, CV_LOAD_IMAGE_COLOR ).convertTo( img, CV_32F );
	else
		cv::imread( path, CV_LOAD_IMAGE_GRAYSCALE ).convertTo( img, CV_32F );
	if (!img.data)
		std::cout <<  "Could not open or find the image" << std::endl ;
	return 0;
}
int image::imshow( const cv::Mat &img )
{
	cv::Mat tmp = img;
	img.convertTo( tmp, CV_8U );
	namedWindow( "Display window", cv::WINDOW_NORMAL );
	cv::imshow( "Display window", tmp );
	cv::waitKey( 0 );
	return 0;
}
int image::imsave( const std::string path, cv::Mat &img )
{
	if(!img.data)
		std::cout <<  "Could not open or find the image" << std::endl ;
	else
		imwrite( path, img );
	return 0;
}
int image::drawPoint( cv:: Mat &img, float x, float y, float scale, float orientation )
{
	float rho = scale;
	cv::Point pt1 = cv::Point( x, y );
	cv::Point pt2 = cylindrical2Catesian( pt1, rho, -orientation );	// orientation is clockwise
	cv::circle( img, pt1, rho, cv::Scalar( 0, 255, 255 ), 1, 8, 0 );
	cv::line( img, pt1, pt2, cv::Scalar( 0, 255, 255 ), 1, 8, 0 );
	return 0;
}
/*
 *
 *
 * Image filtering
 */
int imfilter::gaussian1D( float *kernelPtr, float sigma, int kernelSize )
{
	float sum = 0;
	for (int i = 0; i < kernelSize; ++i)
	{
		kernelPtr[i] = (1/(pow( 2.0f * PI, 0.5f ) * sigma)) * exp( -pow( i - (float(kernelSize - 1 )/2), 2 )  / (2 * pow( sigma, 2 )) );
		sum += kernelPtr[i];
	}
	// Normalize kernel
	for (int i = 0; i < kernelSize; ++i)
		kernelPtr[i] /= sum;
	return 0;
}
int imfilter::gaussian2D( float *kernelPtr, float sigma )
{
//	float kernelSize = gaussianSize( sigma );
//	float kernel1D[int( kernelSize )];
//	gaussian1D(kernel1D, sigma);
//	for (int j = 0; j < kernelSize; ++j)
//	{
//		for (int i = 0; i < kernelSize; ++i)
//			kernelPtr[int( kernelSize )*j + i] = kernel1D[i]*kernel1D[j];
//	}
	return 0;
}
/*
 *
 *
 *  Helper functions
 */
int iDivUp( int num, int den ){return (num%den != 0) ? (num/den + 1) : (num/den);}
int iAlignUp( int A, int a ){return (A%a != 0) ? (A + a - A%a) : (A);}
cv::Point cylindrical2Catesian( cv::Point &ptCylin, float rho, float theta )
{
	cv::Point ptCart;
	ptCart.x = ptCylin.x + rho * cos( theta * PI/180 );
	ptCart.y = ptCylin.y + rho * sin( theta * PI/180 );
	return ptCart;
}
