#include <stdio.h>
#include <string>

#include "sift.h"


int main( int argc, char **argv )
{

	/* Create a string array with source paths*/
	std::string srcPath[BATCH_SIZE];
	for (int i = 0; i < BATCH_SIZE; ++i)
		srcPath[i] = std::string( argv[i+1] );

	/* Define destination paths*/
	std::string dstPath("img/result.ppm" );
	/* Launch sift*/
	sift( dstPath, srcPath);
	return 0;
}
