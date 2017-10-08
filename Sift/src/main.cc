#include <stdio.h>
#include <string>

#include "sift.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;
int main( int argc, char **argv )
{
	// 	read argv[1] for img src locations
    ifstream inFile;
    inFile.open( argv[1] );
    if (!inFile)
    {
        cout << "Unable to open file";
        exit(1); // terminate with error
    }

    //	count number of images in batch
    int nImgs = 0;
    std::string path;
    while (inFile >> path)
        nImgs++;
	inFile.clear();
	inFile.seekg( 0, inFile.beg );

    //	load image locations
    if (nImgs > 0)
    {
    	std::string srcPath[nImgs];
		for (int i = 0; std::getline( inFile, path ); ++i)
		{
			srcPath[i] = path;
			std::cout << srcPath[i] << std::endl;
		}

		//	define resultant image path
		std::string dstPath( "result/result.ppm" );

		// 	launch sift
		sift( dstPath, srcPath, nImgs=1 );
    }
    inFile.close();


	return 0;
}
