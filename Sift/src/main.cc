#include <stdio.h>
#include <string>

#include "sift.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

int main( int argc, char **argv )
{
	// 	read argv[1] for img src locations
    std::ifstream inFile;
    inFile.open( argv[1] );
    if (!inFile)
    {
        std::cout << "Unable to open file";
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
			srcPath[i] = path;
		std::string dstPath(argv[2]);
		// 	launch sift
		sift( dstPath, srcPath, nImgs );
    }
    inFile.close();
    return 0;
}
