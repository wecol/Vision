// DistanceTran.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp> 
using namespace cv;

int _tmain(int argc, _TCHAR* argv[])
{
	IplImage *src;
	src = cvLoadImage("cornfield.bmp");
	cvNamedWindow("lena",CV_WINDOW_AUTOSIZE );
	cvShowImage("lena",src);
	cvWaitKey(0);
	cvDestroyWindow("lena");
	cvReleaseImage(&src);
	return 0;
}

