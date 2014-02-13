// DistanceTran.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp> 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "math.h"
#include <stdio.h>
using namespace cv;

Mat& DistTran(Mat &I);
Mat& NormImage(Mat &I);
static float Round(float f)
{
	return (ceil(f) - f > f - floor(f) ? floor(f) : ceil(f));
}

int ChessBoarDist(int x1, int y1, int x2, int y2)
{
	return(abs(x1-x2)>abs(y1-y2)?abs(x1-x2):abs(y1-y2));
}

float EuclideanDist(int x1, int x2, int y1, int y2)
{
	return sqrt((x1 - x2)*(x1 - x2) - (y1 - y2)*(y1 - y2));
}

int CityBlockDist(int x1, int x2, int y1, int y2)
{
	return abs(x1 - x2) + abs(y1 - y2);
}

int MyMin(int x, int y)
{
	return x < y ? x : y;
}

float MyMin(float x, float y)
{
	return x < y ? x : y;
}

Mat& NormImage(Mat&  I)
{
	CV_Assert(I.depth()!=sizeof(uchar));

	int channels = I.channels();
	int nRows = I.rows*channels;
	int nCols = I.cols;

	int i, j;
	uchar* p;
	int min = 256;
	int max = -1;

	for (int i = 1; i < nRows - 1; ++i)
	{
		p = I.ptr<uchar>(i);
		for (j = 1; j < nCols - 1; ++j)
		{
			if (min>p[j]) min = p[j];
			if (max < p[j]) max = p[j];
		}
	}


	for (int i = 1; i < nRows - 1; ++i)
	{
		p = I.ptr<uchar>(i);
		for (j = 1; j < nCols - 1; ++j)
		{
			p[j] = (p[j] - min) * 255 / (max - min);
		}
	}


	return I;
}

Mat& DistTran(Mat &I)
{
	CV_Assert(I.depth()!=sizeof(uchar));
	int channels = I.channels();
	int nRows = I.rows*channels;
	int nCols = I.cols;


	uchar* p;
	uchar* q;

	float fMin = 0.0;
	float fDis = 0.0;

	for (int i = 1; i < nRows - 1; ++i)
	{
		p = I.ptr<uchar>(i);
		for (int j = 2; j < nCols; ++j)
		{

			q = I.ptr<uchar>(i - 1);
			fDis = EuclideanDist(i, j, i - 1, j - 1);
			fMin = MyMin((float)p[j], fDis + q[j - 1]);

			
			fDis = EuclideanDist(i, j, i - 1, j);
			fMin = MyMin(fMin, fDis + q[j]);

			q = I.ptr<uchar>(i);
			fDis = EuclideanDist(i, j, i, j - 1);
			fMin = MyMin(fMin, fDis + q[j - 1]);

			q = I.ptr<uchar>(i + 1);
			fDis = EuclideanDist(i, j, i + 1, j - 1);
			fMin = MyMin(fMin, fDis + q[j - 1]);

			p[j] = (uchar)Round(fMin);

		}
	}
		

		for (int i = nRows - 2; i > 0; --i)
		{
			p = I.ptr<uchar>(i);
			for (int j = nCols - 1; j >= 0; --j)
			{

				q = I.ptr<uchar>(i - 1);
				fDis = EuclideanDist(i, j, i - 1, j + 1);
				fMin = MyMin((float)p[j], fDis + q[j + 1]);


				q = I.ptr<uchar>(i + 1);
				fDis = EuclideanDist(i, j, i+1, j );
				fMin = MyMin(fMin, fDis + q[j ]);

				
				fDis = EuclideanDist(i, j, i + 1, j + 1);
				fMin = MyMin(fMin, fDis + q[j + 1]);


				q = I.ptr<uchar>(i);
				fDis = EuclideanDist(i, j, i, j + 1);
				fMin = MyMin(fMin, fDis + q[j + 1]);

				

				p[j] = (uchar)Round(fMin);

			}
		}

		



		return I;
}

int _tmain(int argc, _TCHAR* argv[])
{
	/*char * imageName = "test_wushuang.jpg";

	Mat image;
	image = imread(imageName,1);

	if (!image.data)
	{
		printf("no data");
	}

	Mat gray_image;
	cvtColor(image, gray_image, CV_RGB2GRAY);

	DistTran(gray_image);
	NormImage(gray_image);
	imwrite("aftertransform.jpg",gray_image);
	namedWindow("Gray Image",CV_WINDOW_AUTOSIZE);
	imshow("Gray Image",gray_image);
	waitKey(0);*/

	IplImage *src = cvLoadImage("lenna.bmp", 1);
	IplImage *dst = cvCreateImage(cvGetSize(src),IPL_DEPTH_32F,1);
	IplImage *canny = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);

	cvCvtColor(src,canny,CV_RGB2GRAY);
	cvCanny(canny, canny, 100, 200, 3);
	cvDistTransform(canny, dst);

	cvNamedWindow("src", 1);
	cvShowImage("src", src);

	cvNamedWindow("canny", 1);
	cvShowImage("canny", canny);

	cvNamedWindow("dst", 1);
	cvShowImage("dst", dst);

	waitKey(0);

	cvReleaseImage(&src);
	cvReleaseImage(&canny);
	cvReleaseImage(&dst);
	return  0;






}

