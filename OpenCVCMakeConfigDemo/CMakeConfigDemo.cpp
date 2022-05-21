// CMakeConfigDemo.cpp : Defines the entry point for the application.
//

#include "CMakeConfigDemo.h"
#include <iostream>	
#include <string>
#include <sstream>

/*OpenCV includes*/
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
	//Read images
	Mat color = imread("../lena.jpg");
	Mat gray = imread("../lena.jpg", IMREAD_GRAYSCALE);

	//Mat color = imread("C://RESEARCH-AND-PROJECT//PROJOCTS//C++//CMakeConfigDemo//images//lena.jpg");
	//Mat gray = imread("C://RESEARCH-AND-PROJECT//PROJOCTS//C++//CMakeConfigDemo//images//lena.jpg", IMREAD_GRAYSCALE);

	//Check for invalid inout
	if (!color.data) {
		cout << "Could not open or find image\n";
		return -1;
	}
	
	//Write images
	imwrite("LenaGray.jpg", gray);

	//Get same pixels with openCV function
	int myRow = color.cols - 1;
	int myCol = color.rows - 1;

	Vec3b pixel = color.at<Vec3b>(myRow, myCol);
	cout << "Pixel value (B,G,R): (" << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << ")\n";

	//Show image
	imshow("Lena BGR", color);
	imshow("LenaGray", gray);

	//Wait for any key press
	waitKey(0);
	return 0;
}
