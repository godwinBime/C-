#include <iostream>
#include <string> 
#include <sstream>

//OpenCV includes
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;

int main(int arg, const char** argv) {
	//Read images
	//Mat color = imread("C:\\RESEARCH-AND-PROJECT\\PROJOCTS\\C++\\ReadWriteImages\\lena.jpg");
	//Mat gray = imread("C:\\RESEARCH-AND-PROJECT\\PROJOCTS\\C++\\ReadWriteImages\\lena.jpg", IMREAD_GRAYSCALE);

	Mat color = imread("images/lena.jpg");
	Mat gray = imread("images/lena.jpg", IMREAD_GRAYSCALE);

	if (!color.data) {
		cout << "Could not open or find the image...\n";
		return -1;
	}

	/*Write images*/
	imwrite("images/lenagray.jpg", gray);

	//Get same ppixel with OpenCV function
	int myRow = color.cols - 1;
	int myCol = color.rows - 1;

	Vec3b pixel = color.at<Vec3b>(myRow, myCol);
	cout << "\n\nPixel value (B, G, R): (" << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << ")\n\n\n";

	//Show images
	imshow("Lena BGR", color);
	imshow("Lena gray", gray);

	//Wait for any key press
	waitKey(0);
	destroyAllWindows();
	return 0;
}