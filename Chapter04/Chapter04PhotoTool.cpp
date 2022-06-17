// Chapter04PhotoTool.cpp : Defines the entry point for the application.
//

#include "Chapter04PhotoTool.h"
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>

//OpenCV includes
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

Mat img;

//OpenCV command line parser functions
//Keys accepted by command line parser
const char* keys = {
	"{help h usage ? || pprint this message}"
	"{@image || Image to process}"
};

void showHistoCallback(int state, void* userData) {
	//Separate image in BRG
	vector<Mat> bgr;
	split(img, bgr);

	//Create the histogram for 256 bins
	//The number of possible values [0..255]
	int numBins = 256;

	//Set the ranges for B, G, R last is not included
	float range[] = { 0, 256 };
	const float* histRange = { range };

	Mat bHist, gHist, rHist;

	calcHist(&bgr[0], 1, 0, Mat(), bHist, 1, &numBins, &histRange);
	calcHist(&bgr[1], 1, 0, Mat(), gHist, 1, &numBins, &histRange);
	calcHist(&bgr[2], 1, 0, Mat(), rHist, 1, &numBins, &histRange);

	//Draw the histogram
	//We have to drawlines for each channel
	int width = 512;
	int height = 300;

	//Create image with gray base
	Mat histImage(height, width, CV_8UC3, Scalar(20, 20, 20));

	//Normalize the histogram to size of the image
	normalize(bHist, bHist, 0, height, NORM_MINMAX);
	normalize(gHist, gHist, 0, height, NORM_MINMAX);
	normalize(rHist, rHist, 0, height, NORM_MINMAX);

	int binStep = cvRound((float)width / (float)numBins);
	for (int i = 1; i < numBins; i++) {
		line(histImage,
			Point(binStep * (i - 1), height - cvRound(bHist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(bHist.at<float>(i))),
			Scalar(255, 0, 0));

		line(histImage,
			Point(binStep * (i - 1), height - cvRound(gHist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(gHist.at<float>(i))),
			Scalar(0, 255, 0));

		line(histImage,
			Point(binStep * (i - 1), height - cvRound(rHist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(rHist.at<float>(i))),
			Scalar(0, 0, 255));

	}

	imshow("Histogram", histImage);
}

void equalizeCallback(int state, void* userData) {
	Mat result;

	//Convert BGR image to YCCbCr
	Mat ycrcb;
	cvtColor(img, ycrcb, COLOR_BGR2YCrCb);


	//Split image into channels
	vector<Mat> channels;
	split(ycrcb, channels);

	//Equalize the Y channel only
	equalizeHist(channels[0], channels[0]);

	//Merge the result channels
	merge(channels, ycrcb);

	//Convert coolor ycrcb to BGR
	cvtColor(ycrcb, result, COLOR_YCrCb2BGR);

	//Show image
	imshow("Equalized", result);
}

void lomoCallback(int state, void* userData) {
	Mat result;

	const double exponentialE = std::exp(1.0);

	//Create Look-up table for color curve effect
	Mat lut(1, 256, CV_8UC1);
	uchar* plut = lut.data;

	for (int i = 0; i < 256; i++) {
		double x = (double)i / 256.0;
		//lut.at<uchar>(i) = cvRound(256 * (1 / (1 + pow(exponentialE, -((x - 0.5) / 0.1)))));
		plut[i] = cvRound(256 * (1 / (1 + pow(exponentialE, -((x - 0.5) / 0.1)))));
	}

	//Split the image channels and apply curve transform only to red channel
	vector<Mat> bgr;
	split(img, bgr);
	LUT(bgr[2], lut, bgr[2]);

	//Merge results
	merge(bgr, result);

	//Create image for halo dark
	Mat halo(img.rows, img.cols, CV_32FC3, Scalar(0.3, 0.3, 0.3));

	//Create circle
	circle(halo, Point(img.cols / 2, img.rows / 2), img.cols / 3, Scalar(1, 1, 1), -1);
	blur(halo, halo, Size(img.cols / 3, img.cols / 3));

	//Convert the result to float to allow multiply by 1 factor
	Mat resultf;
	result.convertTo(resultf, CV_32FC3);

	//Multiply our result with halo
	multiply(resultf, halo, resultf);

	//Convert to 8 bits
	resultf.convertTo(result, CV_8UC3);

	//Show result
	imshow("Lomography", result);
}

void cartoonCallback(int state, void* userData) {
	/*EDGES*/
	//Apply medium filter to remove possible noise
	Mat imgMedian;
	medianBlur(img, imgMedian, 7);

	//Detect edges with canny
	Mat imgCanny;
	Canny(imgMedian, imgCanny, 50, 150);

	//Dilate the edges
	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
	dilate(imgCanny, imgCanny, kernel);

	//Scale edges values to 1 and  invert values
	imgCanny = imgCanny / 255;
	imgCanny = 1 - imgCanny;

	//Use float values to allow multiply between 0  and 1
	Mat imgCannyf;
	imgCanny.convertTo(imgCannyf, CV_32FC3);

	//Blur the edges to do smooth effect
	blur(imgCannyf, imgCannyf, Size(5, 5));

	/*COLOR*/
	//Apply bilateral filter to homogenizes color
	Mat imgBF;
	bilateralFilter(img, imgBF, 9, 150.0, 150.0);

	//Truncate colors
	Mat result = imgBF / 25;
	result = result * 25;

	/*MERGES COLOR + EDGES*/
	//Create 3 channels of edges
	Mat imgCanny3c;
	Mat cannyChannels[] = { imgCannyf, imgCannyf, imgCannyf };
	merge(cannyChannels, 3, imgCanny3c);

	//Convert color result to float
	Mat resultf;
	result.convertTo(resultf, CV_32FC3);

	//Multiply color and edges matrices
	multiply(resultf, imgCanny3c, resultf);

	//Convert to 8 bits color
	resultf.convertTo(result, CV_8UC3);

	//Show image
	imshow("Result", result);
}

int main(int argc, const char** argv) {
	CommandLineParser parser(argc, argv, keys);
	parser.about("Chapter 04. PhotoTool v1.0.0");

	//If request help shows
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	//String imgFile = parser.get<String>(0);

	//Check if params are correctly parsed in this variables
	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}

	String imgFile = "../lena.jpg";

	//Load image to process
	img = imread(imgFile);

	//Create window
	namedWindow("Input");

	//Create UI buttons
	createButton("Show histogram", showHistoCallback, NULL, QT_PUSH_BUTTON, 0);
	createButton("Equalize histogram", equalizeCallback, NULL, QT_PUSH_BUTTON, 0);
	createButton("Lomography effect", lomoCallback, NULL, QT_PUSH_BUTTON, 0);
	createButton("Cartoonize Effect", cartoonCallback, NULL, QT_PUSH_BUTTON, 0);
	/*
	if (img.data) {
		cout << "\n\n Image exist...\n\n\n";
		createButton("Show histogram", showHistoCallback, NULL, QT_PUSH_BUTTON, 0);
		createButton("Equalize histogram", equalizeCallback, NULL, QT_PUSH_BUTTON, 0);
		createButton("Lomography effect", lomoCallback, NULL, QT_PUSH_BUTTON, 0);
		createButton("Cartoonize Effect", cartoonCallback, NULL, QT_PUSH_BUTTON, 0);
	}
	else {
		cout << "\n\n EXCEPTION: No image to display...";
		return -1;
	}
	*/
	//Show image
	imshow("Input", img);
	waitKey(0);
	return 0;
}