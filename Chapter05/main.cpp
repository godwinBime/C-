#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <memory>

//OpenCV includes
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "utils/MultipleImageWindow.h"

using namespace cv;
using namespace std;

shared_ptr<MultipleImageWindow> miw;

/*OpenCV command line parser functions
Keys accepteed by command line parser*/

const char* keys = {
	"{help h usage ? || print this message}"
	"{@image "" Image to process}"
	"{@lightPattern || Image light pattern to apply to image input}"
	"{lightMethod | 1 | Method to remove background light, 0 difference, 1 div}"
	"{segMethod | 1 | Method to segment: 1 connected components, 2 connected components with stats, 3 find contours}"
};

Mat removeLight(Mat img, Mat pattern, int method) {
	Mat aux;
	//if method is normalized
	if (method == 1) {
		//Required change our image to 32 float for division
		Mat img32, pattern32;
		img.convertTo(img32, CV_32F);
		pattern.convertTo(pattern32, CV_32F);

		//Divide the image by the pattern
		aux = 1 - (img32 / pattern32);

		//Convert 8 bits format and scale
		aux.convertTo(aux, CV_8U, 255);
	}else {
		aux = pattern - img;
	}

	return aux;
}

Mat calculateLightPattern(Mat img) {
	Mat pattern;
	/*Basic and effective way to calculate the 
	light pattern from one image*/
	blur(img, pattern, Size(img.cols / 3, img.cols / 3));
	return pattern;
}

static Scalar randomColor(RNG& rng) {
	auto iColor = (unsigned)rng;
	return Scalar(iColor & 255, (iColor >> 8) & 255, (iColor >> 16) & 255);
}

void ConnectedComponents(Mat img) {
	//Use connected components to divide our image in 
	//multiple connected component objects
	Mat labels; 
	auto numObjects = connectedComponents(img, labels);
	
	//Check the number of objects detected
	if (numObjects < 2) {
		cout << "\n\n\nNo objects detected." << "\n\n\n";
		return;
	}
	else{
		cout << "\n\n Number of objects detected: " << numObjects - 1 << "\n\n";
	}

	//create output image coloring the objects
	Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
	RNG rng(0xFFFFFFFF);
	for (auto i = 1; i < numObjects; i++) {
		Mat mask = labels == i;
		output.setTo(randomColor(rng), mask);
	}

	imshow("Result", output);
	miw->addImage("Result", output);
}

void ConnectedComponentsStats(Mat img) {
	//Use connected components with stats
	Mat labels, stats, centroids;
	auto numObjects = connectedComponentsWithStats(img, labels, stats, centroids);

	//Check the number of objects detected
	if (numObjects < 2) {
		cout << "\n\n\nNo objects detected.\n\n";
		return;
	}
	else {
		cout << "\n\n\nNumber of objects detected: " << numObjects - 1 << "\n\n\n";
	}

	//Create output image coloring the objects and show area
	Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
	RNG rng(0xFFFFFFFF);

	for (auto i = 1; i < numObjects; i++) {
		cout << "\n\n\Object " << i << " with pos: " << centroids.at<Point2d>(i) << " with area " << stats.at<int>(i, CC_STAT_AREA) << "\n\n\n";
		Mat mask = labels == i;
		output.setTo(randomColor(rng), mask);

		//Draw text with area
		stringstream ss;
		ss << "area: " << stats.at<int>(i, CC_STAT_AREA);
		putText(output, ss.str(), centroids.at<Point2d>(i), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
	}
	imshow("Result", output);

	miw->addImage("Result", output);
}

void FindContoursBasic(Mat img) {
	vector<vector<Point>> contours;
	findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);

	//Check the number of objects detected
	if (contours.size() == 0) {
		cout << "\n\n No objects detected: " << "\n\n\n";
	}
	else {
		cout << "\n\n\n Number of objects detected: " << contours.size() << "\n\n\n";
	}

	RNG rng(0xFFFFFFFF);
	for (auto i = 0; i < contours.size(); i++) {
		drawContours(output, contours, i, randomColor(rng));
	}
	imshow("Result", output);
	miw->addImage("Ressult", output);
}

int main(int argc, const char** argv){
	CommandLineParser parser(argc, argv, keys);
	parser.about("Chapter 5. Photo Tool v1.0.0");

	//if requires help, show
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	String imgFile = parser.get<String>(0);
	String lightPatternFile = parser.get<String>(1);
	auto methodLight = parser.get<int>("lightMethod");
	auto methodSeg = parser.get<int>("segMethod");

	//Check if params are correctly parsed in this variables
	if (!parser.check()) {
		parser.printErrors();
			return 0;
	}

	//Load image to process
	Mat img = imread(imgFile, 0);
	if (img.data == NULL) {
		cout << "\n\n\nError loading image " << imgFile << "\n\n";
		return 0;
	}

	//Create multiple image window
	miw = make_shared<MultipleImageWindow>("Main window", 3, 2, WINDOW_AUTOSIZE);

	//Remove noise
	Mat imgNoise, imgBoxSmooth;
	medianBlur(img, imgNoise, 3);
	blur(img, imgBoxSmooth, Size(3, 3));

	//Load image to process
	Mat lightPattern = imread(lightPatternFile, 0);
	if (lightPattern.data == NULL) {
		//Calculate the light pattern
		lightPattern = calculateLightPattern(imgNoise);
	}
	medianBlur(lightPattern, lightPattern, 3);

	//Apply the light pattern
	Mat imgNoLight;
	imgNoise.copyTo(imgNoLight);
	if (methodLight != 2) {
		imgNoLight = removeLight(imgNoise, lightPattern, methodLight);
	}

	//Binarize image for segment
	Mat imgThr;
	if (methodLight != 2) {
		threshold(imgNoLight, imgThr, 30, 255, THRESH_BINARY);
	}
	else{
		threshold(imgNoLight, imgThr, 140, 255, THRESH_BINARY_INV);
	}

	//Show images
	miw->addImage("Input", img);
	miw->addImage("Input without noise", imgNoise);
	miw->addImage("Light Pattern", lightPattern);
	miw->addImage("No light", imgNoLight);
	miw->addImage("Threshold", imgThr);

	switch (methodSeg) {
	case 1:
		ConnectedComponents(imgThr);
		break;
	case 2:
		ConnectedComponentsStats(imgThr);
		break;
	case 3:
		FindContoursBasic(imgThr);
		break;
	}

	miw->render();
	waitKey(0);
	return 0;
}