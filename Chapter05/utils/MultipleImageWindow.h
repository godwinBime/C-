// Chpater05.h : Include file for standard system include files,
// or project specific include files.
/*
Multiple image window

Thia class creates a  window with multiple images,
shoow on it a grid with optional titles on each one
@author: David Millan Escriva 
*/

#ifndef MIW_h
#define MIW_h


#pragma once

#include <iostream>
#include <string>	
using namespace std;

//OpenCV includes
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

class MultipleImageWindow {
public:
	/*
	Constructor
	Create new window with a max of cols * row images
	
	@param string window_title
	@param int cols number of cols
	@param int rows number of rows
	@param int flags see highgui window documentation*/

	MultipleImageWindow(string windowTitle, int cols, int rows, int flags);

	/*Add new image to stack os window
	@param Mat image
	@param string title caption of image to show
	@return int position of  image in stack*/

	int addImage(string title, Mat image, bool render = false);

	/*Remove an image from position n*/
	void removeImage(int position);

	/*Render/redraw/update window*/
	void render();

private:
	int cols;
	int rows;
	int canvasWidth;
	int canvasHeight;
	string windowTitle;
	vector<string> titles;
	vector<Mat> images;
	Mat canvas;
};

// TODO: Reference additional headers your program requires here.
#endif // !MIW_H