//
// Created by Godwin on 7/11/2022.
//
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <memory>

using namespace std;
using namespace cv;

Mat performErosion(const Mat& inputImage, int erosionElement, int erosionSize){
    Mat outputImage;
    int erosionType;

    if (erosionElement == 0)
        erosionType = MORPH_RECT;
    else if (erosionElement == 1)
        erosionType = MORPH_CROSS;
    else if (erosionElement == 2)
        erosionType = MORPH_ELLIPSE;

    //Create the structuring element for erosion
    Mat element = getStructuringElement(erosionType, Size(2 * erosionSize + 1, 2 * erosionSize + 1), Point
    (erosionSize, erosionSize));

    //Erode the image using the structuring element
    erode(inputImage, outputImage, element);

    return outputImage;
}

int main____(int argc, char* argv[]){
    Mat inputImage, outputImage;

    if (argc < 3){
        cerr <<"\n\nInsufficient input args. The format is:\n$ ./main image.jpg 5\n\n";
        return -1;
    } else{
        cout<<"\n\nCorrect input args passed...\n\n";
    }

    //Read the input image
    inputImage = imread(argv[1]);

    //Read the input value for the amount of erosion
    int erosionSize;
    istringstream iss(argv[2]);
    iss >> erosionSize;

    //Check validity of input image
    if (!inputImage.data){
        cout <<"\n\nInvalid input image. Exiting...\n\n";
        return -1;
    }else{
        cout<<"\n\nValid input image...\n\n";
    }

    int erosionElement = 0;

    //Create windows to display the input and output images.
    namedWindow("Input Image", WINDOW_AUTOSIZE);
    namedWindow("Output Image", WINDOW_AUTOSIZE);

    //erode the image
    outputImage = performErosion(inputImage, erosionElement, erosionSize);

    //Display the output image
    imshow("Input Image", inputImage);
    imshow("Output Image", outputImage);

    //Wait until the user hits a key on the keyboard
    waitKey(0);
    return 0;
}