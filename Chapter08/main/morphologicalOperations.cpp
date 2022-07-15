//
// Created by Godwin on 7/15/2022.
//
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

//Create the structuring element for erosion
Mat element(int morphologyType, int morphologySize){
    Mat elementOutput = getStructuringElement(morphologyType, Size(2 * morphologySize + 1, 2 * morphologySize + 1), Point(morphologySize, morphologySize));
    return elementOutput;
}

void morphElement(int morphologyElement, int morphologyType){
    if (morphologyElement == 0)
        morphologyType = MORPH_RECT;
    else if (morphologyElement == 1)
        morphologyType = MORPH_CROSS;
    else if (morphologyType == 2)
        morphologyType = MORPH_ELLIPSE;
}

Mat performOpening(Mat& inputImage, int morphologicalElement, int morphologySize){
    Mat outputImage, tempImage;
    int morphologyType = 0;

    morphElement(morphologicalElement, morphologyType);

    //Apply morphological opening to the image using the structuring element.
    erode(inputImage, tempImage, element(morphologyType, morphologySize));
    dilate(tempImage, outputImage, element(morphologyType, morphologySize));

    //Return the output image
    return outputImage;
}

Mat performClosing(Mat& inputImage, int morphologyElement, int morphologySize){
    Mat outputImage, tempImage;
    int morphologyType = 0;

    morphElement(morphologyElement, morphologyType);

    //Apply morphological opening to the image using the structuring element
    dilate(inputImage, tempImage, element(morphologyType, morphologySize));
    erode(tempImage, outputImage, element(morphologyType, morphologySize));

    //Return the output image
    return outputImage;
}

Mat performMorphologicalGradient(Mat& inputImage, int morphologyElement, int morphologySize){
    Mat outputImage, tempImage1, tempImage2;
    int morphologyType = 0;

    morphElement(morphologyElement, morphologyType);

    //Apply morphological Gradient to the image using the structuring element
    dilate(inputImage, tempImage1, element(morphologyType, morphologySize));
    erode(inputImage, tempImage2, element(morphologyType, morphologySize));

    //Return the output image
    return  tempImage1 - tempImage2;
}

Mat performTopHat(Mat& inputImage, int morphologyElement, int morphologySize){
    Mat outputImage;
    int morphologyType = 0;

    morphElement(morphologyElement, morphologyType);

    //Create the structuring element for erosion
    Mat elements = element(morphologyType, morphologySize);

    //Apply top hat operation to the image using the structuring element.
    outputImage = inputImage - performOpening(inputImage, morphologyElement, morphologySize);

    //Return the output image
    return outputImage;
}

Mat performBlackHat(Mat& inputImage, int morphologyElement, int morphologySize){
    Mat outputImage;
    int morphologyType = 0;

    morphElement(morphologyElement, morphologyType);

    //Create the structuring element for erosion
    Mat elements = element(morphologyType, morphologySize);

    //Apply black hat operation to the image using the structuring element
    Mat blackHatOutputImage = performClosing(inputImage, morphologyElement, morphologySize) - inputImage;

    outputImage = blackHatOutputImage - inputImage;

    //Return the output image
    return outputImage;
}

int main(int argc, char* argv[]){
    Mat inputImage, openingOutputImage, closingOutputImage;
    Mat gradientOutputImage, topHatOutputImage, blackHatOutputImage;

    if (argc < 3){
        cerr<<"\n\nInsufficient input args. The format is:\n$ ./main image.jpg\n\n";
        return -1;
    } else
        cout<<"\n\nSufficient input args received...\n\n";

    //Read the input image
    inputImage = imread(argv[1]);

    //Read the input value for the amount of erosion
    int morphologySize;
    istringstream iss(argv[2]);

    iss >> morphologySize;

    //Check the validity of the input image
    if (!inputImage.data){
        cerr<<"\n\nInvalid input image. Exiting...\n\n";
        return -1;
    } else
        cout<<"\n\nImage input is valid...\n\n";

    int morphologyElement = 0;

    //Create windows to display the input and output images
    namedWindow("Input Image", WINDOW_AUTOSIZE);
    namedWindow("Output Image after opening", WINDOW_AUTOSIZE);
    namedWindow("Output Image after closing", WINDOW_AUTOSIZE);
    namedWindow("Output Image after morphological gradient", WINDOW_AUTOSIZE);
    namedWindow("Output Image after black hat", WINDOW_AUTOSIZE);

    //Apply morphological opening
    openingOutputImage = performOpening(inputImage, morphologyElement, morphologySize);

    //Apply morphological closing
    closingOutputImage = performClosing(inputImage, morphologyElement, morphologySize);

    //Apply morphological gradient
    gradientOutputImage = performMorphologicalGradient(inputImage, morphologyElement, morphologySize);

    //Apply top hat operation
    topHatOutputImage = performTopHat(inputImage, morphologyElement, morphologySize);

    //Apply black hat operation
    blackHatOutputImage = performBlackHat(inputImage, morphologyElement, morphologySize);

    //Display the output image
    imshow("Input Image", inputImage);
    imshow("Output Image after opening", openingOutputImage);
    imshow("Output image after closing", closingOutputImage);
    imshow("Output image after morphological gradient.", gradientOutputImage);
    imshow("Output image after top hat", topHatOutputImage);
    imshow("Output image after black hat", blackHatOutputImage);

    //Wait until the user hits a key on the keyboard.
    waitKey(0);

    return 0;
}