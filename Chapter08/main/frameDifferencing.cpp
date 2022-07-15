//
// Created by Godwin on 7/11/2022.
//
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;

Mat frameDiff(Mat prevFrame, Mat curFrame, Mat nextFrame){
    Mat diffFrames1, diffFrames2, output;

    //Compute absolute difference between current frame and the next
    absdiff(nextFrame, curFrame, diffFrames1);

    //Compute absolute difference between current frame and the previous
    absdiff(curFrame, prevFrame, diffFrames2);

    //Bitwise "AND" operation between the previous two diff images
    bitwise_and(diffFrames1, diffFrames2, output);

    return output;
}

/**
 * Extract and returns a frame from the webcam
 * **/

Mat getFrame(VideoCapture cap, float scalingFactor){
    Mat frame, output;

    //Capture the current frame
    cap >> frame;

    //Resize the frame
    resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

    //Convert to grayscale
    cvtColor(frame, output, COLOR_BGR2GRAY);

    return output;
}

int main_(int argc, char* argv[]){
    Mat frame, prevFrame, curFrame, nextFrame;
    char ch;

    //Create the capture object
    //0 -> input arg that specifies it should take the
    // input from the webcam
    VideoCapture cap(0);

    //If you cannot open the webcam, stop the execution
    if (!cap.isOpened()){
        cout<<"\n\nFailed to open webcam. Exiting...\n\n";
        return -1;
    } else{
        cout<<"\n\nSuccessfully opened webcam...\n\n";
    }

    //Create GUI windows
    namedWindow("Frame");

    //Scaling factor to resize the input frames from
    //the webcam.
    float scalingFactor = 0.75;

    prevFrame = getFrame(cap, scalingFactor);
    curFrame = getFrame(cap, scalingFactor);
    nextFrame = getFrame(cap, scalingFactor);

    //Iterate until the user presses the 'Esc' key
   do{
       //Show the object movement
       imshow("Object Movement", frameDiff(prevFrame, curFrame, nextFrame));

       //Update the variables and grab the next frame
       prevFrame = curFrame;
       curFrame = nextFrame;
       nextFrame = getFrame(cap, scalingFactor);

       //Get the keyboard input and check if it's 'Esc'
       //27 -> ASCII value for 'Esc' key
       ch = waitKey(30);
       if (ch == 27){
           break;
       }
   } while (true);

   //Release the video capture object
   cap.release();

   //Close all windows
   destroyAllWindows();
    return 1;
}
