//
// Created by Godwin on 7/6/2022.
//

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>

#define CV_HAAR_SCALE_IMAGE 2

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
    string faceCascadeName = argv[1];
    string eyeCascadeName = argv[2];
    Mat eyeMask = imread(argv[3]);
    CascadeClassifier faceCascade, eyeCascade;
    VideoCapture cap(0);

    if (!cap.isOpened()){
        cerr <<"\n\nError opening webcam...\n\n";
        return -1;
    } else{
        cout<<"\n\nWebcam access granted...\n\n";
    }

    if (!eyeCascade.load(eyeCascadeName)){
        cerr <<"\n\nError loading eye cascade file...\n\n";
        return -1;
    } else{
        cout <<"\n\nLoaded eye cascade file...\n\n";
    }

    if (!faceCascade.load(faceCascadeName)){
        cerr <<"\n\nError loading face cascade file...\n\n";
        return -1;
    } else{
        cout <<"\n\nLoaded face cascade file...\n\n";
    }

    if (!eyeMask.data){
        cerr <<"\n\nError loading mask image...\n\n";
        return -1;
    } else{
        cout <<"\n\nLoaded mask image...\n\n";
    }

    float scalingFactor = 0.75;

    Mat frame, frameGray;
    Mat frameROI, eyeMaskSmall;
    Mat grayMaskSmall, grayMaskSmallThresh, grayMaskSmallThreshInv;
    Mat maskedEye, maskedFrame;

    char ch;

    namedWindow("Frame");
    vector<Rect> faces;

    //Draw green circles around the eyes
    while (true) {
        cout<<"\nInside while loop...\n";
        //Capture the current frame
        cap >> frame;

        //Resize the frame
        cout<<"\nResizing frame...\n";
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);
        cout<<"\nSuccessfully resized frame...\n";

        //Convert to gray scale
        cout<<"\nConvert image to gray scale...\n";
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        cout<<"\nSuccessfully converted image to gray scale...\n";

        //Equalize the histogram
        cout<<"\nEqualize...\n";
        equalizeHist(frameGray, frameGray);
        cout<<"\nEqualize success...\n";

        //Detect faces
        cout<<"\nDetecting face...\n";
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
        cout<<"\nFace successfully detected...\n";

        vector<Point> centers;

        for (int i = 0; i < faces.size(); i++) {
            cout<<"\nInside faces for-loop\n";
            Mat faceROI = frameGray(faces[i]);
            vector<Rect> eyes;

            //In each face, detect eyes
            cout<<"\nDetecting eyes...\n";
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
            cout<<"\nEyes detected...\n";

            //For each eye detected, compute the center
            for (int j = 0; j < eyes.size(); j++) {
                cout<<"\nInside eye detection for-loop\n";
                Point center(faces[i].x + eyes[j].x + int(eyes[j].width * 0.5),
                             faces[i].y + eyes[j].y + int(eyes[j].height * 0.5));
                centers.push_back(center);
            }
        }

        //Overlay sunglasses if both eyes are detected
        if (centers.size() == 2){
            cout<<"\nInside if statement...\n";
            cout<<"\nBoth eyes detected...\n";
            Point leftPoint, rightPoint;

            //Identify the left and right eyes
            if (centers[0].x < centers[1].x){
                leftPoint = centers[0];
                rightPoint = centers[1];
                cout<<"\nFirst eye detected...\n";
            } else{
                leftPoint = centers[1];
                rightPoint = centers[0];
                cout<<"\nSecond eye detected...\n";
            }

            //Custom parameters to make the glasses fit
            //You may have to adjust them to make sure it works
            int w = 2.3 * (rightPoint.x - leftPoint.x);
            int h = int (0.4 * w);
            int x = leftPoint.x - 0.25 * w;
            int y = leftPoint.y - 0.5 * h;

            //Extract ROI covering both eyes
            cout<<"\nAbout to extract ROI\n";
            frameROI = frame(Rect(x, y, w, h));
            cout<<"\nROI extracted...\n";

            //Resize the sunglasses image based on the dimensions
            //of the above ROI
            resize(eyeMask, eyeMaskSmall, Size(w, h));

            //Convert the previous image to grayscale
            cout<<"\nAbout to convert the previous image to gray scale...\n";
            cvtColor(eyeMaskSmall, grayMaskSmall, COLOR_BGR2GRAY);
            cout<<"\nConversion complete...\n";

            //Threshold the previous image to isolate the foreground object
            cout<<"\nIsolate the foreground objects...\n";
            threshold(grayMaskSmall, grayMaskSmallThresh, 245, 255, THRESH_BINARY_INV);
            cout<<"\nForeground objects isolation successful...\n";

            //Create mask by inverting the previous image
            //(because we don't want the background to affect the overlay)
            cout<<"\nInvert previous image to create mask...\n";
            bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv);
            cout<<"\nInversion complete...\n";

            //Use bitwise "AND" operator to extract precise boundary of sunglasses
            cout<<"\nExtract precise boundary of sunglasses...\n";
            bitwise_and(eyeMaskSmall, eyeMaskSmall, maskedEye, grayMaskSmallThresh);
            cout<<"\nExtraction completed...\n";

            //Use bitwise "AND" operator to overlay sunglasses
            cout<<"\nOverlay sunglasses...\n";
            bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv);
            cout<<"\nOverlay completed...\n";

            //Add the previously masked images and place it in the original
            //frame ROI to create the final image
            cout<<"\nReplace image in original frame...\n";
            add(maskedEye, maskedFrame, frame(Rect(x, y, w, h)));
            cout<<"\nReplaced successfully...\n";
        }

        //Show the current frame
        imshow("Frame", frame);

        //Get the keyboard input and check if it's 'Esc'
        //27 -> ASCII value for 'Esc' key
        ch = waitKey(30);
        if (ch == 27){
            break;
        }

    }
    //Release the video capture object
    cap.release();

    //Close all windows
    destroyAllWindows();
    return 1;
}