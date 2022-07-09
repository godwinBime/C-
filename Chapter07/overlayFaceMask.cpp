//
// Created by Godwin on 7/4/2022.
//

#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int mainn(int argc, char *argv[]) {
    string faceCascadeName = argv[1];
    float scalingFactor = 0.75;
    CascadeClassifier faceCascade;

    if (!faceCascade.load(faceCascadeName)){
        cerr << "\n\nError loading cascade file. Exiting ...\n\n";
        return -1;
    }else{
        cout <<"\n\nLoaded cascade (xml) file successfully...\n\n";
    }

    Mat faceMask = imread(argv[2]);
    if (!faceMask.data){
        cerr <<"\n\nError loading mask image. Exiting...\n\n";
    }else{
        cout <<"\n\nMask image loaded successfully...\n\n";
    }

    //Current frame
    Mat frame, frameGray;
    Mat frameROI, faceMaskSmall;
    Mat grayMaskSmall, grayMaskSmallThresh, grayMaskSmallThreshInv;
    Mat maskedFace, maskedFrame;

    vector<Rect> faces;

    /**Create the capture object.
     * 0 -> input arg that specifies it should take the input from the webcam**/
    VideoCapture cap(0);

    //if you cannot open the webcam, stop the execution
    if (!cap.isOpened()){
        cerr <<"\n\nUnable to open webcam. Exiting...\n\n";
        return -1;
    }else{
        cout <<"\n\nWebcam opened successfully...\n\n";
    }

    //Create GUI window
    namedWindow("Frame");

    //Iterate until the user presses the Esc key
    while (true){
        //Capture the current frame
        cap >> frame;

        //Resie the frame
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        //Convert to grayscale
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);

        //Equalize the histogram
        equalizeHist(frameGray, frameGray);

        //Detect faces
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 3, 0, Size(30, 30));
        //faceCascade.detectMultiScale(frameGray, faces, 1.1, 3, 0, Size(30, 30), Size(200,200));

        cout <<"\n\nFaces detected successfully\n\n";

        cout <<"\n\n====Faces length: " << faces.size() <<"\n\n";

        //Draw green rectangle around the face
        for(auto& face: faces){
            cout <<"\n\nInside for-loop\n\n";
            Rect faceRect(face.x, face.y, face.width, face.height);

            /**Custom parameters to make the mask fit your face. You may
             * have to play around with them to make sure it works.**/
            int x = face.x - int(0.1 * face.width);
            int y = face.y - int(0.0 * face.height);
            int w = int(1.1 * face.width);
            int h = int(1.3 * face.height);

            cout << "\n\nAfter customizing mask parameters to fit on face...\n\n";

            //Extract region of interest (ROI) covering your face
            if (0 <= face.x && 0 <= face.width && face.x + face.width <= frame.cols
            && 0 <= face.y && 0 <= face.height && face.y + face.height < frame.rows) {
                cout<<"\n\nAbout to extract ROI covering my face...\n\n";
                frameROI = frame(Rect(x, y, w, h));
                cout << "\n\nAfter extracting ROI\n\n";
            }else{
                cout<<"\n\nOut of ROI...\n\n";
                return -1;
            }

            //Resize the face mask image based on the dimensions
            //of the above ROI
            resize(faceMask, faceMaskSmall, Size(w, h));

            cout <<"\n\nAfter resizing face mask\n\n";

            //Convert the previous image to grayscale
            cvtColor(faceMaskSmall, grayMaskSmall, COLOR_BGR2GRAY);

            cout <<"\n\nAfter converting previous image to grayscale...\n\n";

            /**Threshold the previous image to isolate the pixels
             * associated only with the face mask.**/
            threshold(grayMaskSmall, grayMaskSmallThresh, 230, 255, THRESH_BINARY_INV);

            cout<<"\n\nAfter Threshold....\n\n";

            /**Create mask by inverting the previous image
             * (because we don't want the background to affect the overlay)**/
            bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv);

            cout <<"\n\nAfter creating mask...\n\n";

            //Use bitwise 'AND' operator to extract precise boundary of face mask
            bitwise_and(faceMaskSmall, faceMaskSmall, maskedFace, grayMaskSmallThresh);

            cout<<"\n\nAfter using bitwise AND to extract...\n\n";

            //Use bitwise 'AND' to overlay face mask
            bitwise_and(frameROI, frameROI, maskedFace, grayMaskSmallThreshInv);

            cout<<"\n\nAfter using bitwise AND to overlay...\n\n";

            //Add the previously masked images and place it in the original frame ROI
            //to create the final image.
            if (x > 0 && y > 0 && x + w < frame.cols && y + h < frame.rows) {
                cout <<"\n\nMask values are greater than zero...\n\n";
                add(maskedFace, maskedFrame, frame(Rect(x, y, w, h)));
                cout <<"\n\nPlaced added image in original frame ROI\n\n";
            }else{
                cout<<"\n\nPreviously masked images could not be added...\n\n";
                return -1;
            }
        }

        cout<<"\n\nOut of for-loop\n\n";

        //Show current frame
        imshow("Frame", frame);

        cout<<"\n\nSuccessfully displayed current frame...\n\n";

        //Get the keyboard input and check if it's 'Esc'
        //27 -> ASCII value of 'Esc' key
        auto ch = waitKey(30);
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

