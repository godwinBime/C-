//
// Created by Godwin on 7/25/2022.
//
#include <opencv2/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);

const char* about
="This sample uses Single-Shot Detector"
 "(https://arxiv.org/abs/1512.02325)"
 "with ResNet-10 architecture to detect faces on camera/video/image.\n"
 "More information about the training is available here:"
 "<OPENCV_SRC_DIR>/samples/dnn/face_detector/how_to_train_face_detector.txt\n"
 ".caffemodel model's file is available here:"
 "<OPENCV_SRC_DIR>/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel\n"
 ".prototxt file is available here:"
 "<OPENCV_SRC_DIR>/samples/dnn/face_detector/deploy.prototxt\n";

const char* params
="{help | false | print usage}"
 "{proto || model configuration (deploy.prototxt)}"
 "{model || model weights (res10_300_ssd_iter_14000.caffemodel)}"
 "{camera_device | 0 | camera device number }"
 "{video || video or image for detection }"
 "{opencl | false | enable OpenCL}"
 "{min_confidence | 0.5 | min confidence}";

int main(int argc, char** argv){
    CommandLineParser parser(argc, argv, params);
    if (parser.get<bool>("help")){
        cout<<"\n\n" << about <<"\n\n";
        parser.printMessage();
        return 0;
    }

    String modelConfiguration = parser.get<string>("proto");
    String modelBinary = parser.get<string>("model");

    //![Initialize network]
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);

    //!{{Initialize network]
    if (net.empty()){
        cerr<<"\n\nCan't load network by using the following files...\n";
        cerr<<"prototxt: " << modelConfiguration <<"\n";
        cerr<<"caffemodel: " << modelBinary <<"\n";
        cerr<<"Models are available here: \n";
        cerr<<"<OPENCV_SRC_DIR>/samples/dnn/face_detector\n";
        cerr<<"Or here:\n";
        cerr<< "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector\n";
        exit(-1);
    } else{
        cout<<"Successfully loaded network...\n\n";
    }
    VideoCapture cap;
    if (parser.get<String>("video").empty()){
        cout<<"Inside video parser...\n\n";
        int cameraDevice = parser.get<int>("camera_device");
        cap = VideoCapture(cameraDevice);
        if (!cap.isOpened()){
            cout<<"\n\nCouldn't find camera: " << cameraDevice <<"\n\n";
            return -1;
        } else{
            cout<<"\n\nCamera opened successfully...\n\n";
        }
    } else{
        cap.open(parser.get<String>("video"));
        if (!cap.isOpened()){
            cout<<"\n\nCouldn't open image or video: " << parser.get<String>("video") <<"\n\n";
            return -1;
        } else{
            cout<<"Successfully opened image or video...\n\n";
        }
    }

    for(;;){
        Mat frame;
        cap >> frame; //Get a new frame from camera/video or read image
        if (frame.empty()){
            waitKey();
            break;
        }
        Mat inputBlob = blobFromImage(frame, inScaleFactor, Size(inWidth, inHeight), meanVal, false, false);
        net.setInput(inputBlob, "data");//Set the network input
        Mat detection = net.forward("detection_out");//compute output
        vector<double> layersTiming;
        double freq = getTickFrequency() / 1000;
        double times = net.getPerfProfile(layersTiming) / freq;
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        ostringstream ss;
        ss <<"FPS: " << 100 / times << " ; time: " << times << "ms\n";
        putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));

        float confidentThreshold = parser.get<float>("min_confidence");
        for(int i = 0; i < detectionMat.rows; i++){
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > confidentThreshold){
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                Rect object((int )xLeftBottom, (int )yLeftBottom, (int )(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));
                rectangle(frame, object, Scalar(0, 255, 0));

                ss.str("");
                ss << confidence;
                String conf(ss.str());
                String label = "Face: " + conf;
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height), Size(labelSize.width, labelSize.height + baseLine)), Scalar(255, 255, 255), FILLED);
                putText(frame, label, Point(xLeftBottom, yLeftBottom), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
            }
        }
        imshow("Detections", frame);
        if (waitKey(1) >= 0)
            break;
    }
    return 0;
}