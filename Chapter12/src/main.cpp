#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace dnn;

//Initialize parameters
float confThreshold = 0.5;//Confidence threshold
float nmsThreshold = 0.4; //Non-max Suppression threshold
int inpWidth = 416;//Width of the network's input image
int inpHeight = 416;//Height of network's input image
vector<string> classes;

//Draw the predicted boundary box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame){
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 255, 255), 1);

    //Get the label for the class name and its confidence
    string confLabel = format("%.2f", conf);
    string label = "";
    if (!classes.empty())
        label = classes[classId] + ":" + confLabel;

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_AA);
}

//Remove the bounding boxes with low confidence using non-maxima suppression
void postProcess(Mat& frame, const vector<Mat>& outs){
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for(size_t i = 0; i < outs.size(); ++i){
        /**Scan through all the bounding boxes output
         * from the network and keep only the ones with
         * the highest score for the box.**/
         auto *data = (float*)outs[i].data;
         for(int j = 0; j < outs[i].rows; ++j, data += outs[i].cols){
             Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
             Point classIdPoint;
             double confidence;

             //Get the value and location of the maximum score
             minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
             if(confidence > confThreshold){
                 int centerX = (int)(data[0] * frame.cols);
                 int centerY = (int)(data[1] * frame.rows);
                 int width = (int )(data[2] * frame.cols);
                 int height = (int )(data[3] * frame.rows);
                 int left = centerX - width / 2;
                 int top = centerY - height / 2;

                 classIds.push_back(classIdPoint.x);
                 confidences.push_back((float )confidence);
                 boxes.push_back(Rect(left, top, width, height));
             }
         }
    }

    /**
     * Perform non maximum suppression to eliminate redundant
     * overlapping boxes with lower confidences**/
     vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for(size_t i = 0; i < indices.size(); ++i){
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
    }
}

//Get the names of the output layers
vector<String> getOutputNames(const Net& net){
    static vector<String> names;
    if (names.empty()){
        //Get the indices of the output layers, i.e. the layers
        //with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //Get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        //Get the names of the output layers in names
        names.resize(outLayers.size());
        for(size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}


int main_(int argc, char **argv) {
    //Load names of classes
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line))
        classes.push_back(line);

    //Give the configuration and weight files
    string modelConfiguration = "yolov3.cfg";
    string modelWeights = "yolov3.weights";

    //Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    Mat input, blob;
    input = imread(argv[1]);

    //Stop the program if reached end of video
    if (input.empty()){
        cout<<"\n\nNo input image...\n\n";
        return 0;
    }

    //Create a 4D blob from a frame
    blobFromImage(input, blob, 1 / 255.0, Size(inpWidth, inpHeight),
                  Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    //Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, getOutputNames(net));

    //Remove the bounding boxes with low confidence
    postProcess(input, outs);

    /**Put efficiency information. The function 'getPerfProfile()'
     * returns the overall time for inference(t) and the timings
     * for each of the layers (in layersTimes)**/
     vector<double> layersTime;
     double freq = getTickFrequency() / 1000;
     double t = net.getPerfProfile(layersTime) / freq;
     string label = format("Inferences time to compute the image: %.2f ms", t);
     cout<<"\n\nLabel \n\n";
    imshow("Deep Learning. Chapter 12", input);
    imwrite("result.jpg", input);
    waitKey(0);
    return 0;
}
