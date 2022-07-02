#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <memory>
#include "utils/MultipleImageWindow.h"

//OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv::ml;

shared_ptr<MultipleImageWindow> miw;
Ptr<SVM> svm;
Mat lightPattern;

Scalar green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255);

/**OpenCV command line parser function keys accepted by
 * command line parser.**/
 const char* keys = {
         "{help h usage? || print this message}"
         "{@image || Image to classify.}"
 };

 static Scalar randomColor(RNG& rng){
     int iColor = (unsigned )rng;
     return Scalar (iColor&255, (iColor >> 8)&255, (iColor >> 16)&255);
 }

/**
 * Extract the features for all objects in one image.
 * @param Mat img input image
 * @param vector<int> left output of the left coordinates
 * for each object.
 * @param vector<int> top output of top coordinates for each object
 * @param vector<vector<float>> a matrix of raws of features for each
 * object detected.
 * **/
vector<vector<float>> extractFeatures(Mat img, vector<int>* left = NULL, vector<int>* top = NULL){
    vector<vector<float>> output;
    vector<vector<Point>> contours;
    Mat input = img.clone();

    vector<Vec4i> hierarchy;

    findContours(input, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    //Check the number of methods detected
    if (contours.size() == 0){
        return output;
    }

    RNG rng(0xFFFFFFFF);
    for(auto i = 0; i < contours.size(); i++){
        Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
        drawContours(mask, contours, i, Scalar(1), FILLED, LINE_8, hierarchy, 1);
        Scalar areaS = sum(mask);
        float area = areaS[0];

        if (area > 500) {//if the are is greater than the min
            RotatedRect r = minAreaRect(contours[i]);
            float width = r.size.width;
            float height = r.size.height;
            float ar = (width < height) ? height / width : width / height;

            vector<float> row;
            row.push_back(area);
            row.push_back(ar);
            output.push_back(row);

            if (left != NULL) {
                left->push_back((int) r.center.x);
            }

            if (top != NULL) {
                top->push_back((int) r.center.y);
            }

            //Add image to the multiple image window class, See the class
            //on full GitHub code
            miw->addImage("Extract Features", mask * 255);
            miw->render();
            waitKey(10);
        }
    }
    return output;
}

/**Remove the light and return new image without light.
 * @param img Mat image to remove the light pattern
 * @param pattern Mat img with light pattern.
 * @return a new image Mat without light.**/
Mat removeLight(Mat img, Mat pattern){
    cout<<"\n\n\nRemove light initiated in removeLight(...)\n\n\n";
    Mat aux;

    //Require change our image to 32 float for division
    Mat img32, pattern32;

    cout<<"\n\n\nConvert to 32 float for division...\n\n\n";
    img.convertTo(img32, CV_32F);
    pattern.convertTo(pattern32, CV_32F);

    cout<<"\n\n\nSuccessfully converted images to 32 float\n\n\n";

    //Divide the image by the pattern
    cout<<"\n\n\nDivide the img32 by pattern32...\n\n\n";
    aux = 1 - (img32 / pattern32);

    //Scale it to convert to 8 bit format
    cout<<"\n\n\nScale the image...\n\n\n";
    aux = aux * 255;

    //Convert 8 bits format
    aux.convertTo(aux, CV_8U);

    return aux;
}

/**Preprocess an input image to extract components and stats
 * @param Mat input image to preprocess
 * @param Mat binary image**/

Mat preProcessImage(Mat input){
    cout<<"\n\n\nPreprocess image received...\n\n\n";
    Mat result;
    if (input.channels() == 3){
        cout<<"\n\n\nWe have 3 channels...\n\n\n";
        cvtColor(input, input, COLOR_RGB2GRAY);
        cout<<"\n\n\nExiting if statement in preProcessImage(...)\n\n\n";
    } else{
        cout <<"\n\n\nError: inconsistent input channels...\n\n\n";
    }

    //Remove noise
    Mat imgNoise, imgBoxSmooth;

    cout<<"\n\n\nRemove noise...preProcessImage(...)\n\n\n";
    medianBlur(input, imgNoise, 3);

    //Apply the light pattern
    Mat imgNoLight;
    cout<<"\n\n\nCopy image to remove light...\n\n\n";
    imgNoise.copyTo(imgNoLight);
    cout<<"\n\nParse image for light to be removed...\n\n\n";
    imgNoLight = removeLight(imgNoise, lightPattern);
    cout<<"\n\n\nLight successfully removed\n\n\n";

    //Binarize image for segment
    threshold(imgNoLight, result, 30, 255, THRESH_BINARY);
    return result;
}

bool readFolderAndExtractFeatures(string folder, int label, int numForTest,
                                  vector<float> &trainingData, vector<int> &responsesData,
                                  vector<float> &testData,
                                  vector<float> &testResponsesData){
    VideoCapture images;
    if(!images.open(folder)){
        cout <<"\n\n\nCannot open the folder images\n\n\n";
        return false;
    } else{
        cout << "\n\nSuccessfully opened image folder....\n\n\n";
    }

    Mat frame;
    int imageIndex = 0;

    while (images.read(frame)){
        cout<<"\n\n\nInside while loop...\n\n";
        //Preprocess image
        cout<<"\n\n\nGetting preprocess image from readFolderAndExtractFeatures(...)\n\n\n";
        Mat pre = preProcessImage(frame);

        //Extract features
        cout<<"\n\n\nExtract features";
        vector<vector<float>> features = extractFeatures(pre);

        cout << "\n\n\nFeatures extracted successfully";
        for(auto i = 0; i < features.size(); i++){
            cout <<"\n\n\nInside For loop\n\n";
            if(imageIndex >= numForTest){
                trainingData.push_back(features[i][0]);
                trainingData.push_back(features[i][1]);
                responsesData.push_back(label);
            } else{
                testData.push_back(features[i][0]);
                testData.push_back(features[i][1]);
                testResponsesData.push_back((float) label);
            }
        }
        cout <<"\n\n\nOut of for loop";
        imageIndex++;
    }
    cout <<"\n\n\nEnd of while loop";
    return true;
}

void plotTrainData(Mat trainingData, Mat labels, float *error = NULL){
    float areaMax, arMax, areaMin, arMin;
    areaMax = arMax = 0;
    areaMin = arMin = 99999999;

    //Get the min of each feature for normalize plot image
    for(int i = 0; i < trainingData.rows; i++){
        float area = trainingData.at<float>(i, 0);
        float ar = trainingData.at<float>(i, 1);

        if (area > areaMax)
            areaMax = area;

        if(ar > arMax)
            areaMax = area;

        if(ar > arMax)
            arMax = ar;

        if (area < areaMin)
            areaMin = area;

        if (ar < arMin)
            arMin = ar;
    }

    //Create image for plot
    Mat plot = Mat::zeros(512, 512, CV_8UC3);

    /**Plot each of two features in a 2D graph using an image
     * where x is area and y is aspect ratio**/
     for(int i = 0; i < trainingData.rows; i++){
         //Set the x y pos for each data
         float area = trainingData.at<float>(i, 0);
         float ar = trainingData.at<float>(i, 1);

         int x = (int)(512.0f * ((area - areaMin) / (areaMax - areaMin)));
         int y = (int)(512.0f * ((ar - arMin) / (arMax - arMin)));

         //Get label
         int label = labels.at<int>(i);

         //Set color depend of label
         Scalar color;
         if (label == 0) {
             color = green; //NUT
         }else if(label == 1) {
             color = blue; //RING
         }else if (label == 2) {
             color = red; //SCREW
         }
         circle(plot, Point(x, y), 3, color, -1, 8);
     }

    if (error != NULL){
        stringstream ss;
        ss << "Error: " << *error << "\t";
        putText(plot, ss.str().c_str(), Point(20, 512 - 40), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(200, 200, 200), 1,
                LINE_AA);
    }
    miw->addImage("Plot", plot);
}
void trainAndTest(){
    vector<float> trainingData;
    vector<int> responsesData;
    vector<float> testData;
    vector<float> testResponsesData;
    int numForTest = 20;
    cout <<"\n\n\nGetting training data\n\n\n";

    //Get the nut images
    readFolderAndExtractFeatures("data/nut/tuerca_%04d.pgm", 0, numForTest, trainingData, responsesData, testData,
                                 testResponsesData);

    cout <<"\n\n\nSuccessfully read NUTS...";

    //Get and process the ring images
    readFolderAndExtractFeatures("data/ring/arandela_%04d.pgm", 1, numForTest, trainingData, responsesData, testData,
                                 testResponsesData);

    cout <<"\n\n\nSuccessfully read RINGS...";
    //Get and process the screw images
    readFolderAndExtractFeatures("data/screw/tornillo_%04d.pgm", 2, numForTest, trainingData, responsesData, testData,testResponsesData);

    cout<<"\n\n\nSuccessfully read SCREWS";

    cout <<"\n\n\nNum of train samples: " << responsesData.size() << endl;
    cout <<"\n\n\nNum of test samples: " << testResponsesData.size() <<endl;

    //Merge all data
    Mat trainingDataMat(trainingData.size() / 2, 2, CV_32FC1, &trainingData[0]);
    Mat responses(responsesData.size(), 1, CV_32SC1, &responsesData[0]);
    Mat testDataMat(testData.size() / 2, 2, CV_32FC1, &testData[0]);
    Mat testResponses(testResponsesData.size(), 1, CV_32FC1,&testResponsesData[0]);

    Ptr<TrainData> tData = TrainData::create(trainingDataMat, ROW_SAMPLE, responses);

    //Set up SVM's parameters
    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setNu(0.05);
    svm->setKernel(cv::ml::SVM::CHI2);
    svm->setDegree(1.0);
    svm->setGamma(2.0);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    //Train the SVM
    svm->train(tData);

    if(testResponsesData.size() > 0){
        cout <<"\n\n\nEvaluation" << endl;
        cout << "==================\n";

        //Test the machine learning model
        Mat testPredict;
        svm->predict(testDataMat, testPredict);
        cout <<"\n\nPrediction Done!!";

        //Error calculation
        Mat errorMat = testPredict != testResponses;
        float error = 100.0f * countNonZero(errorMat) / testResponsesData.size();
        cout <<"\n\n\nError: " << error << "%\n";

        //Plot training data with error label
        plotTrainData(trainingDataMat, responses, &error);
    } else{
        plotTrainData(trainingDataMat, responses);
    }
}

int main(int argc, const char** argv){
    CommandLineParser parser(argc, argv, keys);
    parser.about("Chapter 06. Classification v1.0.0");

    //If requires help, show
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }

    String imgFile = parser.get<String>(0);
    String lightPatternFile = "data/pattern.pgm";

    //Check if params are correctly parsed in the variables
    if (!parser.check()){
        cout <<"\n\n\nERROR: see details below\n\n\n";
        parser.printErrors();
        return 0;
    }

    //Create the multiple image Window
    miw = make_shared<MultipleImageWindow>("Main Window", 2, 2, WINDOW_AUTOSIZE);

    //Load image to process
    Mat img = imread(imgFile, 0);
    if (img.data == NULL){
        cout <<"\n\n\nError loading image "<< imgFile;
        return 0;
    } else{
        cout <<"\n\n\nimg loaded successfully";
    }

    Mat imgOutput = img.clone();
    cvtColor(imgOutput, imgOutput, COLOR_GRAY2BGR);

    //Load image to process
    lightPattern = imread(lightPatternFile, 0);
    if (lightPattern.data == NULL){
        //Calculate light pattern
        cout << "\n\n\nERROR: No light pattern loaded";
        return 0;
    } else{
        cout <<"\n\n\nlightPattern File exist....";
    }

    cout <<"\n\n\nmedianBlur.................";
    medianBlur(lightPattern, lightPattern, 3);

    cout <<"\n\n\nget train and test data";
    trainAndTest();

    cout <<"\n\n\nTrain data received...\n\n";

    //Preprocess image
    cout<<"\n\n\nGetting preprocess image from main()\n\n\n";
    Mat pre = preProcessImage(img);

    cout<<"\n\n\nImage preprocessing complete from main()\n\n\n";
    //End processing

    //Extract features
    vector<int> posTop, posLeft;
    vector<vector<float>> features = extractFeatures(pre, &posLeft, &posTop);
    cout<<"\n\n\nNum of objects extracted features: " << features.size() << "\n\n\n";
    for(int i = 0; i < features.size(); i++){
        cout <<"\n\n\nData Area AR: " << features[i][0] << " " << features[i][1];
        Mat trainingDataMat(1, 2, CV_32FC1, &features[i][0]);
        cout <<"\n\n\nFeatures to predict: " << trainingDataMat;
        float result = svm->predict(trainingDataMat);
        cout <<"\n\n\nResult: " << result <<"\n\n\n";

        stringstream ss;
        Scalar  color;
        if (result == 0){
            color = green;
            ss << "NUT";
        } else if (result == 1){
            color = blue;
            ss << "RING";
        } else if (result == 2){
            color = red;
            ss << "SCREW";
        }
        putText(imgOutput, ss.str(), Point2d(posLeft[i], posTop[i]), FONT_HERSHEY_SIMPLEX, 0.4, color);
    }

    //Show images
    miw->addImage("Binary image", pre);
    miw->addImage("Result", imgOutput);
    miw->render();
    waitKey(0);
    return 0;
}