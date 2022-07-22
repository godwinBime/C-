//
// Created by Godwin on 7/22/2022.
//
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/text.hpp"

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::text;

vector<Mat> separateChannels(const Mat& src){
    vector<Mat> channels;

    //Gray scale images
    if (src.type() == CV_8U || src.type() == CV_8UC1){
        channels.push_back(src);
        channels.push_back(255 - src);
        return channels;
    }

    //Colored images
    if (src.type() == CV_8UC3){
        computeNMChannels(src, channels);
        int size = static_cast<int>(channels.size()) - 1;
        for(int c = 0; c < size; c++)
            channels.push_back(255 - channels[c]);
        return channels;
    }

    //Other types
    cout<<"\n\nInvalid image format!\n\n";
    exit(-1);
}

/**
int main(int argc, const char* argv[]){
    const char* image = argc < 2 ? "easel.jpg" : argv[1];
    auto input = imread(image);
    Mat processed;
    cvtColor(input, processed, COLOR_BGR2GRAY);
    auto channels = separateChannels(processed);

    //Create ERFilter objects with the 1st and 2nd stage classifiers
    auto filter1 = createERFilterNM1(
            loadClassifierNM1("trained_classifierNM1.xml"), 15, 0.00015f, 0.13f, 0.2f, true, 0.1f);

    auto filter2 = createERFilterNM2(
            loadClassifierNM2("trained_classifierNM2.xml"), 0.5);

    //Extract text regions using Newmann and Matas algorithm
    cout<<"\n\nProcessing " << channels.size() <<" channels...\n\n";
    vector<vector<ERStat>> regions(channels.size());
    for(int c = 0; c < channels.size(); c++){
        cout<<"\n\n   Channel  " << (c + 1) <<"\n\n";
        filter1->run(channels[c], regions[c]);
        filter2->run(channels[c], regions[c]);
    }
    filter1.release();
    filter2.release();

    //Separate character groups from regions
    vector<vector<Vec2i>> groups;
    vector<Rect> groupRects;
    erGrouping(input, channels, regions, groups, groupRects, ERGROUPING_ORIENTATION_HORIZ);


    //Draw group boxes
    for(const auto& rect : groupRects)
        rectangle(input, rect, Scalar(0, 255, 0), 3);

    imshow("Grouping", input);
    waitKey(0);
    return 0;
}
 **/
