//
// Created by Godwin on 7/22/2022.
//
#include "src/detection.cpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/text.hpp"

using namespace std;
using namespace cv;
using namespace cv::text;

Mat deSkewAndCrop(Mat input, const RotatedRect& box){
    Mat cropped;
    double angle = box.angle;
    auto size = box.size;

    //Adjust the boc angle
    if (angle < -45.0){
        angle += 90.0;
        std::swap(size.width, size.height);
    }

    //Rotate the text according to the angle
    auto transform = getRotationMatrix2D(box.center, angle, 1.0);
    Mat rotated;
    warpAffine(input, rotated, transform, input.size(), INTER_CUBIC);

    //Crop the result
    getRectSubPix(rotated, size, box.center, cropped);
    copyMakeBorder(cropped, cropped, 10, 10, 10, 10, BORDER_CONSTANT,Scalar(0));
    return cropped;
}

Mat drawER(const vector<Mat>& channels, const vector<vector<ERStat>>& regions,
           const vector<Vec2i>& group, const Rect& rect){
    Mat out = Mat::zeros(channels[0].rows + 2, channels[0].cols + 2, CV_8UC1);
    int flags = 4 + (255 << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
    for(int g = 0; g < group.size(); g++){
        int idx = group[g][0];
        auto er = regions[idx][group[g][1]];

        //Ignore root region
        if (er.parent == NULL)
            continue;

        int px = er.pixel % channels[idx].cols;
        int py = er.pixel / channels[idx].cols;

        //Create a point and add it to the list
        Point p(px, py);

        //Draw the extremal region
        floodFill(channels[idx], out, p, Scalar(255), nullptr , Scalar(er.level), Scalar(0), flags);
    }

    //Crop just the text area and find its points
    out = out(rect);
    vector<Point> points;
    findNonZero(out, points);

    //Use deSke and crop to crop it perfectly
    return deSkewAndCrop(out, minAreaRect(points));
}

cv::Ptr<BaseOCR> initOCR(const string& ocr){
    if (ocr == "hmm"){
        Mat transitions;
        FileStorage fs("OCRHMM_transitions_tables.xml", FileStorage::READ);
        fs["transition_probabilities"] >> transitions;
        fs.release();
        return OCRHMMDecoder::create(
                loadOCRHMMClassifierNM("OCRHMM_knn_model_data.xml.gz"),
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", transitions,
                Mat::eye(62, 62, CV_64FC1));
    } else if (ocr == "tesseract" || ocr == "tess"){
        return OCRTesseract::create(nullptr, "eng+por");
    }

    throw string("Invalid OCR Engine: ") + ocr;
}


int main(int argc, const char* argv[]) {
    const char *image = argc < 2 ? "easel.jpg" : argv[1];
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
    cout << "\n\nProcessing " << channels.size() << " channels...\n\n";
    vector<vector<ERStat>> regions(channels.size());
    for (int c = 0; c < channels.size(); c++) {
        cout << "\n\n   Channel  " << (c + 1) << "\n\n";
        filter1->run(channels[c], regions[c]);
        filter2->run(channels[c], regions[c]);
    }
    filter1.release();
    filter2.release();

    //Separate character groups from regions
    vector<vector<Vec2i>> groups;
    vector<Rect> groupRects;
    erGrouping(input, channels, regions, groups, groupRects, ERGROUPING_ORIENTATION_HORIZ);

    //Text detection
    cout <<"\n\nDetect Text: ";
    cout<<"-------------------\n\n";

    auto ocr = initOCR("tesseract");
    for(int i = 0; i < groups.size(); i++){
        auto wordImage = drawER(channels, regions, groups[i], groupRects[i]);
        string word;
        ocr->run(wordImage, word);
        cout<<word <<endl;
    }
}
