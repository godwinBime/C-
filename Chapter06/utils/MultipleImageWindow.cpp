//
// Created by Godwin on 6/27/2022.
//
// Chpater05.cpp : Defines the entry point for the application.
//

#include "MultipleImageWindow.h"

MultipleImageWindow::MultipleImageWindow(const string& windowTitle, int cols, int rows, int flags) {
    this->windowTitle = windowTitle;
    this->cols = cols;
    this->rows = rows;
    namedWindow(windowTitle, flags);

    //TODO: detect resolution of desktop and show full resolution canvas
    this->canvasWidth = 1200;
    this->canvasHeight = 700;
    this->canvas = Mat(this->canvasHeight, this->canvasWidth, CV_8UC3);
    imshow(this->windowTitle, this->canvas);
}

int MultipleImageWindow::addImage(const string& title, Mat image, bool render) {
    this->titles.push_back(title);
    this->images.push_back(image);

    if (render) {
        this->render();
    }
    return this->images.size() - 1;
}

void MultipleImageWindow::removeImage(int pos) {
    this->titles.erase(this->titles.begin() + pos);
    this->images.erase(this->images.begin() + pos);
}

void MultipleImageWindow::render() {
    //Clean our canvas
    this->canvas.setTo(Scalar(20, 20, 20));

    //Width and height of cell. Add 10 px of padding between images
    int cellWidth = (canvasWidth / cols);
    int cellHeight = (canvasHeight / rows);
    int margin = 10;
    int maxImages = (this->images.size() > cols * rows) ? cols * rows : this->images.size();
    int i = 0;

    auto titlesIterator = this->titles.begin();
    for (auto img : this->images) {
        string title = *titlesIterator;
        int cellX = (cellWidth) * ((i) % cols);
        int cellY = (cellHeight)*floor((i) / (float)cols);
        Rect mask(cellX, cellY, cellWidth, cellHeight);

        //Draw a rectangle for each cell mat
        rectangle(canvas, Rect(cellX, cellY, cellWidth, cellHeight), Scalar(200, 200, 200), 1);

        //For each cell draw an image if exists
        Mat cell(this->canvas, mask);

        //Resize image to cell size
        Mat resized;
        double cellAspect = (double)cellWidth / (double)cellHeight;
        double imgAspect = (double)img.cols / (double)img.rows;
        double f = (cellAspect < imgAspect) ? (double)cellWidth / (double)img.cols : (double)cellHeight / (double)img.rows;
        resize(img, resized, Size(0, 0), f, f);

        if (resized.channels() == 1) {
            cvtColor(resized, resized, COLOR_GRAY2BGR);
        }

        //Assign the image
        Mat subCell(this->canvas, Rect(cellX, cellY, resized.cols, resized.rows));
        resized.copyTo(subCell);
        putText(cell, title.c_str(), Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 0, 0), 1, LINE_AA);
        i++;
        ++titlesIterator;

        if (i == maxImages)
            break;
    }

    //Show image
    imshow(this->windowTitle, this->canvas);

}

