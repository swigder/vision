//
//  hallway.cpp
//  vision
//
//  Created by Suri on 2012-04-29.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "hallway.h"
using namespace cv;

void hallway() {
    IplImage *srcImg = cvLoadImage("/Users/xx/Documents/school/vision/project/vision/vision/ushall1.jpg", CV_LOAD_IMAGE_COLOR);
    
    // need grayscale, 8-bit image for canny and hough
    IplImage *src8bitgray = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1); 
    
    // store the canny edges
    IplImage *cannyImg = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1);
    
    // store the image with lines overlaid
    IplImage *houghColorImg = cvCreateImage(cvGetSize(srcImg), 8, 3);
    cvCopy(srcImg, houghColorImg);
    
    // need grayscale, 8-bit image for canny and hough
    cvCvtColor(srcImg, src8bitgray, CV_BGR2GRAY);
    
    // canny edge detection
    cvCanny(src8bitgray, cannyImg, 210, 255); 

    // hough to get lines
    CvSeq *lines = hough(cannyImg, houghColorImg);
    
    // get vanishing point
    CvMat *point = vanishing(lines, houghColorImg);
    
    // display what we've done
    cvNamedWindow("Source", 1);
    cvShowImage("Source", srcImg);
    
    cvNamedWindow("Canny", 1);
    cvShowImage("Canny", cannyImg);
    
    cvNamedWindow("Hough", 1);
    cvShowImage("Hough", houghColorImg);
    
    cvWaitKey();
}

CvSeq *hough(IplImage *src, IplImage *dst) {
    CvMemStorage* storage = cvCreateMemStorage(0);
    
    CvSeq *lines = cvHoughLines2(src,
                          storage,
                          CV_HOUGH_STANDARD,
                          1,
                          CV_PI/180,
                          100,
                          0,
                          0);
    
    for(int i = 0; i < MIN(lines->total,100); i++ ) {
        float *line = (float*)cvGetSeqElem(lines,i);
        float rho = line[0];
        float theta = line[1];
        
        // let's remove those lines that are obviously not hallway
        if (theta < M_PI / 6 || theta > 5 * M_PI / 6) {
            cvSeqRemove(lines, i);
            continue;
        }
        
        CvPoint pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        cvLine(dst, pt1, pt2, CV_RGB(255,0,0), 2, 8);
    }
    
    return lines;
}

CvMat *vanishing(CvSeq *lines, IplImage *dst) {
    CvMat *A = cvCreateMat(2, 2, CV_64FC1);
    CvMat *A1 = cvCreateMat(2, 2, CV_64FC1);
    CvMat *b = cvCreateMat(2, 1, CV_64FC1);
    CvMat *x = cvCreateMat(2, 1, CV_64FC1);
    
    // cluster to get two predominant lines
    for (int i = 0; i < 2; i++) {
        float *line = (float*)cvGetSeqElem(lines, i);
        float rho = line[0];
        float theta = line[1];
                        
        cvmSet(A, i, 0, cos(theta));
        cvmSet(A, i, 1, sin(theta));
        cvmSet(b, i, 0, rho);
    }
    
    // find the intersection of the lines
    cvInvert(A, A1);
    cvMatMul(A1, b, x);
    
    double x0 = cvmGet(x, 0, 0);
    double y0 = cvmGet(x, 1, 0);
    
    cout << x0 << endl << y0 << endl;
    
    cvCircle(dst, cvPoint(x0, y0), 10, CV_RGB(0,255,0));

    return x;
}