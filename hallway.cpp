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

void openfile() {
    IplImage *srcImg = cvLoadImage("/Users/xx/Documents/school/vision/project/vision/vision/ushall1.jpg", CV_LOAD_IMAGE_COLOR);
    
    IplImage *src8bitgray = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1); 
    IplImage *cannyImg = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1); 

    IplImage *dstImg = cvCreateImage(cvGetSize(srcImg), 8, 1);
    IplImage *dstColorImg = cvCreateImage( cvGetSize(srcImg), 8, 3 );
    
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq *lines = 0;
    
    cvCvtColor(srcImg, src8bitgray, CV_BGR2GRAY); 
    cvCanny(src8bitgray, cannyImg, 100, 200); 
    
    cvCvtColor(cannyImg, dstColorImg, CV_GRAY2BGR);
    
    lines = cvHoughLines2(dstImg,
                          storage,
                          CV_HOUGH_STANDARD,
                          1,
                          CV_PI/180,
                          100,
                          0,
                          0 );
    
    for(int i = 0; i < MIN(lines->total,100); i++ ) {
        float *line = (float*)cvGetSeqElem(lines,i);
        float rho = line[0];
        float theta = line[1];
        CvPoint pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        cvLine(dstColorImg, pt1, pt2, CV_RGB(255,0,0), 3, 8 );
    }
    
    cvNamedWindow("Source", 1);
    cvShowImage("Source", srcImg);
    
    cvNamedWindow("Canny", 1);
    cvShowImage("Canny", dstColorImg);
    
    cvNamedWindow("Hough", 1);
    cvShowImage("Canny", dstColorImg);
    
    cvWaitKey();
}