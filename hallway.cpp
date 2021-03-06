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
//    IplImage *srcImg = cvLoadImage("hallway1.jpg", CV_LOAD_IMAGE_COLOR);    
    IplImage *srcImg = cvLoadImage("/Users/xx/Documents/school/vision/project/vision/vision/hallway9.jpg", CV_LOAD_IMAGE_COLOR);    
    
    // need grayscale, 8-bit image for canny and hough
    IplImage *src8bitgray = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1); 
    cvCvtColor(srcImg, src8bitgray, CV_BGR2GRAY);
    
    // store the image with lines overlaid
    IplImage *houghColorImg = cvCreateImage(cvGetSize(srcImg), 8, 3);
    cvCopy(srcImg, houghColorImg);
    
    // store the canny edges
    IplImage *cannyImg = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1);
    IplImage *cannyR = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1);
    IplImage *cannyG = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1);
    IplImage *cannyB = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1);
    IplImage *cannyCombined = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1);
        
    //Get color channels.
    IplImage *srcR = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1);
    IplImage *srcG = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1);
    IplImage *srcB = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1);
    cvSplit(srcImg, srcB, srcG, srcR, NULL);
    
    // canny edge detection
    cvCanny(src8bitgray, cannyImg, 50, 255);
    cvCanny(srcR, cannyR, 50, 255);
    cvCanny(srcG, cannyG, 50, 255);
    cvCanny(srcB, cannyB, 50, 255);
    
    // let's combine the three RGB canny Images
    cvAdd(cannyR, cannyG, cannyCombined);
    cvAdd(cannyCombined, cannyB, cannyCombined);

    // hough to get lines
    CvSeq *lines = hough(cannyImg, houghColorImg);
        
    // get vertical line segments
    CvSeq *vertlines = verticalLineSegments(cannyCombined, houghColorImg);
    
    // get vanishing point
    CvPoint vp = vanishing(lines, houghColorImg);
    
    // get lines that go through vp, and those intersecting vertical
    CvSeq *vpLines = linesThroughVp(lines, houghColorImg, &vp);
    CvSeq *vpVert = linesIntersectingSegmentsBelowVP(vpLines, vertlines, vp);

//    drawLinesLines(lines, houghColorImg, CV_RGB(0,255,0));
    drawLinesLines(vpLines, houghColorImg, CV_RGB(255,0,0));
    drawLinesPoints(vertlines, houghColorImg, CV_RGB(0,0,255));
    cvCircle(houghColorImg, vp, 10, CV_RGB(0,255,0));
//    drawLinesLines(vpVert, houghColorImg, CV_RGB(0,0,255));
    CvSeq* floorLines = getFloorEdges(vpVert, vp);
    drawLinesLines(floorLines, houghColorImg, CV_RGB(255,255,0), 2);
                
    // display what we've done
    cvNamedWindow("Source", 1);
    cvShowImage("Source", srcImg);
    
//    cvNamedWindow("Canny", 1);
//    cvShowImage("Canny", cannyImg);
    
    cvNamedWindow("Canny Combined", 1);
    cvShowImage("Canny Combined", cannyCombined);

    cvNamedWindow("Hough", 1);
    cvShowImage("Hough", houghColorImg);
    
//    cvNamedWindow("CannyR", 1);
//    cvShowImage("CannyR", cannyR);
//    
//    cvNamedWindow("CannyG", 1);
//    cvShowImage("CannyG", cannyG);
//    
//    cvNamedWindow("CannyB", 1);
//    cvShowImage("CannyB", cannyB);
    
    cvWaitKey();
}

# pragma mark - helper drawing functions

void drawLinesPoints(CvSeq *lines, IplImage *img, CvScalar color = CV_RGB(255,0,0)) {    
    for (int i = 0; i < lines->total; i++ ) {
        CvPoint* line = (CvPoint*)cvGetSeqElem(lines, i);
        cvLine(img, line[0], line[1], color, 1, 8 );
    }
}

void drawLinesLines(CvSeq *lines, IplImage *img, CvScalar color = CV_RGB(255,0,0), int lineWidth /* =1 */) {
    int total = lines->total;
    for (int i = 0; i < total; i++) {
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
        
        cvLine(img, pt1, pt2, color, lineWidth, 8);
    }    
}

void colorFloor(IplImage *img, CvScalar color, CvSeq *lines) {
    // there will always be TWO lines + image bottom (for now)
}

#pragma mark helper geometric functions

CvSeq *linesContainingPoints(CvSeq *lines, CvSeq *points) {
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq *retLines = cvCreateSeq(0, sizeof(CvSeq), sizeof(float) * 3, storage);
    
    int total = lines->total;
    for (int i = 0; i < total; i++) {
        float *line = (float*)cvGetSeqElem(lines,i);
        float rho = line[0];
        float theta = line[1];
        
        for (int j = 0; j < points->total; j++) {
            CvPoint* point = (CvPoint *)cvGetSeqElem(points,j);
            
            // check if line goes through point
            // don't forget to check for nan!
            double exy = -cos(theta) / sin(theta) * point->x + rho / sin(theta);
            if (exy == exy && (point->y - exy) < 5) {
                cvSeqPush(retLines, line);;
            }
        }
    }
        
    return retLines;
}

CvSeq *linesContainingPoint(CvSeq *lines, CvPoint *point) {
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq *retLines = cvCreateSeq(0, sizeof(CvSeq), sizeof(float) * 3, storage);
    
    int total = lines->total;
    for (int i = 0; i < total; i++) {
        float *line = (float*)cvGetSeqElem(lines,i);
        if (lineContainsPoint(line, *point, 5)) {
            cvSeqPush(retLines, line);;
        }
    }
    
    return retLines;
}

CvPoint lineThroughPoints(CvPoint *pt1, CvPoint *pt2) {
    CvMat *A = cvCreateMat(2, 2, CV_64FC1);
    CvMat *A1 = cvCreateMat(2, 2, CV_64FC1);
    CvMat *b = cvCreateMat(2, 1, CV_64FC1);
    CvMat *x = cvCreateMat(2, 1, CV_64FC1);
    
    cvmSet(A, 0, 0, pt1->x);
    cvmSet(A, 0, 1, 1);
    cvmSet(A, 1, 0, pt2->x);
    cvmSet(A, 1, 1, 1);
    cvmSet(b, 0, 0, pt1->y);
    cvmSet(b, 1, 0, pt2->y);
    
    cvInvert(A, A1);
    cvMatMul(A1, b, x);
    
    double cossint = cvmGet(x, 0, 0);
    double rsint = cvmGet(x, 1, 0);
    
    double theta = atan(1/cossint);
    double rho = rsint * sin(theta);
    
    return cvPoint(int(rho), int(theta));
}

bool lineContainsPoint(float *line, CvPoint point, int tolerance = 2) {
    float rho = line[0];
    float theta = line[1];
    
    double exy = -cos(theta) / sin(theta) * point.x + rho / sin(theta);
    return exy == exy && abs(point.y - exy) < tolerance;
}


# pragma mark - vision

CvSeq *hough(IplImage *src, IplImage *dst) {
    CvMemStorage* storage = cvCreateMemStorage(0);
    
    CvSeq *lines = cvHoughLines2(src,
                          storage,
                          CV_HOUGH_STANDARD,
                          1,
                          CV_PI/180,
                          75,
                          0,
                          0);
    
    return lines;
}

CvPoint vanishing(CvSeq *lines, IplImage *dst) { 
    CvSize size = cvGetSize(dst);
    int pixels[size.width][size.height];
    for (int i = 0; i < size.width; i++) {
        for (int j = 0; j < size.height; j++) {
            pixels[i][j] = 0;
        }
    }
    
    CvMat *A = cvCreateMat(2, 2, CV_64FC1);
    CvMat *A1 = cvCreateMat(2, 2, CV_64FC1);
    CvMat *b = cvCreateMat(2, 1, CV_64FC1);
    CvMat *x = cvCreateMat(2, 1, CV_64FC1);
    
    int total = lines->total;
    for (int i = 0; i < total; i++) {
        for (int j = i + 1; j < total; j++) {
            float *line1 = (float*)cvGetSeqElem(lines, i);
            float *line2 = (float*)cvGetSeqElem(lines, j);
            float rho1 = line1[0];
            float theta1 = line1[1];
            float rho2 = line2[0];
            float theta2 = line2[1];
            
            cvmSet(A, 0, 0, cos(theta1));
            cvmSet(A, 0, 1, sin(theta1));
            cvmSet(A, 1, 0, cos(theta2));
            cvmSet(A, 1, 1, sin(theta2));
            cvmSet(b, 0, 0, rho1);
            cvmSet(b, 1, 0, rho2);
            
            cvInvert(A, A1);
            cvMatMul(A1, b, x);
            
            double x0 = cvmGet(x, 0, 0);
            double y0 = cvmGet(x, 1, 0);
            
            if (x0 > 0 && x0 < size.width && y0 > 0 && y0 < size.height) {
                pixels[int(x0)][int(y0)]++;
            }
        }
    }
    
    CvPoint vp;
    int max = -1;
    int tmpmax = 0;
    for (int i = 0; i < size.width - 5; i+=5) {
        for (int j = 0; j < size.height - 5; j+=5) {
            tmpmax = 0;
            for (int k = 0; k < 5; k++) {
                for (int l = 0; l < 5; l++) {
                    tmpmax += pixels[i+k][j+l];
                }
            }
            if (tmpmax > max) {
                max = tmpmax;
                vp = cvPoint(i + 3, j + 3);
            }
        }
    }
    
    cout << "max: " << max << endl;
    cout << "point: " << vp.x << ", " << vp.y << endl;
            
    return vp;
}

#pragma mark intersecting

CvSeq *linesIntersectingCorners(CvSeq *lines, IplImage *src) {
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq *retLines = cvCreateSeq(0, sizeof(CvSeq), sizeof(float) * 3, storage);
    
    int i, corner_count = 150;
    
    IplImage *eig_img, *temp_img;
    CvPoint2D32f *corners;
    
    eig_img = cvCreateImage (cvGetSize(src), IPL_DEPTH_32F, 1);
        temp_img = cvCreateImage(cvGetSize (src), IPL_DEPTH_32F, 1);
        corners = (CvPoint2D32f *)cvAlloc(corner_count * sizeof(CvPoint2D32f));
    
        cvGoodFeaturesToTrack (src, eig_img, temp_img, corners, &corner_count, 0.1, 15);
        cvFindCornerSubPix (src, corners, corner_count,
                        cvSize (3, 3), cvSize (-1, -1), cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
        
    IplImage *cannyImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
    cvCanny(src, cannyImg, 50, 255); 
    
    for (i = 0; i < corner_count; i++) {
        CvPoint point = cvPointFrom32f(corners[i]);
        if ((cvGet2D(cannyImg, point.y - 1, point.x).val[0] != 0 || cvGet2D(cannyImg, point.y + 1, point.x).val[0] != 0)) {            
            for (int k = 0; k < lines->total; k++) {
                float *line = (float*)cvGetSeqElem(lines,k);
                float rho = line[0];
                float theta = line[1];
                
                double exy = -cos(theta) / sin(theta) * point.x + rho / sin(theta);
                if (exy == exy && abs(point.y - exy) < 5) {
                    cvSeqPush(retLines, line);
                }
            }
        }
    }

    return retLines;
}

CvSeq *linesThroughVp(CvSeq *lines, IplImage *img, CvPoint *vp) {
    CvSeq *retLines = linesContainingPoint(lines, vp);
    
    cvCircle(img, *vp, 10, CV_RGB(0,255,0));
    
    return retLines;
}

CvSeq *linesIntersectingSegments(CvSeq *lines, CvSeq *segments) {
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq *retLines = cvCreateSeq(0, sizeof(CvSeq), sizeof(float) * 3, storage);
    
    for (int i = 0; i < lines->total; i++) {
        float *line = (float*)cvGetSeqElem(lines, i);

        for (int j = 0; j < segments->total; j++) {
            CvPoint* segment = (CvPoint*)cvGetSeqElem(segments, j);
            
            if (lineContainsPoint(line, segment[0], 5) || lineContainsPoint(line, segment[1], 5)) {
                cvSeqPush(retLines, line);
            }
        }
    }
    
    return retLines;
}

CvSeq *linesIntersectingSegmentsBelowVP(CvSeq *lines, CvSeq *segments, CvPoint vp) {
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq *retLines = cvCreateSeq(0, sizeof(CvSeq), sizeof(float) * 3, storage);
    
    for (int i = 0; i < lines->total; i++) {
        float *line = (float*)cvGetSeqElem(lines, i);
        
        for (int j = 0; j < segments->total; j++) {
            CvPoint* segment = (CvPoint*)cvGetSeqElem(segments, j);
            
            if (segment[0].y > vp.y && lineContainsPoint(line, segment[0], 5)) {
                cvSeqPush(retLines, line);
            }
            else if (segment[1].y > vp.y && lineContainsPoint(line, segment[1], 5)) {
                cvSeqPush(retLines, line);
            }
        }
    }
    
    return retLines;
    
}

#pragma mark lines in direction

CvSeq *verticalLineSegments(IplImage *src, IplImage *dst) {
    CvMemStorage* storageret = cvCreateMemStorage(0);
    CvSeq *retLines = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint) * 2, storageret);
    
    IplImage *cannyImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
    cvCanny(src, cannyImg, 5, 10); 
    
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq *lines = cvHoughLines2(cannyImg,
                                 storage,
                                 CV_HOUGH_PROBABILISTIC,
                                 1,
                                 CV_PI/180,
                                 25,
                                 25,
                                 10);
    
    for (int i = 0; i < lines->total; i++ ) {
        CvPoint* line = (CvPoint*)cvGetSeqElem(lines, i);
        if (abs(line[0].x - line[1].x) < 5) {
            cvSeqPush(retLines, line);
        }
    }
    
    return retLines;
}

void verticalHorizontalLines(IplImage *src, IplImage *dst, CvSeq *vert, CvSeq *hori) {
    IplImage *cannyImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
    cvCanny(src, cannyImg, 5, 10); 
    
    CvMemStorage* storage = cvCreateMemStorage(0);
    
    CvSeq *linesStand = cvHoughLines2(cannyImg,
                                     storage,
                                     CV_HOUGH_STANDARD,
                                     1,
                                     CV_PI/180,
                                     100,
                                     0,
                                     0);
    
    for (int i = 0; i < linesStand->total; i++) {
        float *line = (float*)cvGetSeqElem(linesStand,i);
        float rho = line[0];
        float theta = line[1];
        
        CvPoint pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        
        if (abs(pt1.x - pt2.x) < 10) {
            cvSeqPush(vert, line);
        }
        else if (abs(pt1.y - pt2.y) < 10) {
            cvSeqPush(hori, line);
        }
    }
}

CvSeq *getFloorEdges(CvSeq *originalLines, CvPoint vp) {
    int i, j;
    //Get all lines that pass nearby the vanishing point.
    CvSeq *vpLines = linesContainingPoint(originalLines, &vp);
    
    CvMemStorage* storageFloorEdges = cvCreateMemStorage(0);
    CvSeq *floorEdges = cvCreateSeq(0, sizeof(CvSeq), sizeof(float) * 3, storageFloorEdges);
    
    double rho, theta, current_theta = CV_PI / 2;
    int current_line = -1;
    for(i = 0; i < originalLines->total; i++){
        float *line = (float*)cvGetSeqElem(originalLines,i);
        float rho = line[0];
        float theta = line[1];
        if(theta > current_theta && theta < CV_PI){
            current_line = i;
            current_theta = theta;
        }
    }
    if(current_line != -1){
        float *line = (float*)cvGetSeqElem(originalLines,current_line);
        cvSeqPush(floorEdges, line);
    }
    
    current_theta = CV_PI / 2;
    current_line = -1;
    for(i = 0; i < originalLines->total; i++){
        float *line = (float*)cvGetSeqElem(originalLines,i);
        float rho = line[0];
        float theta = line[1];
        if(theta < current_theta && theta > 0){
            current_line = i;
            current_theta = theta;
        }
    }
    if(current_line != -1){
        float *line = (float*)cvGetSeqElem(originalLines,current_line);
        cvSeqPush(floorEdges, line);
    }
    
    return floorEdges;
}

CvSeq *getFloorEdgesWithVertPass(CvSeq *originalLines, CvSeq *vertSegments, CvPoint vp) {
    int i, j;
    //Get all lines that pass nearby the vanishing point.
    CvSeq *vpLines = linesContainingPoint(originalLines, &vp);
    
    CvMemStorage* storageFloorEdges = cvCreateMemStorage(0);
    CvSeq *floorEdges = cvCreateSeq(0, sizeof(CvSeq), sizeof(float) * 3, storageFloorEdges);
    
    double rho, theta, current_theta = CV_PI / 2;
    int current_line = -1;
    for(i = 0; i < originalLines->total; i++){
        float *line = (float*)cvGetSeqElem(originalLines,i);
        float rho = line[0];
        float theta = line[1];
        if(theta > current_theta && theta < CV_PI){
            current_line = i;
            current_theta = theta;
        }
    }
    if(current_line != -1){
        float *line = (float*)cvGetSeqElem(originalLines,current_line);
        cvSeqPush(floorEdges, line);
    }
    
    current_theta = CV_PI / 2;
    current_line = -1;
    for(i = 0; i < originalLines->total; i++){
        float *line = (float*)cvGetSeqElem(originalLines,i);
        float rho = line[0];
        float theta = line[1];
        if(theta < current_theta && theta > 0){
            current_line = i;
            current_theta = theta;
        }
    }
    if(current_line != -1){
        float *line = (float*)cvGetSeqElem(originalLines,current_line);
        cvSeqPush(floorEdges, line);
    }
    
    return floorEdges;
}
