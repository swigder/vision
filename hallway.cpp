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
    cvCanny(src8bitgray, cannyImg, 50, 255); 

    // hough to get lines
    CvSeq *lines = hough(cannyImg, houghColorImg);
    
    // get vanishing point
    CvPoint point = vanishing(lines, houghColorImg);
    
    // get lines that go through vp
    CvSeq *vpLines = removeNonVPLines(lines, houghColorImg, &point);
    
    // get the vertical lines
    CvMemStorage* storageVert = cvCreateMemStorage(0);
    CvMemStorage* storageHori = cvCreateMemStorage(0);
    
    CvSeq *vert = cvCreateSeq(0, sizeof(CvSeq), sizeof(float) * 3, storageVert);
    CvSeq *hori = cvCreateSeq(0, sizeof(CvSeq), sizeof(float) * 3, storageHori);

    verticallines(src8bitgray, houghColorImg, vert, hori);
    
    // get the intersections of vert, hori, vp
    CvSeq *intPoints = interVertHoriVP(src8bitgray, houghColorImg, vert, hori, vpLines);
    
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

CvSeq *removeNonVPLines(CvSeq *lines, IplImage *img, CvPoint *vp) {
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq *retLines = cvCreateSeq(0, sizeof(CvSeq), sizeof(float) * 3, storage);
    
    int total = lines->total;
    for (int i = 0; i < total; i++) {
        float *line = (float*)cvGetSeqElem(lines,i);
        float rho = line[0];
        float theta = line[1];
        
        // check if line goes through vp
        // don't forget to check for nan!
        double exy = -cos(theta) / sin(theta) * vp->x + rho / sin(theta);
        if (exy != exy || vp->y > exy + 5 || vp->y < exy - 5) {
            continue;
        }
        
        cvSeqPush(retLines, line);
                
        CvPoint pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        
        cvLine(img, pt1, pt2, CV_RGB(255,0,0), 2, 8);
    }
    
    cvCircle(img, *vp, 10, CV_RGB(0,255,0));
    
    return retLines;
}

void verticallines(IplImage *src, IplImage *dst, CvSeq *vert, CvSeq *hori) {
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
    
    CvSeq *linesProb = cvHoughLines2(cannyImg,
                                 storage,
                                 CV_HOUGH_PROBABILISTIC,
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
//            cvLine(dst, pt1, pt2, CV_RGB(0,0,255), 2, 8);
            cvSeqPush(vert, line);
        }
        else if (abs(pt1.y - pt2.y) < 10) {
//            cvLine(dst, pt1, pt2, CV_RGB(0,0,255), 2, 8);
            cvSeqPush(hori, line);
        }
    }
    
//    for(int i = 0; i < linesProb->total; i++) {
//        CvPoint* line = (CvPoint*)cvGetSeqElem(linesProb,i);
//        if (abs(line[0].x - line[1].x) < 10) {
//            cvSeqPush(vert, line);
////            cvLine(dst, line[0], line[1], CV_RGB(0,0,255), 2, 8);
//        }
//        else if (abs(line[0].y - line[1].y) < 10) {
//            cvSeqPush(hori, line);
////            cvLine(dst, line[0], line[1], CV_RGB(0,0,255), 2, 8);
//        }
//    }
}

CvSeq *interVertHoriVP(IplImage *src, IplImage *dst, CvSeq *vert, CvSeq *hori, CvSeq *vp) {
    CvMat *A = cvCreateMat(2, 2, CV_64FC1);
    CvMat *A1 = cvCreateMat(2, 2, CV_64FC1);
    CvMat *b = cvCreateMat(2, 1, CV_64FC1);
    CvMat *x = cvCreateMat(2, 1, CV_64FC1);
    
    for (int i = 0; i < vert->total; i++) {
        for (int j = 0; j < hori->total; j++) {
            for (int k = 0; k < vp->total; k++) {
                float *line1 = (float*)cvGetSeqElem(vert, i);
                float *line2 = (float*)cvGetSeqElem(hori, j);
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
                
//                CvPoint* line1 = (CvPoint*)cvGetSeqElem(vert,i);
//                CvPoint* line2 = (CvPoint*)cvGetSeqElem(hori,j);
//
//                CvPoint linevert = lineThroughPoint(&line1[0], &line1[1]);
//                CvPoint linehori = lineThroughPoint(&line2[0], &line2[1]);
//                
//                float rho1 = linevert.x;
//                float theta1 = linevert.y;
//                float rho2 = linehori.x;
//                float theta2 = linehori.y;
//                
//                cvmSet(A, 0, 0, cos(theta1));
//                cvmSet(A, 0, 1, sin(theta1));
//                cvmSet(A, 1, 0, cos(theta2));
//                cvmSet(A, 1, 1, sin(theta2));
//                cvmSet(b, 0, 0, rho1);
//                cvmSet(b, 1, 0, rho2);
//                
//                cvInvert(A, A1);
//                cvMatMul(A1, b, x);
                
                double x0 = cvmGet(x, 0, 0);
                double y0 = cvmGet(x, 1, 0);                
                float *line3 = (float*)cvGetSeqElem(vp,k);
                float rho = line3[0];
                float theta = line3[1];
                
                double exy = -cos(theta) / sin(theta) * x0 + rho / sin(theta);
                if (exy == exy && abs(y0 - exy) < 5) {
                    cvCircle(dst, cvPoint(x0, y0), 10, CV_RGB(0,255,0));
                }
            }
        }
    }
}

CvPoint lineThroughPoint(CvPoint *pt1, CvPoint *pt2) {
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