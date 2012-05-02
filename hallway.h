//
//  hallway.h
//  vision
//
//  Created by Suri on 2012-04-29.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef vision_hallway_h
#define vision_hallway_h

#include <iostream>
using namespace std;

void hallway();
CvSeq *hough(IplImage *src, IplImage *dst);

CvPoint vanishing(CvSeq *lines, IplImage *dst);
CvSeq *verticalLineSegments(IplImage *src, IplImage *dst);
CvSeq *removeNonInterVertLines(CvSeq *lines, IplImage *img);
CvSeq *removeNonVPLines(CvSeq *lines, IplImage *img, CvPoint *point);
void verticallines(IplImage *src, IplImage *dst, CvSeq *vert, CvSeq *hori);
CvSeq *interCornerVP(IplImage *src, IplImage *dst, CvSeq *vert, CvSeq *hori, CvSeq *vp);
CvPoint lineThroughPoint(CvPoint *pt1, CvPoint *pt2);

CvSeq *linesContainingPoint(CvSeq *lines, CvPoint *point);
CvSeq *linesContainingPoints(CvSeq *lines, CvSeq *points);

void drawLinesPoints(CvSeq *lines, IplImage *img, CvScalar color);
void drawLinesLines(CvSeq *lines, IplImage *img, CvScalar color);

#endif
