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

CvSeq *linesContainingPoint(CvSeq *lines, CvPoint *point);
CvSeq *linesContainingPoints(CvSeq *lines, CvSeq *points);
CvPoint lineThroughPoint(CvPoint *pt1, CvPoint *pt2);

void drawLinesPoints(CvSeq *lines, IplImage *img, CvScalar color);
void drawLinesLines(CvSeq *lines, IplImage *img, CvScalar color, int lineWidth = 1);

CvPoint vanishing(CvSeq *lines, IplImage *dst);

CvSeq *linesIntersectingCorners(CvSeq *lines, IplImage *src);
CvSeq *linesThroughVp(CvSeq *lines, IplImage *img, CvPoint *vp);

CvSeq *verticalLineSegments(IplImage *src, IplImage *dst);
void verticalHorizontalLines(IplImage *src, IplImage *dst, CvSeq *vert, CvSeq *hori);

CvSeq *getFloorEdges(CvSeq *originalLines, CvPoint vp); 

#endif
