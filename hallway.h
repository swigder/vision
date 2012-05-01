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
void vanishing(CvSeq *lines);

#endif
