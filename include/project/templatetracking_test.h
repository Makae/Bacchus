#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "utilities.h"
#include <sstream>


using namespace cv;
using namespace std;


///------- template matching -----------------------------------------------------------------------------------------------

Mat TplMatch(Mat &img, Mat &mytemplate);


///------- Localizing the best match with minMaxLoc ------------------------------------------------------------------------

Point minmax(Mat &result);


///------- tracking --------------------------------------------------------------------------------------------------------

void track();


///------- MouseCallback function ------------------------------------------------------------------------------------------

void mouseHandler(int event, int x, int y, int flags, void *param);



///------- Main() ----------------------------------------------------------------------------------------------------------

int call_template_tracking();
