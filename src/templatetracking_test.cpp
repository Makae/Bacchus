#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "../include/project/utilities.h"
#include <sstream>


using namespace cv;
using namespace std;

Point point1, point2, prev_match_point; /* vertical points of the bounding box */
int drag = 0;
Rect rect; /* bounding box */
Mat img, roiImg; /* roiImg - the part of the image in the bounding box */
int select_flag = 0;
bool go_fast = false;
Mat mytemplate;


///------- template matching -----------------------------------------------------------------------------------------------

Mat TplMatch(Mat &img, Mat &mytemplate)
{
	Mat result;

	matchTemplate(img, mytemplate, result, CV_TM_SQDIFF_NORMED);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	return result;
}


///------- Localizing the best match with minMaxLoc ------------------------------------------------------------------------

Point minmax(Mat &result)
{
	double minVal, maxVal;
	Point  minLoc, maxLoc, matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	matchLoc = minLoc;

	return matchLoc;
}


///------- tracking --------------------------------------------------------------------------------------------------------

void track()
{
	if (select_flag)
	{
		//roiImg.copyTo(mytemplate);
		//         select_flag = false;
		go_fast = true;
	}

	//     imshow( "mytemplate", mytemplate ); waitKey(0);
	int template_width = mytemplate.cols;
	int template_height = mytemplate.rows;
	Rect AOI = cv::Rect(
		max<int>(0, prev_match_point.x - template_width), 
		max<int>(0, prev_match_point.y - template_height), 
		template_width * 3,
		template_height * 3);
	AOI.width  = min<int>(AOI.width,  img.cols - AOI.x);
	AOI.height = min<int>(AOI.height, img.rows - AOI.y);
	
	Mat img_aoi = img(AOI);

	Mat result = TplMatch(img_aoi, mytemplate);
	Point aoi_match = minmax(result);
	Point match;
	match.x = aoi_match.x + AOI.x;
	match.y = aoi_match.y + AOI.y;

	std::cout << "match: " << match << endl;

	/// latest match is the new template
	Rect ROI = cv::Rect(match.x, match.y, template_width, template_height);
	roiImg = img(ROI);
	roiImg.copyTo(mytemplate);

	rectangle(img, Point(AOI.x, AOI.y), Point(AOI.x + AOI.width, AOI.y + AOI.height), CV_RGB(0, 255, 0), 0.5);
	rectangle(img, match, Point(match.x + ROI.width, match.y + ROI.height), CV_RGB(255, 0, 0), 0.5);

	imshow("roiImg", roiImg); //waitKey(0);
	prev_match_point = match;
}


///------- MouseCallback function ------------------------------------------------------------------------------------------

void mouseHandler(int event, int x, int y, int flags, void *param)
{
	if (event == CV_EVENT_LBUTTONDOWN && !drag)
	{
		/// left button clicked. ROI selection begins
		point1 = Point(x, y);
		drag = 1;
	}

	if (event == CV_EVENT_MOUSEMOVE && drag)
	{
		/// mouse dragged. ROI being selected
		Mat img1 = img.clone();
		point2 = Point(x, y);
		rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
		imshow("image", img1);
	}

	if (event == CV_EVENT_LBUTTONUP && drag)
	{
		point2 = Point(x, y);
		rect = Rect(point1.x, point1.y, x - point1.x, y - point1.y);
		drag = 0;
		prev_match_point = point1;
		roiImg = img(rect);
		roiImg.copyTo(mytemplate);
		//  imshow("MOUSE roiImg", roiImg); waitKey(0);
	}

	if (event == CV_EVENT_LBUTTONUP)
	{
		/// ROI selected
		select_flag = 1;
		drag = 0;
	}

}



///------- Main() ----------------------------------------------------------------------------------------------------------

int call_template_tracking()
{
	int k;
	/*
	///open webcam
	VideoCapture cap(0);
	if (!cap.isOpened())
	return 1;*/

	img = Utilities::getInstance()->getImage();
	GaussianBlur(img, img, Size(7, 7), 3.0);
	imshow("image", img);

	while (1)
	{
		img = Utilities::getInstance()->getImage();
		if (img.empty())
			break;

		// Flip the frame horizontally and add blur
		cv::flip(img, img, 1);
		GaussianBlur(img, img, Size(7, 7), 3.0);

		if (rect.width == 0 && rect.height == 0)
			cvSetMouseCallback("image", mouseHandler, NULL);
		else
			track();

		imshow("image", img);
		//  waitKey(100);   k = waitKey(75);
		k = waitKey(go_fast ? 30 : 10000);
		if (k == 27)
			break;
	}

	return 0;
}