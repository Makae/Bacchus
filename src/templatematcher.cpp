#include <iostream>
#include <array>

#ifdef __arm__
#include <unistd.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/optflow.hpp>

#include "../include/project/templatematcher.h"
#include "../include/project/utilities.h"

using namespace cv;
using namespace std;

void template_matcher_callback_function(int event, int x, int y, int flags, void* userdata)
{
	Templatematcher * tm = (Templatematcher*)userdata;
	if (event == EVENT_LBUTTONDOWN)
	{
		tm->setSeed(x, y);
	}
}

Templatematcher::Templatematcher() {
	seed = {-1, -1};
	focus = {-1, -1};
	roi_width = 50;
}

Templatematcher::~Templatematcher() {}

void Templatematcher::run(Mat * ptr_img) {
	this->askSeedPoint();
	this->focus = this->seed;
	Mat img;
	while (true) {
		if (waitKey(30) >= 0) break;
		img = Utilities::getInstance()->getImage();
		this->roi = this->getRoi(&img, this->focus, this->roi_width);

	}
}

array<int, 2> Templatematcher::askSeedPoint() {

	namedWindow("Template Seed", 1);
	setMouseCallback("Template Seed", template_matcher_callback_function, this);
	Mat img;
	// WAIT FOR THE SEED CLICK
	while (this->seed[0] == -1 || this->seed[1] == -1) {
		if (waitKey(30) >= 0) break;
		img = Utilities::getInstance()->getImage();
		imshow("Template Seed", img);
	}

	return this->seed;
}

Mat Templatematcher::getRoi(Mat * ptr_img, array<int, 2> center, int width) {
	int x1 = max<int>(0, center[0] - width);
	int y1 = max<int>(0, center[1] - width);
	int x2 = min<int>(ptr_img->cols, center[0] - width);
	int y2 = min<int>(ptr_img->rows, center[1] + width);
	Rect roi(x1, y1, x2, y2);

	return (*ptr_img)(roi);
}

void Templatematcher::setSeed(int x, int y) {
	this->seed[0] = x;
	this->seed[1] = y;
}

void Templatematcher::setFocus(int x, int y) {
	this->focus[0] = x;
	this->focus[1] = y;
}
