#pragma once
#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include <array>

#include <opencv2/core.hpp>

using namespace cv;
class Featuretracker
{
public:
	Featuretracker();
	~Featuretracker();
	
	void run(Mat & ptr_img);
	void setSeed(int x, int y);
	void setFocus(int x, int y);


private:
	Point2f* seed;
	Point2f* focus;
	Mat img_current;
	Mat img_previous;
	Point2f* askSeedPoint();
};

#endif // FEATURE_TRACKER
