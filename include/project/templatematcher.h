#pragma once
#ifndef TEMPLATEMATCHER_H
#define TEMPLATEMATCHER_H

#include <array>

#include <opencv2/core.hpp>

using namespace cv;
class Templatematcher
{
public:
	Templatematcher();
	~Templatematcher();
	
	void run(Mat * ptr_img);
	void setSeed(int x, int y);
	void setFocus(int x, int y);


private:
	Mat roi;
	int roi_width;

	std::array<int, 2> seed;
	std::array<int, 2> focus;

	std::array<int, 2> askSeedPoint();
	Mat getRoi(Mat * ptr_img, std::array<int, 2> focus, int width);
};

#endif // TEMPLATEMATCHER_H
