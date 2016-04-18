#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
class Algorithms
{
	public:
		static void showCanny(Mat* ptr_img, int hist_thresh_low, int hist_thresh_high);
		static void showSIFT(Mat * ptr_img);
		static void showSURF(Mat * ptr_img);
		static void showFlandmark(Mat * ptr_img);

	private:
		Algorithms();
};