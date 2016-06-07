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
		static void showCanny(Mat& img, int hist_thresh_low, int hist_thresh_high);
		static void showViolaJones(Mat & img);
		static void showSnake(Mat & img);
		static void showSIFT(Mat& img);
		static void showSURF(Mat& img);
		static void doFlandmark(Mat& img, int & num_faces, int bbox[4], double *& ptr_flandmarks);
		static void showFlandmark(Mat& img);
		static void showFeatureTracker(Mat& img);
		static void showLucasKanade(Mat& img);
		static void showTemplateMatching(Mat& img);
	private:
		Algorithms();
};