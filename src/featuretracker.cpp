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

#include "../include/project/featuretracker.h"
#include "../include/project/utilities.h"

using namespace cv;
using namespace std;

void feature_tracker_callback_function(int event, int x, int y, int flags, void* userdata)
{
	Featuretracker * tm = (Featuretracker*)userdata;
	if (event == EVENT_LBUTTONDOWN)
	{
		tm->setSeed(x, y);
	}
}

Featuretracker::Featuretracker() {
	this->seed = NULL;
	this->focus = NULL;
}

Featuretracker::~Featuretracker() {}

void Featuretracker::run(Mat & img) {
	std::vector<cv::Point2f> features_prev;
	std::vector<cv::Point2f> features_next;
	std::vector<unsigned char> status;
	std::vector<float>         error;

	this->askSeedPoint();
	features_prev.push_back(*this->seed);
	this->setFocus(this->seed->x, this->seed->y);

	Size window_size(32, 32);
	int  max_pyr_level = 3;
	TermCriteria termination_criterias(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3);
	//Mat pyrA = cvCreateMat(pyr_size.width / 2, pyr_size.height / 2, CV_8);
	Mat img_current_rgb;

	
	bool needInit = true;
	this->img_current.copyTo(this->img_previous);
	while (true) {

		Utilities::getInstance()->getImage().copyTo(img_current_rgb);
		cvtColor(img_current_rgb, this->img_current, CV_BGR2GRAY);
	
		try {
			Mat tmp;
			cv::bitwise_and(this->img_previous, this->img_current, tmp);
			imshow("difference", tmp);
			//cout << "#prev: " << &(*this->img_previous) << "\n";
			//cout << "#current: " << &(*this->img_current) << "\n";
			calcOpticalFlowPyrLK(
				this->img_previous, 
				this->img_current, 
				features_prev, 
				features_next, 
				status, 
				error, 
				window_size, 
				max_pyr_level,
				termination_criterias, 0, 0.001);
		} catch (cv::Exception & e) {
			cerr << e.msg << endl;
			throw e;
		}
		bool noswap = false;
		if (status.size() == 0) {
			for (int i = 0; i < features_prev.size(); i++) {
				cv::line(img_current_rgb, features_prev[i], features_prev[i], cv::Scalar(0, 250, 0));
				cv::circle(img_current_rgb, features_prev[i], 3, cv::Scalar(0, 250, 0), -1);
			}
			noswap = true;
		}
		for (size_t i = 0; i < status.size(); i++)
		{
			if (status[i]) {
				cout << "x:" << features_next[i].x << " y:" << features_next[i].y << "\n";
				cv::line(img_current_rgb, features_prev[i], features_next[i], cv::Scalar(0, 250, 0));
				cv::circle(img_current_rgb, features_next[i], 3, cv::Scalar(0, 250, 0), -1);
			}
		}

		//features_prev = features_next;
		imshow("Tracking in Progress...", img_current_rgb);

		// Switching the pointers for next iteration
		if (!noswap) {
			std::swap(features_next, features_prev);
		}
		cv::swap(this->img_current, this->img_previous);


		setMouseCallback("Tracking in Progress...", feature_tracker_callback_function, this);
		if (waitKey(30) >= 0) 
			break;
		if (this->focus->x != this->seed->x || this->focus->y != this->seed->y) {
			this->setFocus((*this->seed).x, (*this->seed).y);
			features_prev.push_back(*this->focus);
		}
	}
}

Point2f* Featuretracker::askSeedPoint() {

	namedWindow("Template Seed", 1);
	setMouseCallback("Template Seed", feature_tracker_callback_function, this);
	// WAIT FOR THE SEED CLICK
	while (this->seed == 0) {
		if (waitKey(30) >= 0) break;
		this->img_current = Utilities::getInstance()->getImage();
		imshow("Template Seed", this->img_current);
	}
	cvtColor(this->img_current, this->img_current, CV_BGR2GRAY);
	cout << "GOT SEEED";
	destroyWindow("Template Seed");
	return this->seed;
}

void Featuretracker::setSeed(int x, int y) {
	this->seed = new Point2f(x, y);
}

void Featuretracker::setFocus(int x, int y) {
	this->focus = new Point2f(x, y);
}
