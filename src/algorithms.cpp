#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/optflow.hpp>

#include "flandmark_detector.h"

#include "../include/project/algorithms.h"
#include "../include/project/utilities.h"
#include "../include/project/templatematcher.h"
#include "../include/project/featuretracker.h"

using namespace cv;
using namespace std;
using namespace xfeatures2d;

FLANDMARK_Model* ptr_flm_model;
CvHaarClassifierCascade* ptr_face_cascade;
int* ptr_bbox = (int*)malloc(4 * sizeof(int));
double* ptr_landmarks;
bool flandmark_initialized = false;
Mat* ptr_img_prev = new Mat;
std::vector<cv::Point2f> features_prev;



void Algorithms::showCanny(Mat& img, int hist_thresh_low, int hist_thresh_high) {
	Mat img_gray;
	Mat img_edges;
	cvtColor(img, img_gray, CV_BGR2GRAY);
	Canny(img_gray, img_edges, hist_thresh_low, hist_thresh_high);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img_edges.at<char>(i, j) != 0) {
				Vec3b & color = img.at<Vec3b>(i, j);
				color[0] = 255;
				color[1] = 255;
				color[2] = 255;
			}
		}
	}

	//Show the results
	imshow("Original", img);
	imshow("Canny", img_edges);
}

void Algorithms::showSIFT(Mat& img) {
	Mat img_gray;
	cvtColor(img, img_gray, CV_BGR2GRAY);
	cv::Ptr<Feature2D> sift = SIFT::create(10);
	std::vector<KeyPoint> keypoints_1, keypoints_2;

	sift->detect(img_gray, keypoints_1);

	Mat descriptors_1, descriptors_2;
	sift->compute(img_gray, keypoints_1, descriptors_1);

	drawKeypoints(img_gray, keypoints_1, img_gray);
	imshow("SIFT", img_gray);
}

void Algorithms::showSURF(Mat& img) {
	Mat img_gray;
	cvtColor(img, img_gray, CV_BGR2GRAY);
	cv::Ptr<Feature2D> sift = SURF::create(200);
	std::vector<KeyPoint> keypoints_1, keypoints_2;

	sift->detect(img_gray, keypoints_1);

	Mat descriptors_1, descriptors_2;
	sift->compute(img_gray, keypoints_1, descriptors_1);

	drawKeypoints(img_gray, keypoints_1, img_gray);
	imshow("SURF", img_gray);
}

void Algorithms::showTemplateMatching(Mat& img) {
	Templatematcher *tm = new Templatematcher();
	tm->run(img);
}

void Algorithms::doFlandmark(Mat& img, int& num_faces, int bbox[4], double*& ptr_flandmarks ) {
	IplImage* ptr_img_ipl = cvCloneImage(&(IplImage(img)));
	IplImage* ptr_img_bw = cvCreateImage(cvSize(img.cols, img.rows), IPL_DEPTH_8U, 1);
	Utilities utils = (*Utilities::getInstance());
	int* ptr_bbox = &bbox[0];
	
	if (!flandmark_initialized) {
		utils.initFlandmarkModel("C:\\Libraries\\flandmark\\data\\flandmark_model.dat", ptr_flm_model);
		ptr_landmarks = (double*)malloc(2 * (*ptr_flm_model).data.options.M * sizeof(double));
		utils.initFaceCascade("C:\\Libraries\\flandmark\\data\\haarcascade_frontalface_alt.xml", ptr_face_cascade);
		flandmark_initialized = true;
	}

	cvConvertImage(ptr_img_ipl, ptr_img_bw);

	utils.detectFaceInImage(
		ptr_img_bw,
		num_faces,
		ptr_face_cascade,
		ptr_flm_model,
		ptr_bbox,
		ptr_landmarks);
}

void Algorithms::showFlandmark(Mat& img) {
	char fps[50];
	int bbox[4];
	int num_faces = 0;
	double* landmarks = 0;
	IplImage* ptr_img_ipl = &(IplImage)img;

	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);

	Algorithms::doFlandmark(img, num_faces, bbox, landmarks);

	// display landmarks
	cvRectangle(ptr_img_ipl, cvPoint(bbox[0], bbox[1]), cvPoint(bbox[2], bbox[3]), CV_RGB(255, 0, 0));
	cvRectangle(ptr_img_ipl, cvPoint(ptr_flm_model->bb[0], ptr_flm_model->bb[1]), cvPoint(ptr_flm_model->bb[2], ptr_flm_model->bb[3]), CV_RGB(0, 0, 255));
	cvCircle(ptr_img_ipl, cvPoint((int)ptr_landmarks[0], (int)ptr_landmarks[1]), 3, CV_RGB(0, 0, 255), CV_FILLED);
	for (int i = 2; i < 2 * ptr_flm_model->data.options.M; i += 2)
	{
		cvCircle(ptr_img_ipl, cvPoint(int(ptr_landmarks[i]), int(ptr_landmarks[i + 1])), 3, CV_RGB(255, 0, 0), CV_FILLED);
	}

	if (num_faces > 0) {
		printf("Faces detected: %d", num_faces);
	} else {
		printf("NO Face\n");
	}



	imshow("Flandmark Feature Points", cvarrToMat(ptr_img_ipl));
}

void Algorithms::showFeatureTracker(Mat& img)  {
	Featuretracker *ft = new Featuretracker();
	ft->run(img);
}

void Algorithms::showLucasKanade(Mat& img) {
	Mat* ptr_img_next = new Mat;
	Mat* ptr_img_copy = new Mat;
	std::vector<cv::Point2f>     features_next;
	std::vector<unsigned char> status;
	std::vector<float>         error;

	int bbox[4];
	int num_faces = 0;
	double* landmarks = 0;

	cvtColor(img, (*ptr_img_next), CV_BGR2GRAY);
	*ptr_img_copy = *ptr_img_next;
	bool no_prev_img = ptr_img_prev->cols == 0;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	if (no_prev_img) {
		cout << "Searching Face and points";
		Algorithms::doFlandmark(img, num_faces, bbox, landmarks);
		cout << "Found " << num_faces << " Face with " << ptr_flm_model->data.options.M << " Featurepoints";
		if (num_faces == 0)
			return;
		for (int i = 2; i < 2 * ptr_flm_model->data.options.M; i += 2) {
			features_prev.push_back(cvPoint(float(ptr_landmarks[i]), float(ptr_landmarks[i + 1])));
		}
		// cv::Mat mask;
		// goodFeaturesToTrack(*ptr_img_next, features_prev, 100, 0.3, 7, mask, 3, true, 0.04);
		//features_next = features_prev;
		*ptr_img_prev = *ptr_img_next;
		return;
	}

	// calculate optical flow
	try {
		calcOpticalFlowPyrLK(*ptr_img_prev, *ptr_img_next, features_prev, features_next, status, error, cvSize(31, 31), 3);
	} catch (cv::Exception & e) {
		cerr << e.msg << endl;
		throw e;
	}

	std::vector<cv::Point2f> trackedPts;
	Scalar mask = Scalar(255);
	for (size_t i = 0; i<status.size(); i++)
	{
		if (status[i])
		{
			trackedPts.push_back(features_next[i]);

			cv::circle(mask, features_prev[i], 15, cv::Scalar(0), -1);
			cv::line(*ptr_img_copy, features_prev[i], features_next[i], cv::Scalar(0, 250, 0));
			cv::circle(*ptr_img_copy, features_next[i], 3, cv::Scalar(0, 250, 0), -1);
		}
	}
	*ptr_img_next = *ptr_img_prev;
	//features_prev = features_next;
	imshow("Optical Flow", *ptr_img_copy);
}