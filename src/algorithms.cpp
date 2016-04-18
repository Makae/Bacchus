#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

#include "flandmark_detector.h"

#include "../include/project/algorithms.h"
#include "../include/project/utilities.h"

using namespace cv;
using namespace std;


FLANDMARK_Model* ptr_flm_model;
CvHaarClassifierCascade* ptr_face_cascade;
int* ptr_bbox = (int*)malloc(4 * sizeof(int));
double* ptr_landmarks;
bool flandmark_initialized = false;

void Algorithms::showCanny(Mat* ptr_img, int hist_thresh_low, int hist_thresh_high) {
	Mat img = (*ptr_img);
	Mat img_gray;
	Mat img_edges;
	cvtColor((*ptr_img), img_gray, CV_BGR2GRAY);
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

void Algorithms::showFlandmark(Mat* ptr_img) {
	int t = 0;
	int ms = 0;
	IplImage* ptr_img_ipl = cvCloneImage(&(IplImage)(*ptr_img));
	IplImage* ptr_img_bw = cvCreateImage(cvSize((*ptr_img).cols, (*ptr_img).rows), IPL_DEPTH_8U, 1);
	IplImage* ptr_img_flandmark = cvCloneImage(&(IplImage)(*ptr_img_ipl));
	Utilities utils = (*Utilities::getInstance());
	char fps[50];
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
	if (!flandmark_initialized) {
		utils.initFlandmarkModel("C:\\Libraries\\flandmark\\data\\flandmark_model.dat", ptr_flm_model);
		ptr_landmarks = (double*)malloc(2 * (*ptr_flm_model).data.options.M * sizeof(double));
		utils.initFaceCascade("C:\\Libraries\\flandmark\\data\\haarcascade_frontalface_alt.xml", ptr_face_cascade);
	}

	cvConvertImage(ptr_img_ipl, ptr_img_bw);
	cout << "T*ST";
	utils.detectFaceInImage(
		ptr_img_flandmark, 
		ptr_img_bw, 
		ptr_face_cascade, 
		ptr_flm_model, 
		ptr_bbox, 
		ptr_landmarks);

	t = (double)cvGetTickCount() - t;
	sprintf(fps, "%.2f fps", 1000.0 / (t / ((double)cvGetTickFrequency() * 1000.0)));
	cvPutText(ptr_img_flandmark, fps, cvPoint(10, 40), &font, cvScalar(255, 0, 0, 0));

	imshow("Flandmark Feature Points", cvarrToMat(ptr_img_flandmark));
}