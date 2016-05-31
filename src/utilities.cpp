#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#ifdef __arm__
#include <raspicam/raspicam_cv.h>
#endif
#include "flandmark_detector.h"

#include "../include/project/utilities.h"

using namespace cv;
using namespace std;

Utilities* Utilities::instance = 0;
VideoCapture* std_cap = 0;
#ifdef __arm__
raspicam::RaspiCam* raspi_cap = 0;
#endif


Utilities::Utilities() {}

void Utilities::test() {
	cout << "TEST";
}

void Utilities::detectFaceInImage(IplImage* input, int& num_faces, CvHaarClassifierCascade* cascade, FLANDMARK_Model * model	, int *& bbox, double *& landmarks)
{
	// Smallest face size.
	CvSize minFeatureSize = cvSize(40, 40);
	int flags = CV_HAAR_DO_CANNY_PRUNING;
	// How detailed should the search be.
	float search_scale_factor = 1.1f;
	CvMemStorage* storage;
	CvSeq* rects;
	int nFaces;

	storage = cvCreateMemStorage(0);
	cvClearMemStorage(storage);

	// Detect all the faces in the greyscale image.
	rects = cvHaarDetectObjects(input, cascade, storage, search_scale_factor, 2, flags, minFeatureSize);
	num_faces = rects->total;

	double t = (double)cvGetTickCount();
	for (int iface = 0; iface < (rects ? num_faces : 0); ++iface) {
		CvRect *r = (CvRect*)cvGetSeqElem(rects, iface);

		bbox[0] = r->x;
		bbox[1] = r->y;
		bbox[2] = r->x + r->width;
		bbox[3] = r->y + r->height;

		flandmark_detect(input, bbox, model, landmarks);
		break;

	}


	cvReleaseMemStorage(&storage);
}

void Utilities::initFlandmarkModel(char* data_file, FLANDMARK_Model*& model) {
	// ------------- begin flandmark load model
	int t = (double)cvGetTickCount();
	model = flandmark_init(data_file);
	if (model == 0)
	{
		printf("Structure model was not created. Corrupted file flandmark_model.dat?\n");
		exit(1);
	}
	t = (double)cvGetTickCount() - t;
	int ms = cvRound(t / ((double)cvGetTickFrequency() * 1000.0));
	printf("Structure model loaded in %d ms.\n", ms);
	// ------------- end flandmark load model
}

void Utilities::initFaceCascade(char* cascade_file, CvHaarClassifierCascade*& cascade) {
	cascade = (CvHaarClassifierCascade*) cvLoad(cascade_file, 0, 0, 0);
}

Mat getImageStd() {
	if (std_cap == 0) {
		// capture from web camera init
		std_cap = new VideoCapture(0);
		std_cap->open(0);
	}
	Mat img;
	(*std_cap) >> img;
	return img;
}

#ifdef __arm__
Mat getImageRaspberry() {
	Mat img;
	if (raspi_cap == 0) {
		raspi_cap = new Camera; //Cmaera object
								   //Open camera 
		cout << "Opening Camera..." << endl;
		if (!Camera.open()) { 
			cerr << "Error opening camera" << endl; return -1; 
		}
		//wait a while until camera stabilizes
		cout << "Sleeping for 3 secs" << endl;
		sleep(3);
		//capture
	}

	Camera.grab();
	//extract the image in rgb format
	Camera.retrieve(img);
	return img;
}
#endif

Mat Utilities::getImage() {
	Mat img;
#ifdef __arm__
	img = getRaspberry();
#else
	img = getImageStd();
#endif
	return img;
}