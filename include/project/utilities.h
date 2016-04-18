#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include "flandmark_detector.h"

using namespace cv;

class Utilities
{
	public:
		static Utilities* instance;
		static Utilities* getInstance() {
			if (!instance)
				instance = new Utilities();
			return instance;
		}
		void test();
		void detectFaceInImage(IplImage *orig, IplImage* input, CvHaarClassifierCascade* cascade, FLANDMARK_Model *model, int *bbox, double *landmarks);
		void initFlandmarkModel(char * data_file, FLANDMARK_Model*& model);
		void initFaceCascade(char * cascade_file, CvHaarClassifierCascade*& cascade);
		Mat getImage();

	private:
		Utilities();
};