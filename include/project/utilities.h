#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#ifndef __arm__
extern "C" {
#include "flandmark_detector.h"
}
#endif

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
		#ifndef __arm__
		void detectFaceInImage(IplImage * input, int & num_faces, CvHaarClassifierCascade * cascade, FLANDMARK_Model * model, int *& bbox, double *& landmarks);
		void initFlandmarkModel(char * data_file, FLANDMARK_Model*& model);
		#endif		
		void initFaceCascade(char * cascade_file, CvHaarClassifierCascade*& cascade);
		Mat getImage();

	private:
		Utilities();
};
