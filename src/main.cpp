#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

#include "../include/project/mouthtracker.h"
#include "../include/project/utilities.h"
#include "../include/project/algorithms.h"

using namespace std;
using namespace cv;

const int ALGO_CANNY     = 0;
const int ALGO_FLANDMARK = 1;
const int ALGO_SIFT = 2;
const int ALGO_SURF = 3;


int active_algo = ALGO_SURF;

int hist_thresh_low = 30;
int hist_thresh_high = 90;

int* ptr_bound_h_value = 0;
int* ptr_bound_v_value = 0;

int bound_min = 0;
int bound_max = 255;

void showAlgo(int active_algo, Mat* ptr_img) {
	switch (active_algo) {
	case ALGO_SURF:
		Algorithms::showSURF(ptr_img);
	break;

	case ALGO_SIFT:
		Algorithms::showSIFT(ptr_img);
	break;

	case ALGO_FLANDMARK:
		Algorithms::showFlandmark(ptr_img);
	break;

	case ALGO_CANNY:
	default:
		ptr_bound_h_value = &hist_thresh_low;
		ptr_bound_v_value = &hist_thresh_high;
	
		Algorithms::showCanny(ptr_img, hist_thresh_low, hist_thresh_high);
	break;
	}
}
int handleInput() {
	int key_code = waitKey(20);
	cout << "key:" << key_code << "\n";
	if (key_code == 49) { // 1
		active_algo = ALGO_CANNY;
	} else if (key_code == 50) { // 2
		active_algo = ALGO_FLANDMARK;
	} else if (key_code == 51) { // 3

	}

	if (key_code == 2490368) { // UP
		(*ptr_bound_v_value) = (*ptr_bound_v_value) == bound_max ? bound_max : (*ptr_bound_v_value) + 1;
	
	} else if (key_code == 2621440) { // DOWN
		(*ptr_bound_v_value) = (*ptr_bound_v_value) == bound_min ? bound_min : (*ptr_bound_v_value) - 1;
		
	} else if (key_code == 2424832) { // LEFT
		(*ptr_bound_h_value) = (*ptr_bound_h_value) == bound_min ? bound_min : (*ptr_bound_h_value) - 1;
		
	} else if (key_code == 2555904) { // RIGHT
		(*ptr_bound_h_value) = (*ptr_bound_h_value) == bound_max ? bound_max : (*ptr_bound_h_value) + 1;
	}
	cout << key_code;
	return active_algo;
}
int main(int argc, char** argv)
{
	Utilities* utilities = Utilities::getInstance();
	Mat img;
	Mat img_gray;
	while (true) {
		img = utilities->getImage();
		active_algo = handleInput();
		showAlgo(active_algo, &img);
	}
}