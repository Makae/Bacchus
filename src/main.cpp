#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

#include "../include/project/mouthtracker.h"
#include "../include/project/utilities.h"
#include "../include/project/algorithms.h"
#include "../include/project/templatetracking_test.h"

#include "main.h"

#include "../include/project/meanshift.h"


using namespace std;
using namespace cv;

const int ALGO_CANNY     = 0;
const int ALGO_FLANDMARK = 1;
const int ALGO_SIFT = 2;
const int ALGO_SURF = 4;
const int ALGO_TEMPLATE_MATCHING = 8;
const int ALGO_LUCAS_KANADE = 9;

int active_algo = ALGO_TEMPLATE_MATCHING;

int hist_thresh_low = 30;
int hist_thresh_high = 90;

int* ptr_bound_h_value = 0;
int* ptr_bound_v_value = 0;

int bound_min = 0;
int bound_max = 255;

int current_x = -1;
int current_y = -1;

bool algo_handles_img = false;

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

	case ALGO_LUCAS_KANADE:
		Algorithms::showLucasKanade(ptr_img);
		break;

	case ALGO_TEMPLATE_MATCHING:
		Algorithms::showTemplateMatching(ptr_img);
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
	// cout << "key:" << key_code << "\n";
	if (key_code == 49) { // 1
		active_algo = ALGO_CANNY;
	} else if (key_code == 50) { // 2
		active_algo = ALGO_FLANDMARK;
	} else if (key_code == 51) { // 3
		active_algo = ALGO_SIFT;
	} else if (key_code == 52) { // 4
		active_algo = ALGO_SURF;
	} else if (key_code == 53) { // 5
		active_algo = ALGO_LUCAS_KANADE;
	} else if (key_code == 54) { // 6
		active_algo = ALGO_TEMPLATE_MATCHING;
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
	//cout << key_code;
	return active_algo;
}


int main(int argc, char** argv)
{
	
	call_template_tracking();
	return -1;
	/*Utilities* utilities = Utilities::getInstance();
	Mat img;
	Mat img_gray;
	while(true) {*/
		//img = utilities->getImage();
		active_algo = handleInput();
		showAlgo(active_algo, 0);
	//}
}