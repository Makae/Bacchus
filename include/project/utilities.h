#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

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
		Mat getImage();

	private:
		Utilities();
};