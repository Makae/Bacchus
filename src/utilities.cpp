#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

#include "../include/project/utilities.h"

using namespace cv;
using namespace std;

Utilities* Utilities::instance = 0;
VideoCapture* std_cap = 0;
VideoCapture* raspi_cap = 0;

Utilities::Utilities() {}

void Utilities::test() {
	cout << "TEST";
}

Mat Utilities::getImage() {
	Mat img;
	#ifdef __arm__
	img = getRaspberry();
	#else
	img = getImageStd();
	#endif
	return img;
}

Mat getImageRaspberry() {
	raspicam::RaspiCam Camera; //Cmaera object
							   //Open camera 
	cout << "Opening Camera..." << endl;
	if (!Camera.open()) { cerr << "Error opening camera" << endl; return -1; }
	//wait a while until camera stabilizes
	cout << "Sleeping for 3 secs" << endl;
	sleep(3);
	//capture
	Camera.grab();
	//allocate memory
	unsigned char *data = new unsigned char[Camera.getImageTypeSize(raspicam::RASPICAM_FORMAT_RGB)];
	//extract the image in rgb format
	Camera.retrieve(data, raspicam::RASPICAM_FORMAT_RGB);//get camera image
														 //save
	std::ofstream outFile("raspicam_image.ppm", std::ios::binary);
	outFile << "P6\n" << Camera.getWidth() << " " << Camera.getHeight() << " 255\n";
	outFile.write((char*)data, Camera.getImageTypeSize(raspicam::RASPICAM_FORMAT_RGB));
	cout << "Image saved at raspicam_image.ppm" << endl;
}

Mat getImageStd() {
	if (std_cap != 0) {
		// capture from web camera init
		std_cap = new VideoCapture(0);
		std_cap->open(0);
	}
	Mat img;
	(*std_cap) >> img;
	return img;
}