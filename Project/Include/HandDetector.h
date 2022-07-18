//HandDetector.hpp

//@author: Riccardo Rampon

#ifndef HandDetector_hpp
#define HandDetector_hpp

//opencv
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

//stl
#include <stdio.h>
#include <iostream>
#include <fstream>

class HandDetector {
private:

	cv::dnn::Net net;
	std::vector<std::string> class_names;
	std::vector<cv::Scalar> colors;

	//Constants 
	const float INPUT_WIDTH = 416.0;
	const float INPUT_HEIGHT = 416.0;
	const float CONFIDENCE_THRESHOLD = 0.75; // Confidence threshold
	const float NMS_THRESHOLD = 0.4; //Non-Maxima-Suppression threshold

	const float FONT_SCALE = 0.5;
	const int  FONT_TYPE = cv::FONT_HERSHEY_SIMPLEX;
	const int FONT_THICKNESS = 2;

	void draw_box_prediction(cv::Mat& image, const cv::Scalar& color, int classId, float confidence, int X_top, int Y_top, int X_bottom, int Y_bottom);

	std::vector<std::string> getOutputLayersNames(const cv::dnn::Net& net);

	void refineBBox(const cv::Mat& img, cv::Rect& bbox);

public:

	HandDetector(const cv::dnn::Net& net, const std::vector<std::string>& class_names);

	std::vector<cv::Mat> forward_process(const cv::Mat& imgInput);

	std::vector<std::pair<cv::Rect, cv::Scalar>> post_process(cv::Mat& image, std::vector<cv::Mat>& outputs);

};

#endif
