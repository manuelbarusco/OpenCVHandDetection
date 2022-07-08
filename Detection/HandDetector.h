//HandDetector.h

//Author: Riccardo Rampon

#ifdef handDetector_hpp
#endif handDetector_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

class HandDetector {
	private:

		cv::Mat image;
		std::ofstream outF;

		const float INPUT_WIDTH = 416.0;
		const float INPUT_HEIGHT = 416.0;
		const float CONFIDENCE_THRESHOLD = 0.75; // Confidence threshold
		const float NMS_THRESHOLD = 0.4; //Non-Maxima-Suppression threshold

		const float FONT_SCALE = 0.5;
		const int  FONT_TYPE = cv::FONT_HERSHEY_SIMPLEX;
		const int FONT_THICKNESS = 2;

		cv::Scalar COLOR = cv::Scalar(0,0,255);

	public:
		HandDetector(const cv::Mat& img, const std::string path_file);
		void show_images(const std::vector<cv::Mat>& images);
		std::vector<cv::Mat> forward_process(cv::dnn::Net& net);

		std::vector<cv::Mat> post_process(std::vector<cv::Mat>& outputs, std::vector<std::string>& class_name);
		void create_detection_file(int X_top, int Y_top, int width, int height);

		void draw_box_prediction(std::vector<std::string>& class_name, int classId, float confidence, int left, int top, int right, int bottom);
		std::vector<std::string> getOutputLayersNames(const cv::dnn::Net& net);	

		cv::Mat getROI(cv::Rect& roi);

};



