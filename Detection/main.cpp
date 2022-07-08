
#include "HandDetector.h"
#include<stdio.h>
#include <iostream>
#include <fstream>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main() {
	Mat img = imread("model/rgb/10.jpg");

	String path_file = "model/output_detection/det1";
	//path_file.push_back(to_string(i)) //per altri file: uno per ogni immagine
	path_file.append(".txt");


	HandDetector hd = HandDetector(img, path_file);

	// Configuration & Weights path
	String cfg_path = "model/yolov3_training.cfg";
	String weights_path = "model/yolov3_training_last_v7.weights";

	// Neural Network model
	Net net = readNetFromDarknet(cfg_path, weights_path);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Load name of class
	string classesFile = "model/coco2.names";
	vector<string> classes;
	ifstream fin (classesFile.c_str());
	if (fin.is_open()) {
		string line;
		while (getline(fin, line)) classes.push_back(line);
		cout << line << endl;
		fin.close();
	}//if

	vector<Mat> outs = hd.forward_process(net);

	vector<Mat> out_imgs = hd.post_process(outs, classes);

	hd.show_images(out_imgs);

	return 0;

}//main