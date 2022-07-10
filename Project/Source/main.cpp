//main.cpp

#include "../Include/HandDetector.hpp"
#include "../Include/HandSegmentator.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

/**
@author Manuel Barusco, Simone Gregori, Riccardo Rampon
*/

//main for detection module
int mainDetector(){
  	Mat img = imread("../Test/rgb/24.jpg");

  	String path_file = "Test/output_detection/det1";
  	//path_file.push_back(to_string(i)) //per altri file: uno per ogni immagine
  	path_file.append(".txt");


  	HandDetector hd = HandDetector(img, path_file);

  	// Configuration & Weights path
  	String cfg_path = "../Models/yolov3_training.cfg";
  	String weights_path = "../Models/yolov3_training_last_v7.weights";

  	// Neural Network model
  	Net net = readNetFromDarknet(cfg_path, weights_path);
  	net.setPreferableBackend(DNN_BACKEND_OPENCV);
  	net.setPreferableTarget(DNN_TARGET_CPU);

  	// Load name of class
  	string classesFile = "../Models/coco2.names";
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
}

//main for segmentation module
int mainSegmentation(int argc, const char * argv[]) {

    //Full size image
    Mat img = imread(argv[1]);
    Mat result;

    //TODO: add bounding box coordinates in a vector<Rect>

    vector<Rect> boxs;
    //boxs.push_back(Rect(438, 296, 205, 160));     boxs.push_back(Rect(624,318,226,128));         	//01.jpg
    boxs.push_back(Rect(129, 61, 62, 90));     boxs.push_back(Rect(223,98,92,63));            		//27.jpg
	//boxs.push_back(Rect(142, 41, 39, 65));     boxs.push_back(Rect(10,165,68,36)); 					//22.jpg
	//boxs.push_back(Rect(90, 111, 140, 89));     boxs.push_back(Rect(130,103,125,109)); 				//25.jpg

	//Test rect coordinates on image
	Mat imgRects = img.clone();
	for(int i = 0; i<boxs.size(); i++){
		rectangle(imgRects, boxs[i], cv::Scalar(0, 255, 0));
	}
	imshow("Test position boxs", imgRects);
	waitKey();

    HandSegmentator hs = HandSegmentator(img, boxs.size(), boxs);

    //GrabCut using mask
    result = hs.MiltiplehandSegmentationGrabCutMask();
    imshow("Test GrabCut segmentation using mask", result);
    waitKey();

    //GrabCut using Rect
    result = hs.MiltiplehandSegmentationGrabCutRect();
    imshow("Test GrabCut segmentation using Rect", result);
    waitKey();

    return 0;

}

/** function for only detection mode
@param directory images directory path
*/
void runDetector(const string& pathFolder){
    /*vector<string> imgs;
    glob(pathFolder, imgs, false);

    for(int i = 0; i < imgs.size(); i++){
      Mat img = imread(pathFolder+"/"+imgs[i]);

      HandDetector hd = HandDetector(img,);

      // Configuration & Weights path
      String cfg_path = "../Models/yolov3_training.cfg";
      String weights_path = "../Models/yolov3_training_last_v7.weights";

      // Neural Network model
      Net net = readNetFromDarknet(cfg_path, weights_path);
      net.setPreferableBackend(DNN_BACKEND_OPENCV);
      net.setPreferableTarget(DNN_TARGET_CPU);

      // Load name of class
      string classesFile = "../Models/coco2.names";
      vector<string> classes;
      ifstream fin (classesFile.c_str());
      if (fin.is_open()) {
        string line;
        while (getline(fin, line)) classes.push_back(line);
        //cout << line << endl;
        fin.close();
      }//if

      vector<Mat> outs = hd.forward_process(net);

      vector<Mat> out_imgs = hd.post_process(outs, classes);

      hd.show_images(out_imgs);
    }*/
}

int userMain(){
    //selection of the path to the input images

    cout << "Insert the input images directory path:";
    string path;
    cin >> path;

    //selection of the user execution mode

    cout << "Please select one of the following execution modes: " << "\n"
         << "- d: only detection" << "\n"
         << "- ds: detection and segmentation" << "\n"
         << "- de: detection with evaluation" << "\n"
         << "- se: segmentation with evaluation" << "\n"
         << "Insert the execution mode (d, ds, de or se):";
    string mode;
    cin >> mode;

    if(mode.compare("d") == 0){
        mainDetector();
    } else if (mode.compare("ds") == 0){
       // execuet segmentation
    } else if (mode.compare("de") == 0){
      // execute detection with evaluation
    } else if (mode.compare("se") == 0){
      //execute segmentation with evaluation2
    } else {
      cerr << "Insert a valid execution mode";
      return 1;
    }


    return 0;
}

// final main
int main(){
    userMain();
    return 0;
}
