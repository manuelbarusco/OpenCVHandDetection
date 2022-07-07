#include "HandSegmentator.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int mainSimone(int argc, const char * argv[]) {
    
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

int mainManuel(){
//    Mat img = imread("6.png");
//    HandSegmentator hs = HandSegmentator(img);
//    Mat result = hs.handSegmentation();
//    imshow("Test GrabCut segmentation using mask", result);
//    waitKey();
//    
    return 0;
}

int main(int argc, const char * argv[]) {
    return mainSimone(argc, argv);
    
}

