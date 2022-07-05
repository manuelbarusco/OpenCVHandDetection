#include "HandSegmentator.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
	
	Mat img = imread(argv[1]);
	
	//Rect syntax:  	Rect (x, y, width, height)   (x,y) = coordinate top left corner 
	int numberHands = 2;
	Mat inputRoi[numberHands], results[numberHands];
	//TODO: use coordinate returned by NN
	//Rect rects[] =  {Rect(438, 296, 205, 160), Rect(624,318,226,128)};		//01.jpg
	Rect rects[] =  {Rect(129, 61, 62, 90), Rect(223,98,92,63)};			//27.jpg
	//Test rect coordinates on image 
	Mat imgRects = img.clone();
	for(int i = 0; i<numberHands; i++){
		cv::rectangle(imgRects, rects[i], cv::Scalar(0, 255, 0));
	}
	imshow("Test position rects", imgRects);
	waitKey();
	
	//Create vector of images cropped in ROI
	for(int i = 0; i<numberHands; i++){
		//Crop the image using rectangle 
		inputRoi[i] = img(rects[i]);
	}
	
	
	for(int i = 0; i<numberHands; i++){
		//run the "standard" segmentation on cropped images
		imshow("inputRoi[i]", inputRoi[i]);
		waitKey(0);
		HandSegmentator hs = HandSegmentator(inputRoi[i]);	//TODO: sistemare in modo da usare un solo HandSegmentator: implementare tutto dentro la classe
		results[i] = hs.handSegmentation();
		
		destroyAllWindows();
	}
	
	//GrabCut with mask
//	HandSegmentator hsGrapCut = HandSegmentator(img);
//	//cout<<"Type mask input of GranCut: "<<results[0].type()<<" Size: "<<results[0].size()<<endl;
//	Mat testGrab = hsGrapCut.handSegmentationGrabCutMask(results,rects,numberHands);
//	imshow("Test GrabCut segmentation using mask", testGrab);
//	waitKey();
	
	//GrabCut with rect
	HandSegmentator hsGrapCutR = HandSegmentator(img);
	Mat testGrabR = hsGrapCutR.handSegmentationGrabCutRect(rects,numberHands);
	imshow("Test GrabCut segmentation using mask", testGrabR);
	waitKey();
	
	return 0;
			
}
