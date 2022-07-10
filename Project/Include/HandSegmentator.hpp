//HandSegmentator.h

//@author: Manuel Barusco, Simone Gregori

#ifndef handSegmentator_hpp
#define handSegmentator_hpp

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

class HandSegmentator{
    private:
    //image to be segmented
    cv::Mat inputRoi;
	cv::Mat fullImg;
    cv::Mat edgeMap;
    cv::Mat preprocessedImage;
	int numberHands,isFullimgSet = 0;
	std::vector<cv::Rect> rects;

    //internal methods for segmentation
    cv::Mat regionGrowing(const std::vector<std::pair<int, int>>& seedSet, unsigned char outputValue, float tolerance);


    cv::Mat kmeans(int k, int att, cv::Mat& centers);

    cv::Mat thresholdingYCrCb(cv::Mat &img);

    void minMaxNormalization(cv::Mat& img, float weightX, float weightY, bool treeChannels);

    cv::Mat kmeansSegmentationPositionQuantization(int K, float weighX,float weightY);

    cv:: Mat thresholdingYCrCb();

    void preprocessImage();

	cv::Mat setGrabCutFlag(cv::Mat maskPR, cv::Mat mask, int flagDefault, int flagTrue, int flagPR_True);
public:
    //constructor
    HandSegmentator(const cv::Mat& roi, const int nHands, std::vector<cv::Rect>);

    cv::Mat handSegmentation();
    cv::Mat MiltiplehandSegmentationGrabCutRect();
    cv::Mat MiltiplehandSegmentationGrabCutMask();
};



#endif /* handSegmentation_hpp */
