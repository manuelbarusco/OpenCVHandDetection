//HandSegmentator.h

//Authors: Manuel Barusco, Simone Gregori

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
    cv::Mat edgeMap;
    cv::Mat preprocessedImage;
    
    //internal methods for segmentation
    cv::Mat regionGrowing(const std::vector<std::pair<int, int>>& seedSet, unsigned char outputValue, float tolerance);
    
    
    cv::Mat kmeans(int k, int att, cv::Mat& centers);
    
    cv::Mat thresholdingYCrCb(cv::Mat &img);
    
    void minMaxNormalization(cv::Mat& img, float weightX, float weightY, bool treeChannels);
    
    cv::Mat kmeansSegmentationPositionQuantization(int K, float weighX,float weightY);
    
    cv:: Mat thresholdingYCrCb();
    
    void preprocessImage();
public:
    //constructor
    HandSegmentator(const cv::Mat& roi);
   
    cv::Mat handSegmentation();
};



#endif /* handSegmentation_hpp */