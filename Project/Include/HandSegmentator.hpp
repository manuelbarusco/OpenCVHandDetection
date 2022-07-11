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
      cv::Mat inputImg;
      cv::Mat roi;                             //singe hand RoI
      cv::Mat edgeMap;                         //edgeMap of the input image
      cv::Mat preprocessedImage;               //processed image for image segmentation
  	  int numberHands,isFullimgSet = 0;
  	  std::vector<cv::Rect> rects;             //hand detections in the input image

      //method for input image preprocessing before segmentation

      void preprocessImage();

      //internal methods for hand segmentation

      cv::Mat advancedRegionGrowing(const std::vector<std::pair<int, int>>& seedSet, unsigned char outputValue);

      void minMaxNormalization(cv::Mat& img, float weightX, float weightY, bool treeChannels);

      cv::Mat kmeansSegmentationPositionQuantization(int K, float weighX,float weightY);

      cv:: Mat thresholdingYCrCb();

  	  cv::Mat setGrabCutFlag(cv::Mat maskPR, cv::Mat mask, int flagDefault, int flagTrue, int flagPR_True);

      cv::Mat handSegmentationWithARG();
    public:
      //constructor
      HandSegmentator(const cv::Mat& iImg, const int nHands, std::vector<cv::Rect>);

      cv::Mat MiltiplehandSegmentationGrabCutRect();
      cv::Mat MiltiplehandSegmentationGrabCutMask();
};



#endif /* handSegmentation_hpp */
