//HandSegmentator.h

//@author: Manuel Barusco, Simone Gregori

#ifndef handSegmentator_hpp
#define handSegmentator_hpp

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

class HandSegmentator{
    private:
      cv::Mat inputImg;
      cv::Mat roi;                                                   //single hand RoI
      cv::Mat edgeMap;                                               //edgeMap of the hand RoI
  	  int numberHands = 0;                                           //number of hand detections in the input image
  	  std::vector<std::pair<cv::Rect,cv::Scalar>> rects;             //hand detections in the input image

      //method for RoI image preprocessing before segmentation

      void preprocessRoI();

      //internal methods for hand segmentation

      cv::Mat advancedRegionGrowing(unsigned char outputValue);

      void minMaxNormalization(cv::Mat& img, float weightX, float weightY, bool treeChannels);

      cv::Mat kmeansSegmentationPositionQuantization(int K, float weighX,float weightY);

      void thresholdingYCrCb(cv::Mat& img);

  	  cv::Mat setGrabCutFlag(const cv::Mat& maskPR, const cv::Mat& mask, int flagDefault, int flagTrue, int flagPR_True);

      cv::Mat handMaskWithARG();

      void createBinaryMask(cv::Mat& outGC);


    public:
      //constructor
      HandSegmentator(const cv::Mat& iImg, const int nHands, const std::vector<std::pair<cv::Rect,cv::Scalar>> );

      cv::Mat multiplehandSegmentationGrabCutMask();

      //destructor
      ~HandSegmentator();
};



#endif /* handSegmentation_hpp */
