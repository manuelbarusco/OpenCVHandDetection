//HandSegmentator.h

//@author: Manuel Barusco, Simone Gregori

#ifndef handSegmentator_hpp
#define handSegmentator_hpp

#include <iostream>
#include <opencv2/core.hpp>

class HandSegmentator{
    private:
      cv::Mat inputImg;                                              //input image to segment
      cv::Mat roi;                                                   //single hand RoI
      cv::Mat edgeMap;                                               //edgeMap of the hand RoI
      cv::Mat colorHands;                                            //input image with colored hands
  	  int numberHands = 0;                                           //number of hand detections in the input image
  	  std::vector<std::pair<cv::Rect,cv::Scalar>> rects;             //hand detections in the input image

      //method for RoI image preprocessing before segmentation
      void preprocessRoI();

      //internal methods for hand segmentation

      //method that return a good initilization mask for the grabcut
      cv::Mat advancedRegionGrowing(unsigned char outputValue);
      cv::Mat handMaskWithARG();

      //method for min max normalization for kmeans based on pixel color and position
      void minMaxNormalization(cv::Mat& img, float weightX, float weightY, bool treeChannels);

      //method for performing kmeans based on pixel color and position
      cv::Mat kmeansSegmentationPositionQuantization(int K, float weighX,float weightY);

      //method for simple segmentation based on skin color thresholding
      void thresholdingYCrCb(cv::Mat& img);

	  void combineSkinAndARG(const cv::Mat& skin, cv::Mat& arg);

      //method that returns the final binary mask from the input image
      void createBinaryMask(cv::Mat& outGC);

	  //method that recognizes if img is black and white
	  int isBlackAndWhite(cv::Mat& img);
	
 	  float computeQualityOfSkinMask(cv::Mat& m, int value);
	
	  cv::Mat thresholdingWithSampledColor(cv::Mat& img);
	
	  void testGrabCutMask(cv::Mat& m); //Only for test


    public:
      //constructor
      HandSegmentator(const cv::Mat& iImg, const int nHands, const std::vector<std::pair<cv::Rect,cv::Scalar>> );

      //main method for image segmentation
      cv::Mat multiplehandSegmentationGrabCutMask();

      //method for colored hands segmention results
      cv::Mat getColoredHands();

      //destructor
      ~HandSegmentator();
};



#endif /* handSegmentation_hpp */
