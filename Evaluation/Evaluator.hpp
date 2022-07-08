//Evaluator.hpp

//@author: Manuel Barusco

#ifndef Evaluator_hpp
#define Evaluator_hpp

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>


class Evaluator{
    private:
        std::string groundTruthDirectory;       //folder that contains the detection and segmentation ground truth files
        std::ofstream outputFile;                //file where to print the evalutation metrics
    
    public:
        //constructor
        Evaluator(std::string gtd, std::string of);
    
        void intersectionOverUnion(std::string imgFileName, std::vector<cv::Rect> detections);
    
        double singleIntersectionOverUnion(const cv::Rect& det, const cv::Rect& bb);
    
        void pixelAccuracy(const cv::Mat imgGT, const cv::Mat maskSegm);
        
        //destrcutor
    
        //excpetion class for invalid syntax in the grount truth files
        class InvalidFile{
            public:
                InvalidFile();
        };

        //excpetion class for empty ground truth files
        class EmptyFile{
            public:
                EmptyFile();
        };
    
        //excpetion class for inexistent ground truth files
        class FileDoesNotExist{
            public:
                FileDoesNotExist();
        };
    
        //excpetion class for output files that are impossible to write
        class ImpossibleWriteFile{
            public:
                ImpossibleWriteFile();
        };
    
};

#endif /* Evaluator_hpp */