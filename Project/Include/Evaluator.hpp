//Evaluator.hpp

//@author: Manuel Barusco, Riccardo Rampon

#ifndef Evaluator_hpp
#define Evaluator_hpp

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>

//Evalutor class for detection and segmentation evaluation
class Evaluator{
    private:
        std::string groundTruthDirectory;        //folder that contains the detection and segmentation ground truth files and directories
        std::ofstream outputFile;                //file where to print the evalutation metrics

        double singleIntersectionOverUnion(const cv::Rect& det, const cv::Rect& bb);
    public:
        //constructor
        Evaluator(const std::string& gtd, const std::string& of);

        //method for inteserction over union metric
        void intersectionOverUnion(const std::string& imgFileName, std::vector<cv::Rect>& detections);

        //method for pixel accuracy metric
        void pixelAccuracy(const std::string& imgFileName,const cv::Mat& maskSegm);

        //excpetion class for invalid syntax in the ground truth files
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
