//Evaluator.hpp

//@author: Manuel Barusco, Riccardo Rampon

#ifndef Evaluator_hpp
#define Evaluator_hpp

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>

//Evalutor object for detection and segmentation evaluation (using Intersection Over Union and Pixel Accuracy)
class Evaluator{
    private:
        std::string groundTruthDirectory;        //string with the path of the directory that contains the detection and segmentation ground truth files and directories
        std::ofstream outputFile;                //file where to print the evalutation metrics

        //method for computing one single Inteserction Over Union measure
        double singleIntersectionOverUnion(const cv::Rect& det, const cv::Rect& bb);
    public:
        //constructor
        Evaluator(const std::string& gtd, const std::string& of);

        //method for computing the Inteserction Over Union metric
        void intersectionOverUnion(const std::string& imgFileName, const std::vector<cv::Rect>& detections);

        //method for computing Pixel Accuracy metric
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

        //destructor for resources deallocation
        ~Evaluator();

};

#endif /* Evaluator_hpp */
