//Evaluator.cpp

//@author: Manuel Barusco

#include "Evaluator.hpp"
#include <fstream>
#include <opencv2/core.hpp>

using namespace std;

Evaluator::InvalidFile::InvalidFile(){
    cerr << "Ground Truth file is invalid";
} //InvalidFile

Evaluator::EmptyFile::EmptyFile(){
    cerr << "Ground Truth file is emtpy";
} //EmptyFile

Evaluator::FileDoesNotExist::FileDoesNotExist(){
    cerr << "Image not present in the ground truth dataset";
} //InvalidFile

Evaluator::ImpossibleWriteFile::ImpossibleWriteFile(){
    cerr << "Impossible to write the output file";
}

/** constructor
 @param gtd ground truth directory
 @param of output file path
 */
Evaluator::Evaluator(string gtd, string of){
    groundTruthDirectory = gtd;
    outputFile.open(of);
    if(!outputFile.is_open())
        throw ImpossibleWriteFile();
}

/** intersection over union
 @param imgFileName string with the image name in order  to recover the ground truth for that image
 @param detections vector of rectangles with the detections of that image
 */
void Evaluator::intersectionOverUnion(string imgFileName, vector<cv::Rect> detections){
    string imgName = imgFileName.substr(0, imgFileName.find("."));
    ifstream gtf (groundTruthDirectory+"/det/"+imgName+".txt");
    
    //check for the file
    if(!gtf.is_open()){
        gtf.close();
        throw FileDoesNotExist();
    }
    
    //construct ground truth bounding boxes
    vector<cv::Rect> gtBB = vector<cv::Rect>(); 
    string line;
    bool first = true;
    while(getline(gtf,line)){
        stringstream ss(line);
        
        int x,y,h,w;
        
        ss >> x;
        ss >> y;
        ss >> w;
        ss >> h;
        
        //check for empty file
        if(x==EOF && first)
            throw EmptyFile();
        
        //check for invalid files
        if(x == EOF || y == EOF || w == EOF || h == EOF )
            throw InvalidFile();
        
        cv::Rect r= cv::Rect(x,y,w,h);
        gtBB.push_back(r);
        if(first)
            first = false;
    }
    
    //calculate intersection over union
    for(int i = 0; i < detections.size(); i++){
        
        //for every detection calculate the intesection over union between the detection box and every ground truth box
        vector<double> iousLocal = vector<double>();
        for(int j = 0; j < gtBB.size(); j++){
            double iou = singleIntersectionOverUnion(detections[i], gtBB[j]);
            iousLocal.push_back(iou);
        }
        
        //pick the best intersection over union, it will be that correlate to the right detection box - ground truth box couple
        double iou = *max_element(iousLocal.begin(), iousLocal.end());
        
        //print in the output file the IoU
        outputFile << "Image: " << imgFileName <<
                      "Bounding Box: Coordinates Top Left Pixel: (" << detections[i].tl().x << "," << detections[i].tl().y << ")" <<
                      ", Height: " << detections[i].height <<
                      ", Width: " << detections[i].width <<
                        ", IoU: " << iou << "\n";
    }
    
    //close the input file
    gtf.close();
}

/**
 @param det detection rectangle
 @param bb detection bounding box ground truth
 @return intersection over union of det and bb rectangles
 */
double Evaluator::singleIntersectionOverUnion(const cv::Rect &det, const cv::Rect &bb){
    //TODO: codice per il calcolo della metrica
}

