//Evaluator.cpp

//@author: Manuel Barusco, Riccardo Rampon

#include "../Include/Evaluator.hpp"
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

//Classes for handling exceptions

Evaluator::InvalidFile::InvalidFile(){
    cerr << "Ground Truth file is invalid";
} //InvalidFile

Evaluator::EmptyFile::EmptyFile(){
    cerr << "Ground Truth file is emtpy";
} //EmptyFile

Evaluator::FileDoesNotExist::FileDoesNotExist(){
    cerr << "Image groudn truth not present in the ground truth directory";
} //FileDoesNotExist

Evaluator::ImpossibleWriteFile::ImpossibleWriteFile(){
    cerr << "Impossible to write the output file";
} //ImpossibleWriteFile

/** constructor
 @param gtd ground truth directory
 @param of output file path
 */
Evaluator::Evaluator(const string& gtd, const string& of){
    groundTruthDirectory = gtd;
    outputFile.open(of);
    if(!outputFile.is_open())
        throw ImpossibleWriteFile();
}

/** intersection over union
 @param imgFileName string with the image name in order to recover the ground truth file for that image
 @param detections vector of rectangles with the detections returned for that image
 this method will return all the results in the $outputFile
 */
void Evaluator::intersectionOverUnion(const string& imgFileName, const vector<Rect>& detections){
    string imgNameWithFormat = imgFileName.substr(imgFileName.find_last_of("/")+1,imgFileName.size()-1);

    string imgName = imgNameWithFormat.substr(0,imgNameWithFormat.find("."));

    cout << "Computing IoU for image: " << imgName << "\n";

    //open the ground truth file
    ifstream gtf (groundTruthDirectory+"/"+imgName+".txt");

    //check for the ground truth file
    if(!gtf.is_open()){
        gtf.close();
        throw FileDoesNotExist();
    }

    //construct ground truth bounding boxes from the ground truth files
    vector<Rect> gtBB = vector<Rect>();
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

        Rect r= Rect(x,y,w,h);
        gtBB.push_back(r);
        if(first)
            first = false;
    }

    //calculate intersection over union metric
    for(int i = 0; i < gtBB.size(); i++){

        //for every ground truth bounding box calculate the intesection over union between the box and every detector bounding box found in the image
        vector<double> iousLocal = vector<double>();
        for(int j = 0; j < detections.size(); j++){
            double iou = singleIntersectionOverUnion(detections[j], gtBB[i]);
            iousLocal.push_back(iou);
        }

        //pick the best intersection over union, it will be that correlate to the right detection box - ground truth box couple
        double iou = *max_element(iousLocal.begin(), iousLocal.end());

        //print in the output file the IoU
        outputFile << "Image: " << imgFileName <<
                      ", Ground Truth Bounding Box: Coordinates Top Left Pixel: (" << gtBB[i].tl().x << "," << gtBB[i].tl().y << ")" <<
                      ", Width: " << gtBB[i].width <<
                      ", Height: " << gtBB[i].height <<
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
double Evaluator::singleIntersectionOverUnion(const Rect &det, const Rect &bb){
    Rect intersection = det & bb;
    return static_cast<double>(intersection.area()) / (det.area() + bb.area() - intersection.area());
}

/**
@param imgFileName name of the input image
@param maskSegm segmentation mask to evaluate
this method will return all the results in the $outputFile
*/
void Evaluator::pixelAccuracy(const string& imgFileName, const Mat& maskSegm){

    string imgNameWithFormat = imgFileName.substr(imgFileName.find_last_of("/")+1,imgFileName.size()-1);

    string imgName = imgNameWithFormat.substr(0,imgNameWithFormat.find("."));

    cout << "Computing Pixel Accuracy for image: " << imgName << "\n";

    //open the ground truth image
    Mat imgGT = imread(groundTruthDirectory+"/"+imgName+".png", IMREAD_GRAYSCALE);

    //check if the ground truth file is present
    if(imgGT.empty())
        throw FileDoesNotExist();

    //check for the validity of the ground truth
    if(imgGT.cols != maskSegm.cols || imgGT.rows != maskSegm.rows)
        throw InvalidFile();

    int hand_tp = 0; int hand_fp = 0; int hand_tn = 0; int hand_fn = 0;

    unsigned char mask_intensity, maskGT_intensity = 0;

    double accuracy = 0;

    for (int i = 0; i < maskSegm.rows; i++) {
        for (int j = 0; j < maskSegm.cols; j++) {
            mask_intensity = maskSegm.at<unsigned char>(i, j);
            maskGT_intensity = imgGT.at<unsigned char>(i, j);

            if (maskGT_intensity == 255) { //imgGT(i,j) -> white -> hand
                if (mask_intensity == 255)
                    hand_tp++; // both white
                else
                    hand_fn++; //gt white, but mask is black
            } else { //imgGT(i,j) -> black
                if (mask_intensity == 0)
                    hand_tn++; // tutti e due black
                else
                    hand_fp++; //gt � black e predetto � white
            }//if-else
        }//for
    }//for

    accuracy = static_cast<double>(hand_tp + hand_tn) / (maskSegm.cols * maskSegm.rows);

    //print in the output file the IoU
    outputFile << "Image: " << imgFileName <<
                  " Pixel Accuracy: " << accuracy << "\n";

    //release ground truth file
    imgGT.release();
}//pixelAccuracy

//destructor for resources deallocation
Evaluator::~Evaluator(){
    outputFile.close();
}
