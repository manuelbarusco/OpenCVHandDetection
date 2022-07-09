//Evaluator.cpp

//@author: Manuel Barusco, Riccardo Rampon

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
    cout << groundTruthDirectory+"/det/"+imgName+".txt";
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
    cv::Rect intersection = det & bb;
    return static_cast<double>(intersection.area()) / (det.area() + bb.area() - intersection.area());
}

void Evaluator::pixelAccuracy(std::string imgFileName, const cv::Mat imgGT, const cv::Mat maskSegm){

    string imgName = imgFileName.substr(0, imgFileName.find("."));
    ifstream gtf(groundTruthDirectory + "/det/" + imgName + ".txt");


    int hand_tp = 0; int hand_fp = 0; int hand_tn = 0; int hand_fn = 0;
    //int no_hand_tp = 0; int no_hand_fp = 0; int no_hand_tn = 0; int no_hand_fn = 0;

    unsigned char mask_intensity, maskGT_intensity = 0;
    float hand_pixel_accuracy = 0;
    float no_hand_pixel_accuracy = 0;

    if (imgGT.channels() != 1) cvtColor(imgGT, imgGT, COLOR_BGR2GRAY);
    if (maskSegm.channels() != 1) cvtColor(maskSegm, maskSegm, COLOR_BGR2GRAY);

    for (int i = 0; i < maskSegm.rows; i++) {
        for (int j = 0; j < maskSegm.cols; j++) {
            mask_intensity = maskSegm.at<unsigned char>(i, j);
            maskGT_intensity = imgGT.at<unsigned char>(i, j);

            if (maskGT_intensity == 255) { //imgGT(i,j) -> white -> hand
                if (mask_intensity == 255)
                    hand_tp++; //tutti e due white
                else
                    hand_fn++; //gt è white quello predetto è black
            }
            else { //imgGT(i,j) -> black
                if (mask_intensity == 0)
                    hand_tn++; // tutti e due black
                else
                    hand_fn++; //gt è black e predetto è white
            }//if-else
        }//for
    }//for

    hand_pixel_accuracy = static_cast<float>(hand_tp / (hand_tp + hand_fn));
    no_hand_pixel_accuracy = static_cast<float>(hand_tn / (hand_tn + hand_fp));

    //print in the output file the IoU
    outputFile << "Image: " << imgFileName <<
        "Pixel Accuracy (Hand): " << hand_pixel_accuracy << "\n" <<
        "Pixel Accuracy (No Hand): " << no_hand_pixel_accuracy << "\n";

    //close the input file
    gtf.close();

}//pixelAccuracy

/** destroyer: free all the evaluator resources
 */
Evaluator::~Evaluator(){
    outputFile.close();
}
