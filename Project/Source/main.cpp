//main.cpp

#include "../Include/HandDetector.hpp"
#include "../Include/HandSegmentator.hpp"
#include "../Include/Evaluator.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;

/**
@author Manuel Barusco, Simone Gregori, Riccardo Rampon
*/


/** configureDetector
@return HandDetector configured
*/
HandDetector configureDetector(){
    // Configuration & Weights path
    String cfg_path = "../Models/yolov3_training.cfg";
    String weights_path = "../Models/yolov3_training_last_v7.weights";

    // Neural Network model
    Net net = readNetFromDarknet(cfg_path, weights_path);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Load name of class
    string classesFile = "../Models/coco2.names";
    vector<string> classes;
    ifstream fin (classesFile.c_str());
    if (fin.is_open()) {
      string line;
      while (getline(fin, line)) classes.push_back(line);
      //cout << line << endl;
      fin.close();
    }//if

    return HandDetector(net, classes);
}

/** function for only detection mode
@param pathFolder images directory path
*/
void runDetector(const string& pathFolder){
    vector<string> imgs;
    glob(pathFolder, imgs, false);


    HandDetector hd = configureDetector();

    for(int i = 0; i < imgs.size(); i++){
      Mat img = imread(imgs[i]);

      vector<Mat> outs = hd.forward_process(img);

      vector<pair<Rect,Scalar>> outDetections = hd.post_process(img, outs);

      imshow("Detection", img);
      waitKey();
    }
}

/** function for only segmentation mode
@param pathFolder images directory path
*/
void runSegmentator(const string& pathFolder){
    vector<string> imgs;
    glob(pathFolder, imgs, false);

    HandDetector hd = configureDetector();

    for(int i = 0; i < imgs.size(); i++){
        if(i>= 19){
            Mat imgD = imread(imgs[i]);
            Mat imgS = imgD.clone();
            vector<Mat> outs = hd.forward_process(imgD);

            vector<pair<Rect,Scalar>> outDetections = hd.post_process(imgD, outs);
            imshow("Detection", imgD);
            waitKey();

            HandSegmentator s (imgS, outDetections.size(), outDetections);

            Mat out = s.multiplehandSegmentationGrabCutMask();
            imshow("Segmentation", out);
            waitKey();

            imgD.release();
            imgS.release();
            out.release();
        }
    }
}

/** function for detection+evaluation mode
@param pathFolder images directory path
@param gtPath input images ground truth path
*/
void runDetectorWithEvaluator(const string& pathFolder, const string& gtPath){
    vector<string> imgs;
    glob(pathFolder, imgs, false);

    HandDetector hd = configureDetector();
    Evaluator e (gtPath, "../Test/output_evaluation/resultsDetection.txt");

    for(int i = 0; i < imgs.size(); i++){
      Mat img = imread(imgs[i]);

      vector<Mat> outs = hd.forward_process(img);

      vector<pair<Rect,Scalar>> outDetections = hd.post_process(img, outs);

      string imgNameWithFormat = imgs[i].substr(imgs[i].find_last_of("/")+1,imgs[i].size()-1);
      imwrite("../Test/output_detection/"+imgNameWithFormat, img);

      vector<Rect> detections = vector<Rect>();
      for(int i = 0; i < outDetections.size(); i++)
          detections.push_back(std::get<0>(outDetections[i]));

      e.intersectionOverUnion(imgs[i], detections);
    }
}

/** function for only segmentation+evaluation mode
@param pathFolder images directory path
@param gtPath input images ground truth path
*/
void runSegmentatorWithEvaluator(const string& pathFolder, const string& gtPath){
    vector<string> imgs;
    glob(pathFolder, imgs, false);

    HandDetector hd = configureDetector();
    Evaluator e (gtPath, "../Test/output_evaluation/resultsSegmentation.txt");

    for(int i = 0; i < imgs.size(); i++){
        Mat img = imread(imgs[i]);
        Mat imgS = img.clone();

        cout << "Segmenting image: " << imgs[i] << "\n";
        vector<Mat> outs = hd.forward_process(img);

        vector<pair<Rect,Scalar>> outDetections = hd.post_process(img, outs);

        HandSegmentator s (imgS, outDetections.size(), outDetections);
        Mat outSegmentation = s.multiplehandSegmentationGrabCutMask();

        string imgNameWithFormat = imgs[i].substr(imgs[i].find_last_of("/")+1,imgs[i].size()-1);
        imwrite("../Test/output_segmentation/"+imgNameWithFormat, outSegmentation);

        img.release();
        imgS.release();
        outSegmentation.release();
    }

    glob("../Test/output_segmentation", imgs, false);
    for(int i = 0; i < imgs.size(); i++){
        Mat mask = imread(imgs[i], IMREAD_GRAYSCALE);
        e.pixelAccuracy(imgs[i], mask);
        mask.release();
    }
}

int userMain(){
    //selection of the path to the input images

    cout << "Please insert the input images directory path:";
    string path;
    cin >> path;

    /*std::filesystem::path::path_object (path);
    if(!std::filesystem::exists(path))
        cerr << "Input images directory path is wrong or does not exists";*/

    struct stat buffer;
    if(!((stat (path.c_str(), &buffer)) == 0))
        cerr << "Input images directory path is wrong or does not exists";

    //selection of the user execution mode

    cout << "Please select one of the following execution modes: " << "\n"
         << "- d: only detection" << "\n"
         << "- ds: detection and segmentation" << "\n"
         << "- de: detection with evaluation" << "\n"
         << "- se: segmentation with evaluation" << "\n"
         << "Insert the execution mode (d, ds, de or se):";
    string mode;
    cin >> mode;

    if(mode.compare("d") == 0){
        runDetector(path);
    } else if (mode.compare("ds") == 0){
        runSegmentator(path);
    } else if (mode.compare("de") == 0){
      string gtPath;
      cout << "Please insert the input images ground truth directory path for detection task:";
      cin >> gtPath;
      runDetectorWithEvaluator(path, gtPath);
    } else if (mode.compare("se") == 0){
      string gtPath;
      cout << "Please insert the input images ground truth directory path for segmentation task:";
      cin >> gtPath;
      runSegmentatorWithEvaluator(path, gtPath);
    } else {
      cerr << "Insert a valid execution mode";
      return 1;
    }


    return 0;
}

// final main
int main(){
    userMain();
    return 0;
}
