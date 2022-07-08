//HandDetector.cpp

//Author: Riccardo Rampon

#include "HandDetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace cv;
using namespace cv::dnn;
using namespace std;

/** Constructor */
HandDetector::HandDetector(const Mat& img, const string path_file) {
	image = img.clone();
	//outF(path_file, ios::out);
	outF = ofstream(path_file, ios::out);
	
}//HandDetector

/* forward_process()
* This function will perform the forward-step for the neural netowrk
* @param outs output of the network
* @param classes vector of classes
* @return image post-processed
*/
vector<Mat> HandDetector::forward_process(Net& net){
	Mat blob;
	blobFromImage(image, blob, 1 / 255.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(0, 0, 0), true, false);
	net.setInput(blob);
	
	//forward step
	vector<Mat> outputs;
	net.forward(outputs, getOutputLayersNames(net));

	return outputs;
}//forward_process

/* post_process()
* This function will find possible prediction and keep only the best one
* @param outs outputs of the network
* @param classes vector of classes
* @return image post-processed
*/
vector<Mat> HandDetector::post_process(vector<Mat>& outs, vector<String>& classes){
	vector<Mat> out_images;
	vector<int> classIDs;
	vector<float> confidences;
	vector<Rect> boxes;

	out_images.push_back(image);

	for (int i = 0; i < outs.size(); i++) {

		float* detection = (float*)outs[i].data;
		int numDetections = outs[i].rows; // number of detections

		// Search the highest score prediction returned by the network
		for (int j = 0; j < numDetections; ++j, detection += outs[i].cols){
			
			// all scores found
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
	
			Point classIdPoint;
			double confidence;

			// Get the value and point location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

			// Consider only prediction that have confidence >= threshold 
			if (confidence > CONFIDENCE_THRESHOLD) {

				// Compute all parameter needed to perform boxes' creation
				int center_X = static_cast<int>(detection[0] * image.cols);
				int center_Y = static_cast<int>(detection[1] * image.rows);
				int width = static_cast<int>(detection[2] * image.cols);
				int height = static_cast<int>(detection[3] * image.rows);
				int X_top = static_cast<int>(center_X - width/2);
				int Y_top = static_cast<int>(center_Y - height/2);

				// Push back classID, confidence, box to respectively vectors
				classIDs.push_back(classIdPoint.x);
				confidences.push_back(static_cast<float>(confidence));
				boxes.push_back(Rect(X_top, Y_top, width, height));
			}//if
		}//for
	}//for

	// Non-Maxima-Suppression used to eliminate all redundant and overlapping boxes that have low confidence
	vector<int> indices; // Indices of bboxes
	NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

	// Draw box
	for (int i = 0; i < indices.size(); i++) {
		
		int ind = indices[i];
		Rect bbox = boxes[ind];

		Mat roi = getROI(bbox);
		out_images.push_back(roi);

		draw_box_prediction(classes, classIDs[ind], confidences[ind], bbox.x, bbox.y, bbox.x+bbox.width, bbox.y+bbox.height); // draw box
		//printf("X:%d, Y:%d, W:%d, H:%d\n\n", bbox.x, bbox.y, bbox.width, bbox.height);
		create_detection_file(bbox.x, bbox.y, bbox.width, bbox.height);
	}//for

	return out_images;
}//forward_process

/* draw_box_prediction()
* This function will create the box on the image and the label associated
* @param image image on which to create the box
* @param classes vector of classes
* @param classId class id prediction
* @param confidence confidence value of the prediction
* @param X_top x-coordinate of top-left point
* @param Y_top y-coordinate of top-left point
* @param X_bottom x-coordinate of bottom-right point
* @param Y_bottom y-coordinate of bottom-right point
*/
void HandDetector::draw_box_prediction(vector<String>& classes, int classId, float confidence, int X_top, int Y_top, int X_bottom, int Y_bottom){
	// Create the box
	rectangle(image, Point(X_top, Y_top), Point(X_bottom, Y_bottom), COLOR, 2);
	
	// Create label and put label + classId
	string label = format("%.f", confidence);
	label = classes[classId] + ":" + label;
	putText(image, label, Point(X_top, Y_top-10), FONT_TYPE, FONT_SCALE, COLOR, FONT_THICKNESS);

}//draw_box_prediction

/* getOutputLayersNames()
* This function identify the last layer of the network and provides the associated name, in order to do the forward-step in forward_process() function
* @param net yolo neural network model
* @return names associated to last layer of the network
*/
vector<String> HandDetector::getOutputLayersNames(const Net& net){
	vector<string> names;
	vector<int> outLayers = net.getUnconnectedOutLayers(); //get the indeces of output layers
	vector<string> layersNames = net.getLayerNames(); // get names of all layers in the network
	names.resize(outLayers.size());
	for (int i = 0; i < outLayers.size(); i++) { // get the names of the output layers in names
		names[i] = layersNames[outLayers[i] - 1];
	}//for
	return names;
}//getOutputLayersNames


cv::Mat HandDetector::getROI(Rect& roi){
	//imshow("roi", image(roi));
	//waitKey();
	return image(roi);
}//getROI


/* create_detection_file
* This function will create a file that contains the coordinates needed to identify the boxes and to perform after the evaluation
* @param X_top x-coordinate of the box
* @param Y_top y-coordinate of the box
* @param width width of the box
* @param height height of the box 
*/
void HandDetector::create_detection_file(int X_top, int Y_top, int width, int height){
	if (outF.is_open()){
		outF << X_top << " " << Y_top << " " << width << " " << height << endl;
	}//if
	//outF.close();
}//create_detection_file

/** show_image()
* This function will display the images
*/
void HandDetector::show_images(const vector<Mat>& imgs) {
	imshow("Image", imgs[0]);
	for (int i = 1; i < imgs.size(); i++) {
		String windName = "Roi";
		windName.append(to_string(i));
		imshow(windName, imgs[i]);
	}//for
	waitKey();
}//show_image