//HandDetector.cpp

//@author: Riccardo Rampon

#include "../Include/HandDetector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace cv;
using namespace cv::dnn;
using namespace std;

/** Constructor
@param p_net network object for the hand detection
@param c_names vector of string class names
*/
HandDetector::HandDetector(const Net& p_net, const vector<string>& c_names) {
	net = p_net;
	class_names = c_names;
	//Color to assing to bounging box and segmented hands
	colors = {Scalar(255,178,0), Scalar(0,0,255), Scalar(255,0,0), Scalar(0,0,255), Scalar(76,153,0)};
}//HandDetector

/* forward_process()
* This function will perform the forward-step for the neural netowrk
* @param imgInput input image to the forward process
* @return vector of images after forward process
*/
vector<Mat> HandDetector::forward_process(const Mat& imgInput){
	Mat blob;
	blobFromImage(imgInput, blob, 1 / 255.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(0, 0, 0), true, false);
	net.setInput(blob);

	//forward step
	vector<Mat> outputs;
	net.forward(outputs, getOutputLayersNames(net));

	return outputs;
}//forward_process

/* post_process()
* This function will find possible prediction and keep only the best ones
* @param imgInput input image
* @param outputs outputs of the network after the forward process
* @return vector of pair<Rect,color> where Rect is the bounding box and color is the bounding box color
*/
vector<pair<Rect,Scalar>> HandDetector::post_process(Mat& image, vector<Mat>& outputs){
	vector<int> classIDs;
	vector<float> confidences;
	vector<Rect> boxes;


	for (int i = 0; i < outputs.size(); i++) {

		float* detection = (float*)outputs[i].data;
		int numDetections = outputs[i].rows; // number of detections

		// Search the highest score prediction returned by the network
		for (int j = 0; j < numDetections; ++j, detection += outputs[i].cols){

			// all scores found
			Mat scores = outputs[i].row(j).colRange(5, outputs[i].cols);

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
	vector<int> indices; // Indices of boxes
	NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);
	vector<pair<Rect,Scalar>> out_boxes;
	// Draw box
	for (int i = 0; i < indices.size(); i++) {
		int ind = indices[i];
		Rect bbox = boxes[ind];
		Scalar color = colors[i];
		//refine box in case it goes outside the image
		refineBBox(image, bbox);
		out_boxes.push_back(pair<Rect,Scalar>(bbox,color));
		draw_box_prediction(image, color, classIDs[ind], confidences[ind], bbox.x, bbox.y, bbox.x+bbox.width, bbox.y+bbox.height); // draw box
		create_detection_file(bbox.x, bbox.y, bbox.width, bbox.height);
	}//for

	return out_boxes;
}//post_process

/** method for refining detection boxes in case they go outside the image
@param img input image
@param bbox detection box
*/
void HandDetector::refineBBox(const Mat& img, Rect& bbox){
	//enlarge a little bit the bounding box
	constexpr int inflation = 50;
	bbox += cv::Point(-inflation/2, -inflation/2);
	bbox += cv::Size(inflation, inflation);

	//check cols
	if(bbox.x < 0)
		bbox.x = 0;
	if(bbox.x + bbox.width > img.cols)
		bbox.width = img.cols - bbox.x - 1;

	//check rows
	if(bbox.y < 0)
		bbox.y = 0;
	if(bbox.y + bbox.height > img.rows)
		bbox.height = img.rows - bbox.y -1;
}

/* draw_box_prediction()
* This function will create the box on the image and the label associated
* @param image image on which to create the box
* @param color bounding box color
* @param classId class id prediction
* @param confidence confidence value of the prediction
* @param X_top x-coordinate of top-left point
* @param Y_top y-coordinate of top-left point
* @param X_bottom x-coordinate of bottom-right point
* @param Y_bottom y-coordinate of bottom-right point
*/
void HandDetector::draw_box_prediction(Mat& image, const Scalar& color, int classId, float confidence, int X_top, int Y_top, int X_bottom, int Y_bottom){
	// Create the box
	rectangle(image, Point(X_top, Y_top), Point(X_bottom, Y_bottom), color, 2);
	// Create label and put label + classId
	string label = format("%.f", confidence);
	label = class_names[classId] + ":" + label;
	putText(image, label, Point(X_top, Y_top-10), FONT_TYPE, FONT_SCALE, color, FONT_THICKNESS);

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
