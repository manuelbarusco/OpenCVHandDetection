//HandSegmentator.cpp

//@author: Manuel Barusco, Simone Gregori


#include "../Include/HandSegmentator.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <set>

using namespace std;
using namespace cv;

/** constructor
 @param iImg input img to be segmented
 @param nHands Number of hands in inputImg
 @param r Vector of Rect of size nHands, it contains all the hand detections in iImg
 */
HandSegmentator::HandSegmentator(const Mat& iImg, const int nHands, const vector<pair<Rect,Scalar>> r){
    inputImg = iImg;
    numberHands = nHands;
    rects = r;
}

/** Min-Max normalization for kmeans based on pixel color and position
@param img input img
@param weightX weight for the x-position component in the feature vector
@param weightY weight for the y-position component in the feature vector
@param threeChannels boolean that indicates if img is a 3-channels img
*/
void HandSegmentator::minMaxNormalization(Mat &img, float weightX, float weightY, bool threeChannels){
    int dim = 3;
    if (threeChannels)
        dim = 5;
    int weights[dim];
    if (threeChannels){
        weights[0]= 1; weights[1]= 1; weights[2]= 1; weights[3] = weightY; weights[4] = weightX;
    }
    else{
        weights[0]= 1; weights[1] = weightY; weights[2] = weightX;
    }

    float min[dim],max[dim];

    for (int i = 0; i < dim; i++) {
        min[i] = img.at<float>(0,i);
        max[i] = img.at<float>(0,i);
    }

    for( int y = 0; y < img.rows; y++ ){
        for (int x = 0; x < img.cols; x++) {
            if (img.at<float>(y,x) > max[x]) {
                max[x] = img.at<float>(y,x);
            }
            if (img.at<float>(y,x) < min[x]) {
                min[x] = img.at<float>(y,x);
            }
        }
    }

    for( int y = 0; y < img.rows; y++ ){
        for( int x = 0; x < img.cols; x++ ){
            img.at<float>(y,x) = (img.at<float>(y,x) - min[x])/((max[x] - min[x])*weights[x]);
        }
    }
}

/** method for k-means clustering based on pixel color and position
@param K number of clusters
@param weightX weight for the x-position component in the feature vector
@param weightY weight for the y-position component in the feature vector
@return img segmented
*/
Mat HandSegmentator::kmeansSegmentationPositionQuantization(int K, float weightX,float weightY){
    Mat labels, centers;

    //Conver to float for kmeans
    roi.convertTo(roi, CV_32FC3, 1.0/255.0);

    Mat points(roi.rows * roi.cols, 5, CV_32FC1);
     for( int y = 0; y < roi.rows; y++ )
         for( int x = 0; x < roi.cols; x++ ){
             for(int z = 0; z < 3; z++)
                 points.at<float>(y + x*roi.rows, z) = roi.at<Vec3f>(y,x)[z];
                 points.at<float>(y + x*roi.rows, 3) = (float) y ;
                 points.at<float>(y + x*roi.rows, 4) = (float) x;
         }

    minMaxNormalization(points,weightX,weightY,true);

    int attempts = 10;
    kmeans(points, K, labels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 100000, 0.00001 ), attempts, KMEANS_PP_CENTERS, centers );

    Mat out( roi.size(), CV_32FC3 );
    for( int y = 0; y < roi.rows; y++ )
      for( int x = 0; x < roi.cols; x++ ){
          int cluster_idx = labels.at<int>(y + x*roi.rows,0);
            out.at<Vec3f>(y,x)[0] = centers.at<float>(cluster_idx, 0)*255;
            out.at<Vec3f>(y,x)[1] = centers.at<float>(cluster_idx, 1)*255;
            out.at<Vec3f>(y,x)[2] = centers.at<float>(cluster_idx, 2)*255;
      }
    out.convertTo(out, CV_8UC3);
	//Convert roi back to unsigned char
	roi.convertTo(roi, CV_8UC3, 255);
    return out;
}

/**
method for preprocessing the input RoI in order to obtain a better image for
for the advancedRegionGrowing method, we perform bilateralFiltering and we extract
edges with Canny
*/
void HandSegmentator::preprocessRoI(){
    /*

    // PHASE 1: SHARPENING for edge enhacement

    Mat laplacianKernel = (Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1);

    Mat imgLaplacian;
    filter2D(inputImg, imgLaplacian, CV_32F, laplacianKernel);
    Mat sharp;
    inputImg.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);*/

    // PHASE 2: BILATERAL FILTER for blurring for noise and minor details reduction but still preserving edges
    //bilateral smoothing for image enhancement
    Mat blurred;
    bilateralFilter(roi,blurred,10,50,120,BORDER_DEFAULT);
    roi = blurred;
    //imshow("Blurred", preprocessedImage);
    //waitKey();

   // PHASE 2: EDGE MAP extraction with Canny
    Canny(roi, edgeMap , 10, 150);

    //imshow("Roi", roi);
    //waitKey();
    //imshow("Canny", edgeMap);
    //waitKey();
    //imshow("Edge map", edgeMap);
    //waitKey;
    // PHASE 3: enhacement of edge map with opening for connecting edges

    /*
    //closing for connecting the edges
    int morph_size = 3;

    // Create structuring element
    Mat element = getStructuringElement(
        MORPH_RECT,
        Size(2 * morph_size + 1,
                2 * morph_size + 1),
        Point(morph_size, morph_size));

    // Closing

    morphologyEx(edgeMap, edgeMap,
                    MORPH_CLOSE, element,
                    Point(-1, -1), 2);
    imshow("Edge connected", edgeMap);
    waitKey();*/
}


/** regionGrowing based on kmeans clustering and edge map
 @param outputValue value of the output highlighted pixels (default value = 255)
 @return binary image with the region growing results
 */
Mat HandSegmentator::advancedRegionGrowing(unsigned char outputValue = 255) {

    Mat centers;
    Mat clust_img = kmeansSegmentationPositionQuantization(5, 2, 2);

    //imshow("Clust image", clust_img);
    //waitKey();

    //Analize the clustered colors near the center of the Hand Roi

    //smart color navigation: if the image is vertical, the hand is vertical so there will be
    //much chance of shadow areas, so we move more to the sides (left/right) than compared to above/below

    int x_center = clust_img.cols / 2;
    int y_center = clust_img.rows / 2;
    int r_x;
    int r_y;

    //default search range for squared images
    r_x = clust_img.cols / 30;
    r_y = clust_img.rows / 30;

    //search ranges for vertical or horizontal images
    double col_row_ratio = static_cast<double>(clust_img.cols) / clust_img.rows;
    double row_col_ratio = static_cast<double>(clust_img.rows) / clust_img.cols;

    if(col_row_ratio < 0.75){
        r_x = clust_img.cols / 30;
        r_y = clust_img.rows / 10;
    } else if (row_col_ratio < 0.75){
        r_x = clust_img.cols / 10;
        r_y = clust_img.rows / 30;
    }

    //cout << r_x << "\n";
    //cout << r_y << "\n";

    set<vector<unsigned char>> main_colors = set<vector<unsigned char>>(); //set of clustered colors near the hand RoI center
    vector<pair<int,int>> point_list;

    //construction of the set of clustered colors near the hand RoI center
    for(int i= y_center - r_y; i<y_center + r_y; i++){
        for(int j= x_center - r_x; j<x_center + r_x; j++){
            //Vector color construction
            vector<unsigned char> c = vector<unsigned char>(3);
            c[0] = clust_img.at<Vec3b>(i,j)[0];
            c[1] = clust_img.at<Vec3b>(i,j)[1];
            c[2] = clust_img.at<Vec3b>(i,j)[2];
            main_colors.insert(c);

            point_list.push_back(pair<int,int>(i, j));

        }
    }

    // boolean array/matrix of visited image pixels, same size as image
    // all the pixels are initialised to false
    Mat visited_matrix = Mat::zeros(roi.rows, roi.cols, CV_8U);

    while ( ! point_list.empty()) {
        // Get a point from the list
        pair<int, int> this_point = point_list.back();
        point_list.pop_back();

        int row = this_point.first;
        int col = this_point.second;

        // Visit the point
        visited_matrix.at<unsigned char>(row, col) = outputValue;

        // for each neighbour of this_point
        for (int i = row - 1; i <= row + 1; i++) {
            // vertical index is valid
            if (0 <= i && i < inputImg.rows) {
                for (int j = col - 1; j <= col + 1; j++) {
                    // hozirontal index is valid
                    if (0 <= j && j < inputImg.cols) {
                        unsigned char neighbour_visited = visited_matrix.at<unsigned char>(i, j);
                        vector<unsigned char> pixel_clustered_color = vector<unsigned char>(3);
                        pixel_clustered_color[0] = clust_img.at<Vec3b>(i,j)[0];
                        pixel_clustered_color[1] = clust_img.at<Vec3b>(i,j)[1];
                        pixel_clustered_color[2] = clust_img.at<Vec3b>(i,j)[2];

                        if (!neighbour_visited && main_colors.count(pixel_clustered_color)) { //pixel similarity
                            //pixel similar, we check if it is an edge to indicate that we should not go beyond it
                            if(edgeMap.at<unsigned char>(i, j) == 0)
                                point_list.push_back(pair<int, int>(i, j));
                        }
                    }
                }
            }
        }
    }
    //imshow("RG MASK", visited_matrix);
    //waitKey();
    return visited_matrix;
}

/**
@return Roi  hand mask useful for multiplehandSegmentationGrabCutMask
*/
Mat HandSegmentator::handMaskWithARG(){

    preprocessRoI();

    Mat result= advancedRegionGrowing();
    return result;
}

/** combines two binary mask and sets the GrabCut flags
 @param skin first binary mask to combine
 @param arg second mask to combine with the first
 */
void HandSegmentator::combineSkinAndARG(const Mat& skin, Mat& arg){
	int th = 40;
	if (skin.size() != arg.size()) {
		cout<<"Error: different sizes"<<endl;
	}
	for(int i = 0; i<arg.rows; i++){
		for(int j = 0; j<arg.cols; j++){
			if( skin.at<unsigned char>(i,j) == 255 && arg.at<unsigned char>(i,j) == 255)
				arg.at<unsigned char>(i,j) = GC_FGD;
			else{
				if (skin.at<unsigned char>(i,j) != arg.at<unsigned char>(i,j)) {
					arg.at<unsigned char>(i,j) = GC_PR_FGD;
				}
				else{
					arg.at<unsigned char>(i,j) = GC_PR_BGD;
				}
			}

			//Set to BGD very dark pixels
			if((int)roi.at<Vec3b>(i,j)[0]<th && (int)roi.at<Vec3b>(i,j)[1]<th && (int)roi.at<Vec3b>(i,j)[2]<th){
				arg.at<unsigned char>(i,j) = GC_BGD;
			}
		}
	}
}

/** main method for segmentation: it merge the skin detection mask and the mask returned by  advancedRegionGrowing to set the grabcut's flags and run GrabCut algorithm.
@return inputImg segmentation mask
*/
Mat HandSegmentator::multiplehandSegmentationGrabCutMask(){

    Mat out(inputImg.size(), inputImg.type(),Scalar(0,0,0));
	colorHands = inputImg.clone();
 	int iterations = 5;

 	//for each hand detected in the image apply roi segmentation then grabcut alg.
 	for(int i = 0; i<numberHands; i++){
		//mask of size equal to the original image inizialized with GC_BGD
 		Mat bwBig(inputImg.size(), CV_8UC1,Scalar(GC_BGD));
 		roi = inputImg(std::get<0>(rects[i]));
		Scalar color = std::get<1>(rects[i]);

		Mat skin = roi.clone();

		if (!isBlackAndWhite(inputImg)) {
			thresholdingYCrCb(skin);
		}
		else{
			//Segmentation skin on black and white image
			cvtColor(roi,skin, COLOR_BGRA2GRAY);
			threshold(skin,skin, 0, 255, THRESH_BINARY | THRESH_OTSU);
		}

 		//Segmentation on cropped image
 		Mat bwSmall(roi.size(),CV_8UC1, Scalar(0));
		bwSmall = handMaskWithARG();
		
		//if skin mask has low percentage of actived pixels use thresholdingWithSampledColor
		float thr = 3.5;
		if (computeQualityOfSkinMask(skin, 255)<thr) {
			skin = thresholdingWithSampledColor(roi);
			combineSkinAndARG(skin, bwSmall);
		}
		else{
			//Morphological erosion for creating smaller mask (more likely to be foregorund pixels)
			Mat element = getStructuringElement( MORPH_ELLIPSE, Size(3, 3) );
			int opIterations = 4;
			morphologyEx( bwSmall, bwSmall, MORPH_ERODE, element, Point(-1,-1), opIterations );

			//Compine two Segmentation results
			combineSkinAndARG(skin, bwSmall);
		}

 		//Superimpose smaller hand mask in a mask of size equal to the original image
		bwSmall.copyTo(bwBig(Rect(std::get<0>(rects[i]).tl().x,std::get<0>(rects[i]).tl().y,bwSmall.cols, bwSmall.rows)));

 		//applay GrabCut alg.
 		Mat bgd,fgd;
 		grabCut(inputImg,bwBig,Rect(),bgd,fgd,iterations,GC_INIT_WITH_MASK);
		
 		//Modify all pixels with GC_PR_FGD to GC_FGD for doing one compare
 		for (int i = 0; i<bwBig.rows; i++) {
 			for (int j = 0; j<bwBig.cols; j++) {
 				if(bwBig.at<unsigned char>(i,j) == GC_PR_FGD){
 					bwBig.at<unsigned char>(i,j) = GC_FGD;
 				}
 			}
 		}

 		compare(bwBig, GC_FGD, bwBig, CMP_EQ);			// CMP_EQ -> src1 is equal to src2

		//merge previous results with current hand segmentation 
		inputImg.copyTo(out,bwBig);

		//add color
		for(int i = 0; i < bwBig.rows; i++)
			for(int j = 0; j < bwBig.cols; j++)
				if(bwBig.at<unsigned char>(i,j) != 0){
					if(color[0] != 0)
						colorHands.at<Vec3b>(i,j)[0] = color[0]/2;
					if(color[1] != 0)
						colorHands.at<Vec3b>(i,j)[1] = color[1]/2;
					if(color[2] != 0)
						colorHands.at<Vec3b>(i,j)[2] = color[2]/2;
				}

		bwBig.release();
		bwSmall.release();
 	}

	destroyAllWindows();
	createBinaryMask(out);
    return out;
}

/** method that returns the colored hands image
@return colored hands image
*/
Mat HandSegmentator::getColoredHands(){
    return colorHands;
}

/**  Method that check if an 3 channels image is in black and white color
 @param img input image to test
 @return 1 if img is a black and white image 0 otherwise
 */
int HandSegmentator::isBlackAndWhite(Mat& img){
	int BW_img = 1;
	for(int i = 0; i < img.rows/3; i++){
		for(int j = 0; j < img.cols/3; j++){
			if (img.at<Vec3b>(i,j)[0] != img.at<Vec3b>(i,j)[1]) 
					BW_img = 0;
		}
	}
	return BW_img;
}

/** method that compute the percentage of the pixels that have v assigned as value in a image
 @param m input image 
 @param v value for which a pixels is cosidered active 
 @return percentage of active pixels
 */
float HandSegmentator::computeQualityOfSkinMask(Mat& m, int v){
	float activedPixels = 0;
	for(int i = 0; i < m.rows; i++){
		for(int j = 0; j < m.cols; j++){
			if (m.at<unsigned char>(i,j) == v) 
				activedPixels++;
		}
	}
	float p = (activedPixels/(static_cast<float>(m.rows*m.cols)))*100;
	return p;
}

/** Method that thresholds the input image based on the mean of the color of a random sample of pixels, in the sample are excluded pixels which has a color that is certainly a no-hand pixel.
 @param img input image 
 @return binary mask
 */
Mat HandSegmentator::thresholdingWithSampledColor(Mat& img){
	int lTH = 80, hTH = 400, T = 35,b,g,r;
	int iter = img.rows*img.cols*0.25;
	int meanB = 0, meanG = 0, meanR = 0;
	RNG rng = RNG(26414432);
	
	for (int i = 1; i<iter; i++) {
		//random generates a row index and a column index
		int randI = rng.uniform(0, img.rows-1), randJ = rng.uniform(0, img.cols-1);
		b = static_cast<int>(img.at<Vec3b>(randI,randJ)[0]);
		g = static_cast<int>(img.at<Vec3b>(randI,randJ)[1]);
		r = static_cast<int>(img.at<Vec3b>(randI,randJ)[2]);
		//only consider pixels that have the sum of the three colors between two thresholds
		if ((b+r+g>lTH) && (b+r+g<hTH)) {	
			meanB = meanB + b;
			meanG = meanG + g;
			meanR = meanR + r;
		}
		else
			i--;
	}
	//compute the mean color
	meanB = meanB/iter;
	meanG = meanG/iter;
	meanR = meanR/iter;
	
	//Thresholding
	Mat mask(img.size(),CV_8UC1, Scalar(0));
	for (int i = 0; i<img.rows; i++) {
		for (int j = 0; j<img.cols; j++) {
			if(abs(static_cast<int>(img.at<Vec3b>(i,j)[0]) - meanB) < T && abs(static_cast<int>(img.at<Vec3b>(i,j)[1]) - meanG) < T && abs(static_cast<int>(img.at<Vec3b>(i,j)[2]) - meanR) < T){
				mask.at<unsigned char>(i,j) = 255;
			}
		}
	}
	return mask;
}

/** For Simple thresholding on YCrCb and HSV plane based on skin color
@param img input image to threshold
*/
void HandSegmentator::thresholdingYCrCb(Mat& img){
    Mat hsv;
    cvtColor(img,hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(0, 15, 0), Scalar(17,130,255), hsv);


    Mat element = getStructuringElement( MORPH_RECT, Size(3, 3) );

    morphologyEx(hsv,hsv, MORPH_OPEN, element, Point(-1, -1), 2);

    //converting from gbr to YCbCr color space
    Mat ycrcb;
    cvtColor(img, ycrcb, COLOR_BGR2YCrCb);

    //skin color range for hsv color space
    inRange(ycrcb, Scalar(0, 135, 85), Scalar(255,170,135), ycrcb);

    morphologyEx(ycrcb,ycrcb, MORPH_OPEN, element, Point(-1, -1), 2);

    Mat finalMask;
    bitwise_and(ycrcb,hsv, finalMask);
    medianBlur(finalMask, finalMask, 3);

    morphologyEx(finalMask,finalMask, MORPH_OPEN, element, Point(-1,-1));

    img = finalMask;

}

/**
@param imgGC img to be converted in binary mask
*/
void HandSegmentator::createBinaryMask(Mat& imgGC){
	if (imgGC.channels() != 1)
		cvtColor(imgGC, imgGC, COLOR_BGR2GRAY);
    for(int i = 0; i < imgGC.rows; i++)
        for(int j = 0; j < imgGC.cols; j++)
            if(imgGC.at<unsigned char>(i,j) != 0)
                imgGC.at<unsigned char>(i,j) = 255;
}

//destructor for resources deallocation
HandSegmentator::~HandSegmentator(){
    roi.release();
    inputImg.release();
    edgeMap.release();
}
