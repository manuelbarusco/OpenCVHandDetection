//HandSegmentator.cpp

//Authors: Manuel Barusco, Simone Gregori


#include "HandSegmentator.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <set>

using namespace std;

/** constructor
 @param roi Mat object with the region of interest with the hand that must be segmented or the original image
 @param nHands Number of hand inputRoi
 @param r Vector of Rect of size nHand
 */
HandSegmentator::HandSegmentator(const cv::Mat& roi, const int nHands, const vector<cv::Rect> r){
    inputRoi = roi.clone();
    numberHands = nHands;
    rects = r;
}

//For Simple thresholding on YCrCb plane based on skin color
cv::Mat HandSegmentator::thresholdingYCrCb(){
    int Y_MIN  = 0;
    int Y_MAX  = 255;
    int Cr_MIN = 133;
    int Cr_MAX = 173;
    int Cb_MIN = 77;
    int Cb_MAX = 127;
    cv::Mat mask;
    //first convert our RGB image to YCrCb
    cv::cvtColor(inputRoi,mask,cv::COLOR_BGR2YCrCb);
    cv::inRange(mask,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),mask);
    return mask;
}

//For KMeans based on pixel Position and Color
//for weightX and weightY higher values lead to smaller value for position's features
//Color's features are weighted 1
void HandSegmentator::minMaxNormalization(cv::Mat &img, float weightX, float weightY, bool treeChannels){
    int dim = 3;
    if (treeChannels)
        dim = 5;
    int weights[dim];
    if (treeChannels){
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


/** kmeans, performs kmeans based only on color informations
 @param k number of clusters
 @param att number of attempts
 @param centers input output Mat object for the kmeans centers
 */
cv::Mat HandSegmentator::kmeans(int k, int att, cv::Mat &centers){

    //clone of the input img
    cv::Mat src = inputRoi.clone();

    //performs kmeans in the color image
    cv::Mat samples(src.rows * src.cols, 3, CV_32F);
    for( int y = 0; y < src.rows; y++ )
        for( int x = 0; x < src.cols; x++ )
            for( int z = 0; z < 3; z++)
                samples.at<float>(y + x*src.rows, z) = src.at<cv::Vec3b>(y,x)[z];

    cv::Mat labels;
    cv::kmeans(samples, k, labels,cv::TermCriteria(cv::TermCriteria::EPS+ cv::TermCriteria::COUNT, 10, 1.0), att, cv::KMEANS_PP_CENTERS, centers);

    //clustered image creation
    cv::Mat clust_img( src.size(), src.type() );
    for( int y = 0; y < src.rows; y++ )
        for( int x = 0; x < src.cols; x++ ) {
            int cluster_idx = labels.at<int>(y + x*src.rows,0);
            clust_img.at<cv::Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
            clust_img.at<cv::Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
            clust_img.at<cv::Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
        }

    return clust_img;
}

cv::Mat HandSegmentator::kmeansSegmentationPositionQuantization(int K, float weighX,float weightY){
    cv::Mat labels, centers;

    cv::Mat imgResult = preprocessedImage.clone();

    //Conver to float for kmeans
    imgResult.convertTo(inputRoi, CV_32FC3, 1.0/255.0);

    cv::Mat points(inputRoi.rows * inputRoi.cols, 5, CV_32FC1);
     for( int y = 0; y < inputRoi.rows; y++ )
         for( int x = 0; x < inputRoi.cols; x++ ){
             for(int z = 0; z < 3; z++)
                 points.at<float>(y + x*inputRoi.rows, z) = inputRoi.at<cv::Vec3f>(y,x)[z];
                 points.at<float>(y + x*inputRoi.rows, 3) = (float) y ;
                 points.at<float>(y + x*inputRoi.rows, 4) = (float) x;
         }

    minMaxNormalization(points,weighX,weightY,true);

    int attempts = 10;
    cv::kmeans(points, K, labels, cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 100000, 0.00001 ), attempts, cv::KMEANS_PP_CENTERS, centers );

//    cout<<"Channels of input: "<<img.channels()<<endl;
//    cout<<"Size of center: "<<centers.size<<endl;
//    cout<<"Size of labels: "<<labels.size<<endl;
//    cout<<"Rows*Cols img = "<<img.cols*img.rows<<endl<<endl;
//    cout << "centers = " << endl << " " << centers << endl << endl;
//    cout << "labels = " << endl << " " << labels << endl << endl;
//    cout << "points = " << endl << " " << points << endl << endl;


    cv::Mat out( inputRoi.size(), CV_32FC3 );
    for( int y = 0; y < inputRoi.rows; y++ )
      for( int x = 0; x < inputRoi.cols; x++ ){
          int cluster_idx = labels.at<int>(y + x*inputRoi.rows,0);
            out.at<cv::Vec3f>(y,x)[0] = centers.at<float>(cluster_idx, 0)*255;
            out.at<cv::Vec3f>(y,x)[1] = centers.at<float>(cluster_idx, 1)*255;
            out.at<cv::Vec3f>(y,x)[2] = centers.at<float>(cluster_idx, 2)*255;
      }
    out.convertTo(out, CV_8UC3);
    return out;
}

void HandSegmentator::preprocessImage(){
    /*

    // PHASE 1: SHARPENING for edge enhacement

    Mat laplacianKernel = (Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1);

    Mat imgLaplacian;
    filter2D(inputRoi, imgLaplacian, CV_32F, laplacianKernel);
    Mat sharp;
    inputRoi.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);*/

    // PHASE 2: BILATERAL FILTER for blurring for noise and minor details reduction but still preserving edges

    //bilateral smoothing for image enhancement
    cv::bilateralFilter(inputRoi,preprocessedImage,10,50,120,cv::BORDER_DEFAULT);
    cv::imshow("Blurred", preprocessedImage);
    cv::waitKey();

    // PHASE 2: EDGE MAP extraction with Canny

    cv::Canny(preprocessedImage, edgeMap , 30, 220); //1:3 proportion
    cv::imshow("Edge", edgeMap);
    cv::waitKey();

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


/** regionGrowing
 @param seedSet vector of seed initial points
 @param outputValue value of the output highlighted pixels
 @param tolerance tolerance for intensity region growing
 */
cv::Mat HandSegmentator::regionGrowing(const vector<pair<int, int>>& seedSet, unsigned char outputValue = 255, float tolerance =5) {

    cv::Mat grayscaleROI;

    cv::cvtColor(preprocessedImage, grayscaleROI, cv::COLOR_BGR2GRAY);

    cv::imshow("Preprocessed Image bf rg", preprocessedImage);
    cv::waitKey();

    cv::Mat centers;
    cv::Mat clust_img = kmeansSegmentationPositionQuantization(5, 2, 2);

    imshow("Clust image", clust_img);
    cv::waitKey();

    /* PARTE PER ANALISI COLORE VICINO AL CENTRO */

    //smart color navigation: se l'immagine è verticale, la mano è in verticale e quindi ci sarà
    //molta possibilità di avere delle zone d'ombra, quindi ci spostiamo di più ai lati che
    //rispetto a sopra/sotto

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

    if(col_row_ratio < 0.40){
        r_x = clust_img.cols / 5;
        r_y = clust_img.rows / 10;
    } else if (row_col_ratio < 0.40){
        r_x = clust_img.cols / 10;
        r_y = clust_img.rows / 5;
    }

    set<vector<unsigned char>> main_colors = set<vector<unsigned char>>(); //set of clustered colors near the window center

    //analyze color clusters informations near the center
    for(int i= y_center - r_y; i<y_center + r_y; i++){
        for(int j= x_center - r_x; j<x_center + r_x; j++){
            //Vector color construction
            vector<unsigned char> c = vector<unsigned char>(3);
            c[0] = clust_img.at<cv::Vec3b>(i,j)[0];
            c[1] = clust_img.at<cv::Vec3b>(i,j)[1];
            c[2] = clust_img.at<cv::Vec3b>(i,j)[2];
            //cout << (int) c[0] << (int) c[1] << (int) c[2];
            main_colors.insert(c);
        }
    }

    //DA USARE NEL CASO NON VADA CON INSIEME DI COLORI
    //Vec3b roiCenterCluster = clust_img.at<Vec3b>(inputRoi.rows/2, inputRoi.cols/2);

    // boolean array/matrix of visited image pixels, same size as image
    // all the pixels are initialised to false
    cv::Mat visited_matrix = cv::Mat::zeros(inputRoi.rows, inputRoi.cols, CV_8U);

    // List of points to visit
    vector<pair<int, int>> point_list = seedSet;

    while ( ! point_list.empty()) {
        // Get a point from the list
        pair<int, int> this_point = point_list.back();
        point_list.pop_back();

        int row = this_point.first;
        int col = this_point.second;

        // Visit the point
        visited_matrix.at<unsigned char>(row, col) = outputValue;

        // for each neighbour of this_point
        for (int i = row - 1; i <= row + 1; i++)
        {
            // vertical index is valid
            if (0 <= i && i < inputRoi.rows)
            {
                for (int j = col - 1; j <= col + 1; j++)
                {
                    // hozirontal index is valid
                    if (0 <= j && j < inputRoi.cols)
                    {
                        unsigned char neighbour_visited = visited_matrix.at<unsigned char>(i, j);
                        vector<unsigned char> pixel_clustered_color = vector<unsigned char>(3);
                        pixel_clustered_color[0] = clust_img.at<cv::Vec3b>(i,j)[0];
                        pixel_clustered_color[1] = clust_img.at<cv::Vec3b>(i,j)[1];
                        pixel_clustered_color[2] = clust_img.at<cv::Vec3b>(i,j)[2];

                        if (!neighbour_visited && main_colors.count(pixel_clustered_color)) { //pixel similarity
                            //pixel simile, controlliamo se è un edge per indicare che non dobbiamo andare oltre a lui
                            if(edgeMap.at<unsigned char>(i, j) == 0)
                                point_list.push_back(pair<int, int>(i, j));
                        }
                    }
                }
            }
        }
    }

    return visited_matrix;
}

/** setGrabCutFlag
 @param maskPR binary mask where 255 represent probably foreground
 @param mask binary mask where 255 represent for sure foreground pixels
 @param flagDefault Value to assing to the new mask where pixels are 0 in maskPR
 @param flagTrue  Value to assing to the new mask where pixels are 255 in mask
 @param flagPR_True  Value to assing to the new mask where pixels are 255 in mask PR and 0 in mask
 */
cv::Mat HandSegmentator::setGrabCutFlag(cv::Mat maskPR, cv::Mat mask, int flagDefault, int flagTrue, int flagPR_True){
	if (maskPR.size() != mask.size()) {
		cout<<"Error: different sizes"<<endl;
	}
	Mat out(maskPR.size(), CV_8U, Scalar(flagDefault));
	for(int i = 0; i<maskPR.rows; i++){
		for(int j = 0; j<maskPR.cols; j++){
			if( mask.at<unsigned char>(i,j) == 255)
				out.at<unsigned char>(i,j) = flagTrue;
			else{
				if (maskPR.at<unsigned char>(i,j) == 255 && mask.at<unsigned char>(i,j) == 0) {
					out.at<unsigned char>(i,j) = flagPR_True;
				}
			}
		}
	}
	return out;
}

cv::Mat HandSegmentator::MiltiplehandSegmentationGrabCutMask(){
	if (!isFullimgSet){
		fullImg = inputRoi.clone();
		isFullimgSet = 1;
	}

	Mat out(fullImg.size(), fullImg.type(),Scalar(0,0,0));
	vector<Mat> croppedMasks;
	int iterations = 5;

	cout<<"Number of hand on this image: "<<numberHands<<endl;

	//Single hand segmentation
	//Create vector of images cropped in ROI
	for(int i = 0; i<numberHands; i++){
		Mat bwBig(fullImg.size(), CV_8UC1,Scalar(GC_BGD));
		//Crop the image using rectangle
		Mat handCropped;
		handCropped = fullImg(rects[i]);
		imshow("croppedImg", handCropped);
		waitKey(0);

		//Segmentation on cropped image
		Mat bwSmall(handCropped.size(),CV_8UC1, Scalar(0));
		inputRoi = handCropped.clone();
		bwSmall = handSegmentation();

		//Morphological dilation for creating larger mask for specifing PR_FGD pixels
		Mat bwS_PR_FGD;
		Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(3, 3) );
		int opIterations  = 4;
		morphologyEx( bwSmall, bwS_PR_FGD, MORPH_DILATE, element, Point(-1,-1), opIterations );
		imshow("Binary Image after dilation", bwS_PR_FGD);
		waitKey();

		Mat bwCombined = setGrabCutFlag(bwS_PR_FGD, bwSmall, GC_PR_BGD, GC_FGD, GC_PR_FGD);
//		imshow("Binary Image combined", bwCombined);
//		waitKey();

		//Superimpose smaller hand mask in a mask of size equal to the original image
		bwCombined.copyTo(bwBig(cv::Rect(rects[i].tl().x,rects[i].tl().y,bwCombined.cols, bwCombined.rows)));
//		imshow("Binary Image combined full size", bwBig);
//		waitKey();

		//applay GrabCut alg.
		Mat bgd,fgd;
		grabCut(fullImg,bwBig,Rect(),bgd,fgd,iterations,GC_INIT_WITH_MASK);
		//Modify all pixels with GC_PR_FGD to GC_FGD for doing one compare
		for (int i = 0; i<bwBig.rows; i++) {
			for (int j = 0; j<bwBig.cols; j++) {
				if(bwBig.at<unsigned char>(i,j) == GC_PR_FGD){
					bwBig.at<unsigned char>(i,j) = GC_FGD;
				}
			}
		}
		compare(bwBig, GC_FGD, bwBig, CMP_EQ);			// CMP_EQ -> src1 is equal to src2
		fullImg.copyTo(out,bwBig);
//		string t = "Temp out of hand number " + std::to_string(i);
//		imshow(t, out);
//		waitKey();
		destroyAllWindows();
	}

	return out;
}

cv::Mat HandSegmentator::MiltiplehandSegmentationGrabCutRect(){
    cv::Mat out(inputRoi.size(), inputRoi.type(),cv::Scalar(0,0,0));
    int iterations = 5;
	if (!isFullimgSet){
		fullImg = inputRoi.clone();
		isFullimgSet = 1;
	}

    for (int i = 0; i < numberHands; i++){
        cv::Mat bgd, fgd, hand;

        grabCut(fullImg,hand,rects[i],bgd,fgd,iterations,GC_INIT_WITH_RECT);

		//Modify all pixels with GC_PR_FGD to GC_FGD for doing one compare
		for (int i = 0; i<hand.rows; i++) {
			for (int j = 0; j<hand.cols; j++) {
				if(hand.at<unsigned char>(i,j) == GC_PR_FGD){
					hand.at<unsigned char>(i,j) = GC_FGD;
				}
			}
		}
        compare(hand, GC_FGD, hand, CMP_EQ);            // CMP_EQ -> src1 is equal to src2. GC_PR_FGD -> Likely a foreground pixel
		fullImg.copyTo(out,hand);
//        string t = "Temp out (Rect) of hand number " + std::to_string(i);
//        imshow(t, out);
//        waitKey();
        destroyAllWindows();
      }

    return out;
}

cv::Mat HandSegmentator::handSegmentation(){

    preprocessImage();

    // Create binary image from source image, using otsu, only for comparison
    cv::Mat bw;
    cvtColor(preprocessedImage, bw, cv::COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::imshow("Binary Image", bw);
    cv::waitKey();

    //add only the center pixel as seed point
    vector<pair<int,int>> seedSet;
    seedSet.push_back(pair<int,int>(inputRoi.rows/2, inputRoi.cols/2));


    cv::Mat result= regionGrowing(seedSet);
    cv::imshow("Result", result);
    cv::waitKey();
    return result;
}
