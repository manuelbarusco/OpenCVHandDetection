#include "header.h"

using namespace cv;
using namespace std;

Mat kmeansSegmentationPositionMask (Mat  &img, int K, float weighX,float weightY){
	Mat labels, centers;
	
	// sharing
	Mat kernel = (Mat_<float>(3,3) <<
				  1,  1, 1,
				  1, -8, 1,
				  1,  1, 1); 
	
	Mat imgLaplacian;
	filter2D(img, imgLaplacian, CV_32F, kernel);
	Mat sharp;
	img.convertTo(sharp, CV_32F);
	Mat imgResult = sharp - imgLaplacian;

	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	imshow( "Sharped Image YPlane", imgResult );
	
	//Bilateral filter
	Mat outB;
	bilateralFilter(img,outB,10,20,100,BORDER_DEFAULT);
	imshow("Bilater Filter", outB);
	imgResult = outB.clone();
	
	if (img.type() == CV_8UC1) {
		cout<<"is CV_8UC1"<<endl;
	}
	
	//Conver to float for kmeans
	imgResult.convertTo(img, CV_32FC3, 1.0/255.0);
//	imshow("Converted image",img);
//	waitKey(0);

	Mat points(img.rows * img.cols, 5, CV_32FC1);
	 for( int y = 0; y < img.rows; y++ )
		 for( int x = 0; x < img.cols; x++ ){
			 for(int z = 0; z < 3; z++)
				 points.at<float>(y + x*img.rows, z) = img.at<Vec3f>(y,x)[z];
			 points.at<float>(y + x*img.rows, 3) = (float) y ;
			 points.at<float>(y + x*img.rows, 4) = (float) x;
		 }
	
	minMaxNormalization(points,weighX,weightY,true);
	
	int attempts = 10;
	kmeans(points, K, labels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 100000, 0.00001 ), attempts, KMEANS_PP_CENTERS, centers );

	cout<<"Channels of input: "<<img.channels()<<endl;
	cout<<"Size of center: "<<centers.size<<endl;
	cout<<"Size of labels: "<<labels.size<<endl;
	cout<<"Rows*Cols img = "<<img.cols*img.rows<<endl<<endl;
	//cout << "centers = " << endl << " " << centers << endl << endl;
	//cout << "points = " << endl << " " << points << endl << endl;
	
	Mat out( img.size(), CV_8UC1 );
	for( int y = 0; y < img.rows; y++ )
	  for( int x = 0; x < img.cols; x++ ){ 
		  if (labels.at<unsigned char>(y + x*img.rows,0) == 1) {
			  out.at<unsigned char>(y,x) = 255;
		  }
		  else{
			  out.at<unsigned char>(y,x) = 0;
		  }
	  }
	
	//Morphological closing 
	Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(3, 3) );
	int opIterations  = 1;
	morphologyEx( out, out, MORPH_OPEN, element, Point(-1,-1), opIterations ); 
	
	return out;
}

  
Mat kmeansSegmentationPositionQuantization(Mat  &img, int K, float weighX,float weightY){
	Mat labels, centers;
	
	// sharping
	Mat kernel = (Mat_<float>(3,3) <<
				  1,  1, 1,
				  1, -8, 1,
				  1,  1, 1); 
	
	Mat imgLaplacian;
	filter2D(img, imgLaplacian, CV_32F, kernel);
	Mat sharp;
	img.convertTo(sharp, CV_32F);
	Mat imgResult = sharp - imgLaplacian;

	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	//imshow( "Sharped Image", imgResult );
	
	//Bilateral filter
	Mat outB;
	bilateralFilter(img,outB,10,20,100,BORDER_DEFAULT);
	imshow("Bilater Filter", outB);
	imgResult = outB.clone();
	
	//Conver to float for kmeans
	imgResult.convertTo(img, CV_32FC3, 1.0/255.0);
//	imshow("Converted image",img);
//	waitKey(0);

	Mat points(img.rows * img.cols, 5, CV_32FC1);
	 for( int y = 0; y < img.rows; y++ )
		 for( int x = 0; x < img.cols; x++ ){
			 for(int z = 0; z < 3; z++)
				 points.at<float>(y + x*img.rows, z) = img.at<Vec3f>(y,x)[z];
			 points.at<float>(y + x*img.rows, 3) = (float) y ;
			 points.at<float>(y + x*img.rows, 4) = (float) x;
		 }
	
	minMaxNormalization(points,weighX,weightY,true);
	
	int attempts = 10;
	kmeans(points, K, labels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 100000, 0.00001 ), attempts, KMEANS_PP_CENTERS, centers );

//	cout<<"Channels of input: "<<img.channels()<<endl;
//	cout<<"Size of center: "<<centers.size<<endl;
//	cout<<"Size of labels: "<<labels.size<<endl;
//	cout<<"Rows*Cols img = "<<img.cols*img.rows<<endl<<endl;
//	cout << "centers = " << endl << " " << centers << endl << endl;
//	cout << "labels = " << endl << " " << labels << endl << endl;
//	cout << "points = " << endl << " " << points << endl << endl;
	

	Mat out( img.size(), CV_32FC3 );
	for( int y = 0; y < img.rows; y++ )
	  for( int x = 0; x < img.cols; x++ ){ 
		  int cluster_idx = labels.at<int>(y + x*img.rows,0);
			out.at<Vec3f>(y,x)[0] = centers.at<float>(cluster_idx, 0)*255;
			out.at<Vec3f>(y,x)[1] = centers.at<float>(cluster_idx, 1)*255;
			out.at<Vec3f>(y,x)[2] = centers.at<float>(cluster_idx, 2)*255;
	  }
	out.convertTo(out, CV_8UC3);
	return out;
}

//for weightX and weightY higher values lead to smaller value for position's features
//Color's features are weighted 1
void minMaxNormalization(Mat &img, float weightX, float weightY, bool treeChannels){
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

//Test thresholding on YCrCb plane
Mat thresholdingYCrCb(Mat &img){
	int Y_MIN  = 0;
	int Y_MAX  = 255;
	int Cr_MIN = 133;
	int Cr_MAX = 173;
	int Cb_MIN = 77;
	int Cb_MAX = 127;
	Mat mask;
	//first convert our RGB image to YCrCb
	cvtColor(img,mask,COLOR_BGR2YCrCb);
	inRange(mask,Scalar(Y_MIN,Cr_MIN,Cb_MIN),Scalar(Y_MAX,Cr_MAX,Cb_MAX),mask);
	return mask;
}


/*
 Mat labels, centers;
 // TermCriteria (int type, int maxCount, double epsilon) The accuracy is specified as criteria.epsilon. As soon as each of the cluster centers moves by less than criteria.epsilon on some iteration, the algorithm stops.
 
 // TODO: Blur the image to mitigate noise
 
 //Conver to float for kmeans
 Mat imgf(img.rows * img.cols, 3, CV_32F);
  for( int y = 0; y < img.rows; y++ )
	for( int x = 0; x < img.cols; x++ )
	  for( int z = 0; z < 3; z++)
		  imgf.at<float>(y + x*img.rows, z) = img.at<Vec3b>(y,x)[z];
 
 int attempts = 10;
 kmeans(imgf, K, labels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 100000, 0.00001), attempts, KMEANS_PP_CENTERS, centers );

 
 cout<<"Channels of input: "<<img.channels()<<endl;
 cout<<"Size of center: "<<centers.size<<endl;
 
 //centers = centers.reshape(3,centers.rows);
 //img = img.reshape(3,img.rows);
 
 //ptr<T>(int r) method to obtain a pointer to the beginning of row r
 
 Mat out( img.size(), img.type() );
 for( int y = 0; y < img.rows; y++ )
   for( int x = 0; x < img.cols; x++ ){ 
	 int cluster_idx = labels.at<int>(y + x*img.rows,0);
	   out.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
	   out.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
	   out.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
   }
 

 return out;*/
