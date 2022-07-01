#include "HandSegmentator.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
	
	Mat inputRoi= imread("5.png");
	
	HandSegmentator hs = HandSegmentator(inputRoi);
	
	Mat result = hs.handSegmentation();
	
	imshow("Result", result);
	waitKey();
	return 0;
			
}
