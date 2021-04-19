#ifndef OCR_H
#define OCR_H

#include <vector>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

vector<Mat> initBiases();
vector<Mat> initWeights();
Mat feedforward(vector<Mat> biases, vector<Mat> weights, Mat img);
int argmax(Mat m);
int ocr();
void readMat(string filename, Mat& img);

#endif // !OCR_H



