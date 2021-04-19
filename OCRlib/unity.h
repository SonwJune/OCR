#ifndef OCRLIB_UNITY_H
#define OCRLIB_UNITY_H

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

#endif // !OCRLIB_UNITY_H



#pragma once
