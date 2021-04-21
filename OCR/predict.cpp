#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include "OCR.h"


using namespace std;
using namespace cv;



//����һ��28*28px��ͼƬ������ת��ΪMat(784,1,CV_32FC1)
Mat ProcessImg(string filename)
{
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    Mat res;
    img.convertTo(res, CV_32FC1);
    res = res.reshape(1, { 784,1 }) / 255;
    return res;
}


int main()
{

    //ocr();
    vector<Mat> biases = initBiases();
    readMat("b0", biases[0]);
    readMat("b1", biases[1]);

    vector<Mat> weights = initWeights();
    readMat("w0", weights[0]);
    readMat("w1", weights[1]);
    
    string filename = "d:/test.jpg";
    Mat img = ProcessImg(filename);
    Mat predictAry;
    predictAry = feedforward(biases, weights, img);
    int predict = argmax(predictAry);
    cout << "Ԥ��ֵ��    " << predict << endl;
}