#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include "unity.h"

using namespace std;
using namespace cv;



//接受一张28*28px的图片，将其转换为Mat(784,1,CV_32FC1)
Mat ProcessImg(string filename)
{
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    Mat res;
    img.convertTo(res, CV_32FC1);
    res = res.reshape(1, { 784,1 }) / 255;
    return res;
}


int PredictImgValue(char* b0, char* b1, char* w0, char* w1, char* imgfile)
{
    //ocr();
    vector<Mat> biases = initBiases();
    readMat(b0, biases[0]);
    readMat(b1, biases[1]);

    vector<Mat> weights = initWeights();
    readMat(w0, weights[0]);
    readMat(w1, weights[1]);

    string filename = imgfile;
    Mat img = ProcessImg(filename);
    Mat predictAry;
    predictAry = feedforward(biases, weights, img);
    int predict = argmax(predictAry);
    //cout << "预测值：    " << predict << endl;
    return predict;
}

//此函数用于DLL测试
int OCRadd(int a, int b)
{
    return a + b;
}

void ResizeImg(char* filename)
{
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    Mat res = Mat(28, 28, CV_8UC1);
    resize(img, res, res.size(), 0, 0, INTER_AREA);
    imwrite(filename, res);
}