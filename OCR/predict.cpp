#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include "OCR.h"


using namespace std;
using namespace cv;


//��biases.bin�ж�ȡ����
void getBiases(string filename,vector<Mat>& Biases)
{
    ifstream inBiases("biases.bin", ios::binary);
    if (!inBiases)
    {
        cerr << "�����ļ�����" << endl;
    }
    inBiases.read(reinterpret_cast<char*>(Biases.data()), Biases.size() * sizeof(Biases.front()));
    inBiases.close();
}

//��weights.bin�ж�ȡ����
vector<Mat> getWeights(string filename)
{
    vector<Mat> Weights = initWeights();
    ifstream inWeights("weights.bin", ios::binary);
    if (!inWeights)
    {
        cerr << "�����ļ�����" << endl;
    }
    inWeights.read(reinterpret_cast<char*>(Weights.data()), Weights.size() * sizeof(Weights.front()));
    inWeights.close();
    return Weights;
}


//����һ��28*28px��ͼƬ������ת��ΪMat(784,1,CV_32FC1)
Mat ProcessImg(string filename)
{
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    Mat res;
    img.convertTo(res, CV_32FC1);
    res = res.reshape(1, { 784,1 }) / 255;
    return res;
}

int main00()
{
    string filename = "d:/2.jpg";
    vector<Mat> biases = initBiases();
    getBiases("biases.bin",biases);
    vector<Mat> weights = getWeights("weights.bin");
    
    Mat img = ProcessImg(filename);
    Mat predictAry;
    predictAry = feedforward(biases, weights, img);
    int predict = argmax(predictAry);
    cout << "Ԥ��ֵ��    " << predict << endl;


    return 0;
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