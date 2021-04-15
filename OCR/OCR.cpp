#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "OCR.h"

using namespace cv;
using namespace std;


vector<Mat> GetTrainLbl(string filename);

vector<Mat> initBiases();
vector<Mat> initWeights();

int main(int argc, char** argv)
{
    /*vector<Mat> weights = initWeights();
    vector<Mat> biases = initBiases();
    vector<Mat> imgMat = GetTrainImg("d:/deeplearning/train-images.idx3-ubyte");*/
    vector<Mat> imgLbl = GetTrainLbl("d:/deeplearning/train-labels.idx1-ubyte");

    
    //GetTrainLbl("d:/deeplearning/train-labels.idx1-ubyte");
}

vector<Mat> initBiases()
{
    //初始化biases
    vector<Mat> biases;
    Mat biases0 = Mat(30, 1, CV_32FC1);
    Mat biases1 = Mat(10, 1, CV_32FC1);
    randu(biases0, Scalar::all(0), Scalar::all(1));
    randu(biases1, Scalar::all(0), Scalar::all(1));
    biases.push_back(biases0.clone());
    biases.push_back(biases1.clone());
    return biases;
}

vector<Mat> initWeights()
{
    //初始化biases
    vector<Mat> weights;
    Mat weights0 = Mat(30, 784, CV_32FC1);
    Mat weights1 = Mat(10, 30, CV_32FC1);
    randu(weights0, Scalar::all(0), Scalar::all(1));
    randu(weights1, Scalar::all(0), Scalar::all(1));
    weights.push_back(weights0.clone());
    weights.push_back(weights1.clone());
    return weights;
}

vector<Mat> GetTrainImg(string filename)
{
    //"d:/deeplearning/train-images.idx3-ubyte"
    ifstream inTrainImage(filename, ios::binary);
    if (!inTrainImage)
    {
        cerr << "读写文件错误" << endl;
    }
    char buffer[784];
    vector<Mat> imgMat;
    //读取文件头部16字节
    inTrainImage.seekg(16, ios::cur);
    while (inTrainImage.read(buffer, 784))
    {
        Mat tmp = Mat(28, 28, CV_8UC1, buffer);
        imgMat.push_back(tmp.clone());
    }
    inTrainImage.close();
    return imgMat;
}

vector<Mat> GetTrainLbl(string filename)
{
    //"d:/deeplearning/train-labels.idx1-ubyte"
    ifstream inTrainImage(filename, ios::binary);
    if (!inTrainImage)
    {
        cerr << "读写文件错误" << endl;
    }
    char buffer[1];
    vector<Mat> imgLbl;
    //读取文件头部16字节
    inTrainImage.seekg(8, ios::cur);
    while (inTrainImage.read(buffer, 1))
    {
        int tmp = *buffer;
        Mat m = Mat::zeros(10, 1, CV_32FC1);
        m.row(tmp) = 1.f;
        imgLbl.push_back(m.clone());
    }
    inTrainImage.close();
    return imgLbl;
}
