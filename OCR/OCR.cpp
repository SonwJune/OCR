#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "OCR.h"
#include <random>

using namespace cv;
using namespace std;


vector<Mat> GetTrainLbl(string filename);
vector<Mat> GetTrainImg(string filename);
vector<Mat> initBiases();
vector<Mat> initNabla_b();
vector<Mat> initNabla_w();
vector<Mat> initWeights();
Mat sigmoid(Mat z);
pair<vector<Mat>, vector<Mat>> backprop(pair<Mat, Mat> img, vector<Mat> weights, vector<Mat> biases);
Mat costDerivative(Mat outputActivations, Mat imgLbl);
Mat sigmoidPrime(Mat z);


int main(int argc, char** argv)
{
    vector<Mat> weights = initWeights();
    vector<Mat> biases = initBiases();
    vector<Mat> imgMat = GetTrainImg("d:/deeplearning/train-images.idx3-ubyte");
    vector<Mat> imgLbl = GetTrainLbl("d:/deeplearning/train-labels.idx1-ubyte");

    vector<pair<Mat, Mat>> miniBatches;
    for (size_t i = 0; i !=imgMat.size(); i++)
    {
        miniBatches.push_back({imgMat[i],imgLbl[i]});
    }
    shuffle(miniBatches.begin(), miniBatches.end(), std::default_random_engine(33466));
    for (size_t i = 0; i < miniBatches.size()/10; i++)
    {
        auto iter = miniBatches.begin();
        vector<pair<Mat, Mat>> miniBatch(iter + 10 * i, iter + 10 * (i + 1));
        vector<Mat> nabla_b = initNabla_b();
        vector<Mat> nabla_w = initNabla_w();
        auto c = backprop(miniBatch[0], weights, biases);

    }
    
    

    //GetTrainLbl("d:/deeplearning/train-labels.idx1-ubyte");
}

pair<vector<Mat>,vector<Mat>> backprop(pair<Mat, Mat> img, vector<Mat> weights, vector<Mat> biases)
{
    //pair<Mat, Mat> :<img, label>
    Mat x = img.first;
    Mat y = img.second;
    vector<Mat> nabla_b = initNabla_b();
    vector<Mat> nabla_w = initNabla_w();
    //前向传播
    Mat activation = x;
    vector<Mat> activations({x});   //vector to store all the activations, layer by layer
    vector<Mat> zs;
    for (size_t i = 0; i != weights.size(); i++)
    {
        Mat z = weights[i] * activation + biases[i];
        zs.push_back(z);
        activation = sigmoid(z);
        activations.push_back(activation);
    }
    //反向传播

    Mat delta = costDerivative(activations.back(), y).mul(sigmoidPrime(zs.back()));
    nabla_b.pop_back();
    nabla_b.push_back(delta);
    nabla_w.pop_back();
    nabla_w.push_back(delta * activations[activations.size() - 2].t());
    
    for (size_t i = 2; i < 3; i++)
    {
        Mat z = zs[zs.size() - i];
        Mat sp = sigmoidPrime(z);
        auto a = activations.size() - i - 1 + 1;
        Mat tmp = weights[weights.size() - i+ 1].t() * delta;
        delta = tmp.mul(sp);
        nabla_b.pop_back();
        nabla_b.push_back(delta);
        nabla_w.pop_back();
        nabla_w.push_back(delta * activations[activations.size() - i- 1].t());
    }
    return { nabla_b,nabla_w };
}

Mat costDerivative(Mat outputActivations, Mat imgLbl)
{
    return outputActivations - imgLbl;
}

Mat sigmoid(Mat z)
{
    //The sigmoid function.
    Mat res;
    exp(-z, res);
    return 1.0 / (1.0 + res);
}

Mat sigmoidPrime(Mat z)
{
    //Derivative of the sigmoid function.
    return sigmoid(z).mul((1 - sigmoid(z)));
}

vector<Mat> initNabla_w()
{
    vector<Mat> nabla_w;
    Mat nabla_w0 = Mat::zeros(30, 784, CV_32FC1);
    Mat nabla_w1 = Mat::zeros(10, 30, CV_32FC1);
    nabla_w.push_back(nabla_w0.clone());
    nabla_w.push_back(nabla_w1.clone());
    return nabla_w;
}

vector<Mat> initNabla_b()
{
    vector<Mat> nabla_b;
    Mat nabla_b0 = Mat::zeros(30, 1, CV_32FC1);
    Mat nabla_b1 = Mat::zeros(10, 1, CV_32FC1);
    nabla_b.push_back(nabla_b0.clone());
    nabla_b.push_back(nabla_b1.clone());
    return nabla_b;
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
        Mat tmp1;
        tmp.convertTo(tmp1, CV_32FC1);
        tmp1 = tmp1.reshape(1, { 784,1 }) / 255;
        imgMat.push_back(tmp1.clone());
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
