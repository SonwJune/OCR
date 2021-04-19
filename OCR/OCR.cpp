#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <random>
//#include "OCR.h"

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
double evaluate(vector<Mat> testImgMat, vector<Mat> testImgLbl, vector<Mat> weights, vector<Mat> biases);
int argmax(Mat m);
Mat feedforward(vector<Mat> biases, vector<Mat> weights, Mat img);
void writeMat(string filename, Mat img);
void readMat(string filename, Mat& img);

//根据训练集生成weights, biases, rate
int ocr()
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
        auto deltaNabla = backprop(miniBatch[0], weights, biases);
        for (size_t j = 0; j < nabla_b.size(); j++)
        {
            nabla_b[j] += deltaNabla.first[j];
        }
        for (size_t j = 0; j < nabla_w.size(); j++)
        {
            nabla_w[j] += deltaNabla.second[j];
        }
        double eta = 3.0;
        for (size_t j = 0; j < weights.size(); j++)
        {
            Mat test20 = weights[j] - (eta / 10) * nabla_w[j];
            weights[j] = weights[j] - (eta / 10) * nabla_w[j];
        }
        for (size_t j = 0; j < biases.size(); j++)
        {
            biases[j] = biases[j] - (eta / 10) * nabla_b[j];
        }
        cout << "第" << i << "组" << "    " << "共" << miniBatches.size()/10 << "组" << endl;
    }
    //加载测试集数据
    vector<Mat> testImgMat = GetTrainImg("d:/deeplearning/t10k-images.idx3-ubyte");
    vector<Mat> testImgLbl = GetTrainLbl("d:/deeplearning/t10k-labels.idx1-ubyte");
    double rate = evaluate(testImgMat, testImgLbl, weights, biases);
    cout << "rate:    " << rate*100 << "%"<<endl;
    //GetTrainLbl("d:/deeplearning/train-labels.idx1-ubyte");
    // 
    // 
    // 
    //存储w  b
    writeMat("b0", biases[0]);
    writeMat("b1", biases[1]);
    writeMat("w0", weights[0]);
    writeMat("w1", weights[1]);

    return 0;
}

double evaluate(vector<Mat> testImgMat, vector<Mat> testImgLbl, vector<Mat> weights, vector<Mat> biases)
{
    vector<pair<int, int>> testResults;
    for (size_t i = 0; i < testImgMat.size(); i++)
    {
        Mat res = feedforward(biases, weights, testImgMat[i]);
        int predict = argmax(res);
        int lable = argmax(testImgLbl[i]);
        testResults.push_back({predict,lable});
    }
    int sum = 0;//预测成功的个数
    for (auto& i : testResults)
    {
        if (i.first==i.second)
        {
            sum += 1;
        }
    }
    double rate = (double)sum / testResults.size();
    return rate;
}

int argmax(Mat m)
{
    //返回Mat中最大元素值的下标
    Point maxIndex;
    minMaxLoc(m, 0, 0, 0, &maxIndex);
    return maxIndex.y;
}

Mat feedforward(vector<Mat> biases, vector<Mat> weights,Mat img)
{
    for (size_t i = 0; i < biases.size(); i++)
    {
        img = sigmoid(weights[i] * img + biases[i]);
    }
    return img;
}


pair<vector<Mat>,vector<Mat>> backprop(pair<Mat, Mat> img, vector<Mat> weights, vector<Mat> biases)
{
    //pair<Mat, Mat> :<img, label>
    Mat test = img.first.clone().reshape(1, { 28,28 });//test
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
    Mat test10 = costDerivative(activations.back(), y);
    Mat test11 = sigmoidPrime(zs.back());
    Mat test13 = test10.mul(test11);
    Mat delta = costDerivative(activations.back(), y).mul(sigmoidPrime(zs.back()));
    //nabla_b.pop_back();
    //nabla_b.push_back(delta);
    nabla_b[nabla_b.size() - 1] = delta;
    //nabla_w.pop_back();
    //nabla_w.push_back(delta * activations[activations.size() - 2].t());
    nabla_w[nabla_w.size() - 1] = delta * activations[activations.size() - 2].t();
    
    for (size_t i = 2; i < 3; i++)
    {
        Mat z = zs[zs.size() - i];
        Mat sp = sigmoidPrime(z);
        Mat tmp = weights[weights.size() - i+ 1].t() * delta;
        delta = tmp.mul(sp);
        nabla_b[nabla_b.size() - i] = delta;
        nabla_w[nabla_w.size() - i] = delta * activations[activations.size() - i - 1].t();
    }
    return { nabla_b,nabla_w };
    //反向传播算法返回的两个值错误
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
    randu(biases0, Scalar::all(-1), Scalar::all(1));
    randu(biases1, Scalar::all(-1), Scalar::all(1));
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
    randu(weights0, Scalar::all(-1), Scalar::all(1));
    randu(weights1, Scalar::all(-1), Scalar::all(1));
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
    //读取文件头部8字节
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

//将CV_32FC1的Mat以二进制方式写入文件
void writeMat(string filename,Mat img)
{
    ofstream ofs(filename, ios::binary | ios::out);
    if (!ofs)
    {
        cerr << "读写文件错误" << endl;
    }
    MatIterator_<float> it, end;
    for (it = img.begin<float>(), end = img.end<float>(); it != end; ++it)
    {
        float fPixel = *it;
        ofs.write(reinterpret_cast<char*>(&fPixel), sizeof fPixel);
    }
    ofs.close();
}

//以二进制方式读入文件并转化为CV_32FC1的Mat
void readMat(string filename, Mat& img)
{
    ifstream ifs(filename, ios::binary | ios::in);
    vector<float> fvec;
    
    while (!ifs.eof())
    {
        float b;
        ifs.read(reinterpret_cast<char*>(&b), sizeof(b));
        fvec.push_back(b);
    }
    ifs.close();
    //ifs.eof多运行一次;
    fvec.pop_back();
    int i = 0;
    MatIterator_<float> it, end;
    for (it = img.begin<float>(), end = img.end<float>(); it != end; ++it)
    {
        *it = fvec[i];
        ++i;
    }
}