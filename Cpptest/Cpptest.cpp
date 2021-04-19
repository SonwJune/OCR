// Cpptest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
    std::string imgfile = "d:/outputimg.jpg";
    
    Mat img = imread(imgfile, IMREAD_GRAYSCALE);
    Mat res = Mat(28, 28, CV_8UC1);
    resize(img, res, res.size(), 0, 0,INTER_AREA);

    std::cout << "" << std::endl;
}
