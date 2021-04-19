#pragma once
extern "C" __declspec(dllexport) int PredictImgValue(char* b0, char* b1, char* w0, char* w1, char* imgfile);
extern "C" __declspec(dllexport) int OCRadd(int a, int b);
extern "C" __declspec(dllexport) void ResizeImg(char* filename)