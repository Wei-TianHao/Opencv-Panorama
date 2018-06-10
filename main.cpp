#include<cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include<algorithm>
#include<iostream>
#include "Panorma4461.hpp"
#include "hw6_pa.h"


int main() {
    std::vector<cv::Mat> img_vec;
    cv::Mat img_out;
    std::string data_set = "panorama-data1";
//    std::string data_set = "panorama-data2";
    for(int i = 1538; i < 1549; i++) {
//    for(int i = 1599; i < 1619; i++) {
        std::string file = data_set + "/DSC0"+std::to_string(i)+".JPG";
        img_vec.push_back(cv::imread(file));
    }
    Panorama4461 p;
    double f;
    FILE* K=fopen((data_set+"/K.txt").c_str(),"r");
    fscanf(K,"%lf",&f);
    p.makePanorama(img_vec, img_out, f);
    imshow("result",img_out);
    cv::imwrite(data_set+".png", img_out);
    cv::waitKey();
    
}
