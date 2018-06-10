#ifndef PANORAMA_4461
#define PANORAMA_4461

#include<cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include<algorithm>
#include<iostream>
#include "hw6_pa.h"


class Panorama4461: public CylindricalPanorama {
public:
    cv::Mat cylinder(cv::Mat img, int f)
    {
        int cy_cols, cy_rows;
        cy_cols = 2 * f*atan(0.5*img.cols / f);//柱面图像宽
        cy_rows = 0.5*img.rows*f / sqrt(pow(f, 2)) + 0.5*img.rows;//柱面图像高
        
        cv::Mat img_out = cv::Mat::zeros(cy_rows, cy_cols, CV_8UC3);
        cv::Mat_<cv::Vec3b> img1(img);
        cv::Mat_<cv::Vec3b> img2(img_out);

        int x1(0), y1(0);
        for (int i = 0; i < img.rows; i++)
            for (int j = 0; j < img.cols; j++)
            {
                x1 = f*atan((j - 0.5*img.cols) / f) + f*atan(0.5*img.cols / f);
                y1 = f*(i - 0.5*img.rows) / sqrt(pow(j - 0.5*img.cols, 2) + pow(f, 2)) + 0.5*img.rows;
                if (x1 >= 0 && x1 < cy_cols&&y1 >= 0 && y1<cy_rows)
                {
                    img2(y1, x1) = img1(i, j);
                }
                
            }
        return img_out;
    }
    
    void crop(cv::Mat& img_out) {
        cv::Mat gray, thresh;
        std::vector<std::vector<cv::Point>> contours;
        cv::cvtColor(img_out, gray, CV_BGR2GRAY);
        cv::threshold(gray, thresh, 1, 255, CV_THRESH_BINARY);
        cv::findContours(thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//        cv::Rect r0= cv::boundingRect(cv::Mat(contours[0]));
        //        cv::rectangle(img_out, r0, cv::Scalar(0,0,255));
//        if(r0.area() > 20000)
//            img_out = img_out(r0);
        
        cv::RotatedRect r1=cv::minAreaRect(cv::Mat(contours[0]));
        cv::Point2f vertices2f[4];
        r1.points(vertices2f);
        cv::Point vertices[4];
        for(int i = 0; i < 4; ++i){
            vertices[i] = vertices2f[i];
        }
        
        float angle = r1.angle;
        cv::Size rect_size = r1.size;
        if (r1.angle < -45.) {
            angle += 90.0;
            std::swap(rect_size.width, rect_size.height);
        }
        cv::Mat M, rotated, cropped;
        M = cv::getRotationMatrix2D(r1.center, angle, 1.0);
        cv::warpAffine(img_out, rotated, M, img_out.size(), CV_INTER_CUBIC);
        // crop the resulting image
        cv::getRectSubPix(rotated, rect_size, r1.center, cropped);
        img_out = cropped;
    }
    bool find_homo(cv::Mat& img1, cv::Mat& img2, double f, int& img1_split, int& img2_split, cv::Mat& homo) {
        
        cv::Mat gray1,gray2;
        cvtColor(img1,gray1,CV_RGB2GRAY);
        cvtColor(img2,gray2,CV_RGB2GRAY);
        
        
        cv::SiftFeatureDetector sift_detector(2000);
        std::vector<cv::KeyPoint> kp1,kp2;
        sift_detector.detect(gray1,kp1);
        sift_detector.detect(gray2,kp2);
        
        
        cv::SiftDescriptorExtractor siftDescriptor;
        cv::Mat img_desc1,img_desc2;
        siftDescriptor.compute(gray1,kp1,img_desc1);
        siftDescriptor.compute(gray2,kp2,img_desc2);
        

        cv::FlannBasedMatcher matcher;
        std::vector<cv::DMatch> match_p, good_matches;
        matcher.match(img_desc1,img_desc2,match_p,cv::Mat());
        
        if(match_p.size() < 20) {
            std::cout << "no sufficient good sifts." << std::endl;
            return false;
        }
        
        std::vector<cv::Point2f> img1_p,img2_p;
        for(int i=0; i<match_p.size(); i++)
        {
            img1_p.push_back(kp1[match_p[i].queryIdx].pt);
            img2_p.push_back(kp2[match_p[i].trainIdx].pt);
        }
        
        cv::Mat img_matches;
        
//        drawMatches(img1, kp1, img2, kp2,
//                match_p, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
//                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//        imshow("good_matches", img_matches);
//        cv::waitKey();
//
        
        img2_split=(int)kp2[match_p[0].trainIdx].pt.x;
        img1_split=(int)kp1[match_p[0].queryIdx].pt.x;
        homo=cv::findHomography(img2_p,img1_p,CV_RANSAC);
        
        return true;
    }
    cv::Mat concat(cv::Mat img1, cv::Mat img2, int img1_split, int img2_split, cv::Mat& homo) {
        
        cv::Mat img_out = cv::Mat::zeros(img1.rows, img1.cols + img2.cols, CV_8UC3);
        cv::Mat ret = cv::Mat::zeros(img_out.size(), CV_8UC3);
        cv::Mat tmp = cv::Mat::zeros(img_out.size(), CV_8UC3);
        
//        cv::Mat split_trans = (cv::Mat_<double>(3,3) << 1, 0, img2_split, 0, 1, 0, 0, 0, 1);
//        cv::Mat ROIMat=img2(cv::Rect(cv::Point(img2_split,0), cv::Point(img2.cols,img2.rows)));
//        cv::warpPerspective(ROIMat,tmp, homo * trans * split_trans, img_out.size());
        cv::warpPerspective(img2,img_out,homo,img_out.size());
        
        img1.copyTo(cv::Mat(tmp,cv::Rect(0,0,img1.cols,img1.rows)));
        cv::Mat gray, thresh;
        std::vector<std::vector<cv::Point>> contours;
        cv::cvtColor(tmp, gray, CV_BGR2GRAY);
        cv::threshold(gray, thresh, 1, 255, CV_THRESH_BINARY);
        cv::findContours(thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        cv::Mat mask_image( img_out.size(), CV_8U, cv::Scalar(0));
        cv::drawContours(mask_image, contours, 0, cv::Scalar(255), CV_FILLED);
        cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3,3), cv::Point(0, 0));
        cv::erode(mask_image, mask_image, element);
        cv::erode(mask_image, mask_image, element);
        
        cv::Mat w(img_out.size(), CV_32F, cv::Scalar(0));
        
        
        for(int j = 0; j < tmp.cols; j++){
            int cnt = -1;
            for(int i = 0; i < tmp.rows; i++){
                if(mask_image.at<unsigned char>(i,j) > 0 && cnt == -1)
                    cnt = 20;
                if(cnt > 0){
                    w.at<float>(i,j) = std::max(float(cnt) / 20, w.at<float>(i,j));
                    cnt --;
                }
                if(cnt == 0) break;
            }
        }
        
        for(int j = 0; j < tmp.cols; j++){
            int cnt = -1;
            for(int i = tmp.rows-1; i >= 0; i--){
                if(mask_image.at<unsigned char>(i,j) > 0 && cnt == -1)
                    cnt = 20;
                if(cnt > 0){
                    w.at<float>(i,j) = std::max(float(cnt) / 20, w.at<float>(i,j));
                    cnt --;
                }
                if(cnt == 0) break;
            }
        }
        
        for(int i = 0; i < tmp.rows; i++){
            int cnt = -1;
            for(int j = tmp.cols-1; j >= 0; j--){
                if(mask_image.at<unsigned char>(i,j) > 0 && cnt == -1)
                    cnt = 20;
                if(cnt > 0) {
                    //                    std::cout << float(cnt) / 20 << std::endl;
                    w.at<float>(i,j) = std::max(float(cnt) / 20, w.at<float>(i,j));
                    cnt --;
                }
                if(cnt == 0) break;
            }
        }
//        tmp.copyTo(img_out, mask_image);
//        cv::imshow("fuck12", w);
        double a;
        for(int i = 0; i < tmp.rows; i++){
            for(int j = tmp.cols-1; j >= 0; j--){
                if(mask_image.at<unsigned char>(i,j) > 0)
                    a = w.at<float>(i,j);
                else
                    a = 1;
                
                if(((img_out.at<cv::Vec3b>(i,j) == cv::Vec3b(0,0,0) ||
                   img_out.at<cv::Vec3b>(std::max(0,i-1),j) == cv::Vec3b(0,0,0) ||
                   img_out.at<cv::Vec3b>(std::max(0,i-2),j) == cv::Vec3b(0,0,0) ||
                   img_out.at<cv::Vec3b>(std::min(tmp.rows-1,i+1),j) == cv::Vec3b(0,0,0) ||
                   img_out.at<cv::Vec3b>(std::min(tmp.rows-1,i+2),j) == cv::Vec3b(0,0,0) ||
                   img_out.at<cv::Vec3b>(i,std::max(0,j-1)) == cv::Vec3b(0,0,0) ||
                   img_out.at<cv::Vec3b>(i,std::min(tmp.cols-1,j+1)) == cv::Vec3b(0,0,0)))
                   && tmp.at<cv::Vec3b>(i,j) != cv::Vec3b(0,0,0) ){
//                    std::cout << i << " " << j << std::endl;
                    a = 0;
                }
                
                if(tmp.at<cv::Vec3b>(i,j) == cv::Vec3b(0,0,0))
                    a = 1;
                
                ret.at<cv::Vec3b>(i,j) = tmp.at<cv::Vec3b>(i,j) * (1-a) + img_out.at<cv::Vec3b>(i,j) * a;
            }
        }
//        cv::imshow("fuck", ret);
//        cv::waitKey(0);
//        exit(0);
        return ret;
    }
    

    virtual bool makePanorama(std::vector<cv::Mat>& img_vec,cv::Mat& img_out,double f) {
        for(int i = 0; i < img_vec.size(); i++)
            img_vec[i] = cylinder(img_vec[i], f);
        
        std::vector<cv::Mat> homos(img_vec.size());
        std::vector<int> sp1(img_vec.size()), sp2(img_vec.size());
        for(int i = 1; i < img_vec.size(); i++) {
            find_homo(img_vec[i-1], img_vec[i], f, sp1[i], sp2[i], homos[i]);
        }
        
        img_out = cv::Mat::zeros(img_vec[0].rows*2, img_vec[0].cols*2, CV_8UC3);
        
        cv::Mat trans = (cv::Mat_<double>(3,3) << 1, 0, img_vec[0].cols/2.0, 0, 1, img_vec[0].rows*1, 0, 0, 1);
        
        cv::warpPerspective(img_vec[0], img_out, trans, img_out.size());
        
        homos[1] = trans * homos[1];
        for(int i = 1; i < img_vec.size(); i++) {
            img_out = concat(img_out, img_vec[i], sp1[i], sp2[i], homos[i]);
            if(i < img_vec.size()-1)
                homos[i+1] = homos[i] * homos[i+1];
        }
        crop(img_out);
        return true;
    }

};


#endif
