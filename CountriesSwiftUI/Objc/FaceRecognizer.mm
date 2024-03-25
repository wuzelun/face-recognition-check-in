//
//  FaceRecognizer.m
//  CountriesSwiftUI
//
//  Created by startiasoft on 2024/03/05.
//  Copyright © 2024 Alexey Naumov. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#include <dlib/dnn.h>
#import <dlib/image_processing.h>
#import <dlib/image_processing/frontal_face_detector.h>
#import <dlib/image_processing/render_face_detections.h>
#import <dlib/opencv.h>
#import "FaceRecognizer.h"
#import <stdio.h>

#define MIN_IMG_SIZE 400.0
#define MIN_SOBEL_VALUE 2.0
#define MIN_BRIGHTNESS_VALUE 80
#define MAX_BRIGHTNESS_VALUE 200

@interface FaceRecognizer()
{
    std::deque<int>    _eyesCounter;
    cv::CascadeClassifier  _eyeCascade;
    cv::CascadeClassifier  _faceCascade;
}
@end


using namespace dlib;
using namespace std;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using ResNet = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;
@implementation FaceFeatures
@end
@implementation FaceRecognizer
FaceRecognizer *mFaceRecognizer;
dlib::shape_predictor sp;
ResNet net;

+ (FaceRecognizer *) shared {
    if(!mFaceRecognizer) {
        mFaceRecognizer = [FaceRecognizer new];
        NSString *landmarkPath = [[NSBundle mainBundle] pathForResource:@"shape_predictor_68_face_landmarks" ofType:@"dat"];
        std::string landmarkFileString = [landmarkPath UTF8String];
        NSString *resnetModelPath = [[NSBundle mainBundle] pathForResource:@"dlib_face_recognition_resnet_model_v1" ofType:@"dat"];
        std::string resnetFileString = [resnetModelPath UTF8String];
        deserialize(landmarkFileString) >> sp;
        deserialize(resnetFileString) >> net;
//        [mFaceRecognizer setupDetector];
    }
    return mFaceRecognizer;
}
// 特征提取
- (FaceFeatures *) genFeatures: (UIImage *) img withFaceRect: (CGRect) rect {
    cv::Mat src;
    UIImageToMat(img, src);
    cv::cvtColor(src, src, cv::COLOR_RGBA2BGR);
    dlib::array2d<dlib::bgr_pixel> dlibImage;
    dlib::assign_image(dlibImage, dlib::cv_image<dlib::bgr_pixel>(src));
    long left = rect.origin.x;
    long top = rect.origin.y;
    long right = left + rect.size.width;
    long bottom = top + rect.size.height;
    dlib::rectangle det(left, top, right, bottom);
    dlib::full_object_detection shape = sp(dlibImage, det);
    NSMutableArray *landmarks = [NSMutableArray array];
    for (int i = 0; i < shape.num_parts(); i++) {
        dlib::point p = shape.part(i);
        CGPoint point = CGPointMake(p.x(), p.y());
        NSValue *pValue = [NSValue valueWithCGPoint:point];
        [landmarks addObject:pValue];
    }
    matrix<rgb_pixel> face_chip;
    extract_image_chip(dlibImage, get_face_chip_details(shape,150,0.25), face_chip);
    //图片和特征点坐标传入网络，获得 face_descriptor
    matrix<float,0,1> face_descriptor = net(move(face_chip));
    NSMutableArray *features = [NSMutableArray array];
    for (auto value : face_descriptor) {
        NSNumber *num = [NSNumber numberWithDouble:value];
        [features addObject:num];
    }
    FaceFeatures *faceFeatures = [FaceFeatures new];
    faceFeatures.features = features;
    faceFeatures.landmarks = landmarks;
    return faceFeatures;
}
// 图像增强
- (UIImage *) enhanceImage: (UIImage *) img {
    cv::Mat src;
    UIImageToMat(img, src);
    resizeImg(src);
    cv::medianBlur(src, src, 3);
    cv::Mat sharpen_op = (cv::Mat_<char>(3, 3) <<
                          0, -1, 0,
                          -1, 5, -1,
                          0, -1, 0);
    filter2D(src, src, CV_32F, sharpen_op);
    convertScaleAbs(src, src);
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
    return MatToUIImage(src);
}
void resizeImg(cv::Mat &img) {
    int newWidth, newHeight;
    double ratio = img.cols * 1.0 / img.rows * 1.0; //图片长高比
    if (ratio > 1) {
        newHeight = MIN_IMG_SIZE;
        newWidth = MIN_IMG_SIZE * ratio;
    } else {
        newWidth = MIN_IMG_SIZE;
        newHeight = MIN_IMG_SIZE / ratio;
    }
    cv::resize(img, img, cv::Size(newWidth, newHeight));
    cv::cvtColor(img, img, cv::COLOR_RGBA2BGR);
}

//设置检测器
- (void)setupDetector{
    NSString *faceCascadePath = [[NSBundle mainBundle] pathForResource:@"haarcascade_frontalface_alt2"
                                                                ofType:@"xml"];
    const CFIndex CASCADE_NAME_LEN = 2048;
    char *CASCADE_NAME = (char *) malloc(CASCADE_NAME_LEN);
    CFStringGetFileSystemRepresentation( (CFStringRef)faceCascadePath, CASCADE_NAME, CASCADE_NAME_LEN);
    _faceCascade.load(CASCADE_NAME);


    NSString *eyesCascadePath = [[NSBundle mainBundle] pathForResource:@"haarcascade_eye"
                                                                ofType:@"xml"];
    CFStringGetFileSystemRepresentation( (CFStringRef)eyesCascadePath, CASCADE_NAME, CASCADE_NAME_LEN);
    
    _eyeCascade.load(CASCADE_NAME);
    free(CASCADE_NAME);
}

//检测人脸
- (std::vector<cv::Rect>)checkFacesWithImage:(cv::Mat &)image{
    
    std::vector<cv::Rect> rects;
    cv::Mat gray, smallImg( cvRound (image.rows), cvRound(image.cols), CV_8UC1 );
    cvtColor( image, gray, cv::COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    double scalingFactor = 1.1;
    int minRects = 2;
    cv::Size minSize(30,30);
    _faceCascade.detectMultiScale( smallImg, rects,
                                   scalingFactor, minRects, 0,
                                   minSize );
    return rects;
}

//检测眨眼
- (void)checkBlickWithRects:(std::vector<cv::Rect>)rects andImage:(cv::Mat &)image{
    
    cv::Rect& faceR = rects[0];
    cv::Rect faceEyeZone( cv::Point(faceR.x + 0.12f * faceR.width,
                                    faceR.y + 0.17f * faceR.height),
                         cv::Size(0.76 * faceR.width,
                                  0.4f * faceR.height));
    rects.clear();
    cv::rectangle(image, faceR, cvScalar(0,255,0));
    cv::rectangle(image, faceEyeZone, cvScalar(0,255,0));
    cv::Mat eyeImage(image, faceEyeZone);
    _eyeCascade.detectMultiScale(eyeImage, rects, 1.2f, 5, CV_HAAR_SCALE_IMAGE,
                                  cv::Size(faceEyeZone.width * 0.2f, faceEyeZone.width * 0.2f),
                                  cv::Size(0.5f * faceEyeZone.width, 0.7f * faceEyeZone.height));
    [self registerEyesCount:(int)rects.size()];

}

//记录人眼
- (void)registerEyesCount:(int)count {
    
    NSLog(@"注册:%d",count);
    if (_eyesCounter.empty() || (_eyesCounter[_eyesCounter.size() - 1] != count))
        _eyesCounter.push_back(count);
    
    if (_eyesCounter.size() > 3)
        _eyesCounter.pop_front();
};

//检测眨眼
- (BOOL)checkBlink {
    if (_eyesCounter.size() == 3){
        return (_eyesCounter[2] > 0)
        &&
        (_eyesCounter[1] == 0)
        &&
        (_eyesCounter[0] > 0);
    }
    return NO;
};

- (BOOL) checkAlive: (UIImage *) img {
    cv::Mat src;
    UIImageToMat(img, src);
    
    std::vector<cv::Rect> rects = [self checkFacesWithImage:src];
    
    if(rects.size() > 0) {
        [self checkBlickWithRects:rects andImage:src];
    }

    return [self checkBlink];
}



@end
