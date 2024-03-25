//
//  FaceRecognizer.h
//  CountriesSwiftUI
//
//  Created by startiasoft on 2024/03/05.
//  Copyright © 2024 Alexey Naumov. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN
@interface FaceFeatures: NSObject
@property(strong, nonatomic) NSArray *landmarks;
@property(strong, nonatomic) NSArray *features;
@end
@interface FaceRecognizer : NSObject
+ (FaceRecognizer *) shared;
- (UIImage *) enhanceImage: (UIImage *) img;
- (BOOL) checkAlive: (UIImage *) img;
- (FaceFeatures *) genFeatures: (UIImage *) img withFaceRect: (CGRect) rect;
@end

NS_ASSUME_NONNULL_END
