//
//  CameraController.swift
//  CountriesSwiftUI
//
//  Created by startiasoft on 2024/03/05.
//  Copyright © 2024 Alexey Naumov. All rights reserved.
//

import Foundation
import AVFoundation
import UIKit

struct IdentifiablePoint: Identifiable {
    var id = UUID()
    var x: CGFloat
    var y: CGFloat
}

class CameraController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var session = AVCaptureSession()
    @Published var output = AVCaptureVideoDataOutput();
    @Published var faceRect = CGRect.zero;
    @Published var landmarks: [IdentifiablePoint] = []
    @Published var features: [Double] = []
    var input: AVCaptureDeviceInput!
    @Published var preview: AVCaptureVideoPreviewLayer!
    @Published var enhancedImg: UIImage?
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
    private var lock = false
    
    func check() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: // 已被用户同意使用摄像头
            self.setup()
            return;
        case .notDetermined: // 首次请求使用摄像头
            AVCaptureDevice.requestAccess(for: .video) {[weak self] granted in
                if granted {
                    self?.setup()
                }
            }
            return;
        case .denied: // 用户拒绝了摄像头调用申请
            return
            
        case .restricted: // 用户无法开启摄像头
            return
        @unknown default:
            return;
        }
    }
    
    func setup() {
        do {
            session.beginConfiguration()
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front);
            input = try AVCaptureDeviceInput(device: device!)
            if session.canAddInput(input) {
                session.addInput(input)
            }
            if session.canAddOutput(output) {
                session.addOutput(output)
            }
            output.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
            session.commitConfiguration()
        } catch {
            print(error.localizedDescription)
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard lock == false,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else {
            return
        }
        lock = true
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            print("错误：无法获取图像！")
            lock = false;
            return
        }
        let image = UIImage(cgImage: cgImage)
        detectFaceInImage(img: image) {[weak self] result in
            if result.enhancedImg != nil {
                self?.updateEnhancedImage(img: result.enhancedImg)
            }
            guard result.valid else {
                self?.lock = false
                if result.error != nil {
                    print(result.error?.localizedDescription ?? "发生未知错误")
                }
                return
            }
            self?.updateFaceRectAndFeatures(rect: result.faceRect!, faceFeatures: result.faceFeatures!)
            self?.lock = false
        }
    }
    
    func updateEnhancedImage(img: UIImage?) {
        DispatchQueue.main.async {
            self.enhancedImg = img;
        }
    }
        
    func updateFaceRectAndFeatures(rect: CGRect, faceFeatures: FaceFeatures) {
        DispatchQueue.main.async {
            var landmarks:[IdentifiablePoint] = []
            for value in faceFeatures.landmarks {
                let point = value as! CGPoint
                landmarks.append(IdentifiablePoint(x: point.x, y: point.y))
            }
            self.landmarks = landmarks;
            self.features = faceFeatures.features as! [Double]
            self.faceRect = rect
        }
    }
}
