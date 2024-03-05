//
//  FaceRecognizeView.swift
//  CountriesSwiftUI
//
//  Created by startiasoft on 2024/03/05.
//  Copyright © 2024 Alexey Naumov. All rights reserved.
//

import SwiftUI

struct FaceRecognizeView: View {
    @StateObject private var cameraController = CameraController.init()
    var body: some View {
        CameraPreviewView(cameraController: cameraController)
            .ignoresSafeArea()
        if cameraController.enhancedImg != nil {
            ZStack {
                HStack {
                    Spacer()
                    VStack {
                        DetectResultView(img: cameraController.enhancedImg!, landmarks: cameraController.landmarks, faceRect: cameraController.faceRect)
                    }
                }
            }
        }
    }
}

struct DetectResultView: View {
    var img: UIImage
    var landmarks: [IdentifiablePoint]
    var faceRect: CGRect
    var body: some View {
        ZStack(alignment: .topLeading) {
            Image(uiImage: img)
            Rectangle()
                .fill(Color.clear)
                .border(Color.red)
                .frame(width: faceRect.size.width, height: faceRect.size.height)
                .offset(x: faceRect.origin.x, y: faceRect.origin.y)
            
            ForEach(landmarks) { point in
                Circle()
                    .fill(Color.green)
                    .frame(width: 5, height: 5)
                    .offset(x: point.x, y: point.y)
            }
        }
    }
}
