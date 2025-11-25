//
//  modelhandler.swift
//  eomt_secondtry
//
//  Created by 이수현 on 11/17/25.
//

import CoreML
import UIKit
class ModelHandler {
    private let model: MLModel
    
    init() {
        guard let modelURL = Bundle.main.url(forResource: "EOMT_2", withExtension: "mlmodelc") else {
            fatalError("Model not found")
        }
        
        // 1️⃣ Configuration 생성
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Neural Engine + GPU + CPU 모두 사용
        // 만약 Neural Engine만 사용하고 싶다면:
        // config.computeUnits = .cpuAndNeuralEngine
        
        do {
            // 2️⃣ configuration을 전달하여 모델 로드
            self.model = try MLModel(contentsOf: modelURL, configuration: config)
        } catch {
            fatalError("Failed to load model: \(error)")
        }
    }
    
    func predict(image: UIImage) -> (classLogits: MLMultiArray, maskLogits: MLMultiArray, padInfo: CGRect)?{
        // 1. 전처리 (Letterbox Resize + Padding)
        let targetSize = CGSize(width: 640, height: 640)
        let (inputImage, padRect) = resizeWithPadding(image: image, targetSize: targetSize)
        
        // 2. CVPixelBuffer로 변환
        guard let pixelBuffer = inputImage.toCVPixelBuffer() else {
            return nil
        }
        
        // 3. 모델 입력 생성
        // MLModel의 prediction(from:)을 사용하려면 MLFeatureProvider를 만들어야 합니다.
        // 입력 키는 convert.py에서 지정한 "pixel_values"를 사용해야 합니다.
        let inputFeatures: MLFeatureProvider
        do {
            // CVPixelBuffer를 입력하는 MLFeatureProvider를 만듭니다.
            let inputDict = ["pixel_values": pixelBuffer] as [String: Any]
            inputFeatures = try MLDictionaryFeatureProvider(dictionary: inputDict)
        } catch {
            print("Error creating input feature provider: \(error)")
            return nil
        }
        
        // 4. 추론 실행
        guard let outputFeatures = try? model.prediction(from: inputFeatures) else {
            print("Error: Model prediction failed.")
            return nil
        }
        
        // 5. 결과 추출 (Output Key는 convert.py에서 지정한 이름 사용)
        guard let classLogits = outputFeatures.featureValue(for: "class_logits")?.multiArrayValue,
              let maskLogits = outputFeatures.featureValue(for: "mask_logits")?.multiArrayValue else {
            print("Error: Could not retrieve class_logits or mask_logits from model output.")
            return nil
        }
        
        // 6. 결과 튜플 반환 (전처리에서 얻은 padRect를 함께 반환)
        return (classLogits, maskLogits, padRect)
    }
}
// -----------------------------------------------------------
    // 💡 Helper: resizeWithPadding 함수 (Letterbox 구현)
    // -----------------------------------------------------------
    private func resizeWithPadding(image: UIImage, targetSize: CGSize) -> (UIImage, CGRect) {
        let originalSize = image.size
        let targetWidth = targetSize.width
        let targetHeight = targetSize.height
        
        // 비율 계산 (Python: min(target_w / orig_w, target_h / orig_h))
        let ratio = min(targetWidth / originalSize.width, targetHeight / originalSize.height)
        
        let newWidth = originalSize.width * ratio
        let newHeight = originalSize.height * ratio
        
        // 그릴 위치 계산 (중앙 정렬)
        let x = (targetWidth - newWidth) / 2
        let y = (targetHeight - newHeight) / 2
        let drawRect = CGRect(x: x, y: y, width: newWidth, height: newHeight)
        
        // 그래픽 컨텍스트 시작 (검은 배경)
        let rendererFormat = UIGraphicsImageRendererFormat.default()
        rendererFormat.scale = 1.0 // Core ML 입력을 위해 스케일을 1.0으로 강제
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: rendererFormat)
        
        let newImage = renderer.image { context in
            // 1. 검은색 채우기
            UIColor.black.setFill()
            context.fill(CGRect(origin: .zero, size: targetSize))
            
            // 2. 이미지 중앙에 그리기
            image.draw(in: drawRect)
        }
        
        // drawRect는 나중에 Crop할 때 사용됨 (Python의 pad_info)
        return (newImage, drawRect)
    }


// -----------------------------------------------------------
// 💡 Extension: UIImage -> CVPixelBuffer 변환
// -----------------------------------------------------------
extension UIImage {
    func toCVPixelBuffer() -> CVPixelBuffer? {
        // CGImage를 얻을 수 없으면 실패
        guard let cgImage = self.cgImage else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA, // Core ML ImageType 입력과 호환되는 포맷 (BGRA 또는 ARGB)
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, .init(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        // CGContext 생성
        guard let context = CGContext(
            data: pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue // 32BGRA의 경우
        ) else { return nil }
        
        // 이미지 그리기 (상하 반전 방지 및 원본 이미지 크기에 맞춤)
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1.0, y: -1.0)
        
        // Core ML 모델은 RGB 포맷을 요구하므로, 여기서는 이미지 자체를 그립니다.
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        CVPixelBufferUnlockBaseAddress(buffer, .init(rawValue: 0))
        
        return buffer
    }
}
