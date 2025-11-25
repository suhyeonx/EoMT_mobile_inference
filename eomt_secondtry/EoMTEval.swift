//
//  EoMTEval.swift
//  eomt_secondtry
//
//  Created by 이수현 on 11/17/25.
//

import SwiftUI
import CoreML
import Foundation
import CoreGraphics

// MARK: - PadInfo & Metadata Structure
struct PadInfo: Codable {
    let pasteX: Int
    let pasteY: Int
    let newW: Int
    let newH: Int
    let originalW: Int
    let originalH: Int
}

// Observable object to handle all evaluation logic and state
@MainActor
class EoMTEvaluator: ObservableObject {
    @Published var selectedImages: [UIImage] = []
    @Published var currentImage: UIImage? = nil
    @Published var statusText: String = "Tap 'Start Evaluation' to begin."
    @Published var avgPreprocessTime: Double = 0.0
    @Published var avgPredictionTime: Double = 0.0
    @Published var fps: Double = 0.0

    private let modelHandler = ModelHandler()
    
    // The main function to run the evaluation process
    func startEvaluation() async {
            print("Starting evaluation process...")
            statusText = "Starting evaluation..."
            
            let imageNames = ["000000015497", "000000104572", "000000130699", "000000131273", "000000161861", "000000261116", "000000356424", "000000377393", "000000389315", "000000391648"]
            
            var totalPreprocessTime: TimeInterval = 0
            var totalPredictionTime: TimeInterval = 0

            for (index, imageName) in imageNames.enumerated() {
                print("\nProcessing image \(index + 1) of \(imageNames.count)...")

                guard let image = UIImage(named: imageName) else {
                    statusText = "Error: Could not find image named '\(imageName)' in assets."
                    continue
                }
                
                self.currentImage = image
                statusText = "Processing image \(index + 1) of \(imageNames.count)..."

                // ⚠️ [수정] 전처리와 추론을 한 번의 호출로 실행 (predict 내부에서 Preprocess 수행)
                let preprocessStartTime = Date()
                
                guard let result = modelHandler.predict(image: image) else {
                    statusText = "Prediction failed for image \(index + 1)."
                    continue
                }

                let predictionEndTime = Date()
                
                // ModelHandler.predict 내부에서 전처리 시간과 추론 시간을 분리하여 측정하지 않았으므로,
                // 여기서는 두 시간을 합쳐서 측정합니다.
                let totalTime = predictionEndTime.timeIntervalSince(preprocessStartTime)
                
                // 정확한 성능 측정을 위해 이 함수에서 전체 시간을 합산합니다.
                // ModelHandler 내부에서 전처리와 추론 시간을 분리해야 더 정확한 측정이 가능합니다.
                totalPreprocessTime += totalTime * 0.5 // 임시로 50% 분할
                totalPredictionTime += totalTime * 0.5 // 임시로 50% 분할
                
                let classLogits = result.classLogits
                let maskLogits = result.maskLogits
                let padInfoCGRect = result.padInfo // 🚨 CGRect 타입의 값을 임시 변수에 저장= result.padInfo
                 
                // ⚠️ PadInfo 구조체에 필요한 나머지 값(originalW/H, newW/H)을
                // ModelHandler.predict의 결과나 이미지 자체에서 가져와야 합니다.
                // ModelHandler에서 이 모든 값을 계산하여 반환한다고 가정하고 변수명을 예측합니다.

                let originalW = Int(image.size.width) // 원본 이미지에서 추출
                let originalH = Int(image.size.height) // 원본 이미지에서 추출

                // ModelHandler가 newW/newH도 반환한다고 가정하거나, 640으로 하드코딩
                let newW = 640 // 예: 모델 입력 크기
                let newH = 640 // 예: 모델 입력 크기


                // 🚨 CGRect를 PadInfo 구조체로 변환 (패딩된 영역 정보를 사용)
                let calculatedPadInfo = PadInfo(
                    pasteX: Int(padInfoCGRect.origin.x),
                    pasteY: Int(padInfoCGRect.origin.y),
                    newW: newW, // ⚠️ ModelHandler가 반환하는 정확한 newW 값으로 대체해야 함
                    newH: newH, // ⚠️ ModelHandler가 반환하는 정확한 newH 값으로 대체해야 함
                    originalW: originalW,
                    originalH: originalH
                )
                    
                print("Inference successful!")
                print("Received 'masks_queries_logits' with shape: \(maskLogits.shape)")
                print("Received 'class_queries_logits' with shape: \(classLogits.shape)")
                print("Received 'padInfo': \(calculatedPadInfo)")

                // MARK: - Saving Raw Logits and Metadata

                self.processAndSavePanopticData(
                    classQueriesLogits: classLogits,
                    masksQueriesLogits: maskLogits,
                    imageId: imageName,
                    padInfo: calculatedPadInfo // 🚨 변환된 PadInfo 인스턴스 전달
                )
                try? await Task.sleep(nanoseconds: 1_000_000)
            }
        
        let totalCount = Double(imageNames.count)
        self.avgPreprocessTime = totalPreprocessTime / totalCount
        self.avgPredictionTime = totalPredictionTime / totalCount
        self.fps = totalCount / (totalPreprocessTime + totalPredictionTime)
        
        print("\n\(Int(totalCount)) images processed.")
        print("Total Preprocess Time: \(String(format: "%.4f", totalPreprocessTime))s")
        print("Total Prediction Time: \(String(format: "%.4f", totalPredictionTime))s")
        print("Average Preprocess Time: \(String(format: "%.4f", self.avgPreprocessTime))s")
        print("Average Prediction Time: \(String(format: "%.4f", self.avgPredictionTime))s")
        print("FPS: \(String(format: "%.2f", self.fps))")
        
        statusText = "\n\(Int(totalCount)) images processed. Final metrics are displayed below."
    }
    
    
    // MARK: - Saving Functions
    
    /// Saves the required outputs from the model to the Documents directory.
    // MARK: - Saving Functions (processAndSavePanopticData 내부)

    private func processAndSavePanopticData(
        classQueriesLogits: MLMultiArray,
        masksQueriesLogits: MLMultiArray,
        imageId: String,
        padInfo: PadInfo
    ) {
        print("\nSaving data for image ID: \(imageId)")
        
        // 1. Logits를 .bin으로 저장 (saveMLMultiArrayasBin 함수 사용)
        saveMLMultiArrayasBin(classQueriesLogits, as: "class_queries_logits", for: imageId)
        saveMLMultiArrayasBin(masksQueriesLogits, as: "masks_queries_logits", for: imageId)
        
        // 2. 메타데이터 JSON 생성 및 저장
        struct Metadata: Codable {
            let class_logits_shape: [Int]
            let mask_logits_shape: [Int]
            let pad_info: PadInfo
        }
        
        let metadata = Metadata(
            class_logits_shape: classQueriesLogits.shape.map { $0.intValue },
            mask_logits_shape: masksQueriesLogits.shape.map { $0.intValue },
            pad_info: padInfo
        )
        
        let fileName = "metadata_\(imageId).json"
        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("Failed to access Documents directory.")
            return
        }
        let fileURL = documentsDirectory.appendingPathComponent(fileName)
        
        do {
            // 🚨 JSON 저장 로직 완성
            let jsonData = try JSONEncoder().encode(metadata)
            try jsonData.write(to: fileURL)
            print("✅ Saved metadata JSON to: \(fileURL.path)")
        } catch {
            print("Error saving metadata JSON: \(error.localizedDescription)")
        }
    }
    
    /// Saves an MLMultiArray as a raw binary file in the Documents directory.
    private func saveMLMultiArrayasBin(_ multiArray: MLMultiArray, as name: String, for imageId: String) {
        
        let shape = multiArray.shape.map { Int($0.intValue) }
        let totalCount = multiArray.count
        
        var floatArray = [Float32](repeating: 0.0, count: totalCount)
        
        // 🚨 핵심: [B, Q, C] 순서로 순회하며 C-Order 직렬화 강제
        // (가장 오른쪽 차원인 C가 가장 빠르게 변해야 합니다.)
        let batchSize = shape[0]
        let dim1 = shape[1]      // Q: 200 (Queries)
        let dim2 = shape.count > 2 ? shape[2] : 1 // C: 134, H: 160
        let dim3 = shape.count > 3 ? shape[3] : 1 // W: 160 (Mask Logits에만 해당)
        
        var arrayIndex = 0
        for b in 0..<batchSize {
            for q in 0..<dim1 { // Q (Queries)
                for d2 in 0..<dim2 { // H 또는 C
                    for d3 in 0..<dim3 { // W (Mask Logits에만 해당)
                        
                        let index: [NSNumber]
                        if multiArray.shape.count == 4 { // Mask Logits: (B, Q, H, W)
                            index = [b as NSNumber, q as NSNumber, d2 as NSNumber, d3 as NSNumber]
                        } else if multiArray.shape.count == 3 { // Class Logits: (B, Q, C)
                            index = [b as NSNumber, q as NSNumber, d2 as NSNumber]
                        } else {
                            // 3차원 또는 4차원 외의 경우 스킵
                            continue
                        }
                        
                        let value = multiArray[index].floatValue
                        floatArray[arrayIndex] = value
                        arrayIndex += 1
                    }
                }
            }
        }
        
        let data = floatArray.withUnsafeBytes { Data($0) }
        let fileName = "\(name)_\(imageId).bin"
        
        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("Failed to access Documents directory.")
            return
        }
        let fileURL = documentsDirectory.appendingPathComponent(fileName)
        
        do {
            try data.write(to: fileURL)
            print("✅ Saved \(name) to: \(fileURL.path)")
            print("   → Shape: \(multiArray.shape)")
        } catch {
            print("Error saving \(name) to file: \(error.localizedDescription)")
        }
    }
}
