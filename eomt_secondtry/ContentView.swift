
//  ContentView1.swift
//  eomt_ios_loaddata
//
//  Created by MA-003 on 9/2/25.
//
import SwiftUI
import CoreML

struct ContentView: View {
    @StateObject private var evaluator = EoMTEvaluator()

    // A simple UI to display the image and the metrics
    var body: some View {
        VStack(spacing: 20) {
            Text("EoMT Inference")
                .font(.title)
                .fontWeight(.bold)
                .padding(.top)

            if let image = evaluator.currentImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .cornerRadius(12)
                    .shadow(radius: 5)
                    .padding(.horizontal)
            } else {
                Spacer()
                ProgressView("Loading...")
                Spacer()
            }
            
            Text(evaluator.statusText)
                .font(.headline)
                .padding(.horizontal)

            VStack(alignment: .leading, spacing: 5) {
                Text("Evaluation Timing:")
                    .font(.title3)
                    .fontWeight(.semibold)

                HStack {
                    Text("Avg Preprocess Time:")
                    Spacer()
                    Text(String(format: "%.4f s", evaluator.avgPreprocessTime))
                }
                HStack {
                    Text("Avg Prediction Time:")
                    Spacer()
                    Text(String(format: "%.4f s", evaluator.avgPredictionTime))
                }
                HStack {
                    Text("FPS:")
                    Spacer()
                    Text(String(format: "%.2f", evaluator.fps))
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(10)
            .shadow(radius: 2)
            .padding(.horizontal)

            Spacer()

            Button("Start Evaluation") {
                Task {
                    await evaluator.startEvaluation()
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            
        }
    }
}
