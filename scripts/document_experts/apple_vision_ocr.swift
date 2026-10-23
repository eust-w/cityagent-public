import Foundation
import ImageIO
import Vision

struct Observation: Codable {
    let text: String
    let confidence: Float
    let bbox: [Double]
}

struct ImageResult: Codable {
    let path: String
    let observations: [Observation]
    let error: String?
}

func recognize(path: String) -> ImageResult {
    let url = URL(fileURLWithPath: path) as CFURL
    guard let source = CGImageSourceCreateWithURL(url, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        return ImageResult(path: path, observations: [], error: "unsupported image")
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["zh-Hans", "en-US"]
    request.usesLanguageCorrection = true

    do {
        try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
        let observations = (request.results ?? []).compactMap { item -> Observation? in
            guard let candidate = item.topCandidates(1).first else { return nil }
            let box = item.boundingBox
            return Observation(
                text: candidate.string,
                confidence: candidate.confidence,
                bbox: [box.origin.x, box.origin.y, box.size.width, box.size.height]
            )
        }.sorted {
            if abs($0.bbox[1] - $1.bbox[1]) > 0.015 { return $0.bbox[1] > $1.bbox[1] }
            return $0.bbox[0] < $1.bbox[0]
        }
        return ImageResult(path: path, observations: observations, error: nil)
    } catch {
        return ImageResult(path: path, observations: [], error: String(describing: error))
    }
}

let results = CommandLine.arguments.dropFirst().map { recognize(path: $0) }
let encoder = JSONEncoder()
encoder.outputFormatting = [.withoutEscapingSlashes]
let data = try encoder.encode(results)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
