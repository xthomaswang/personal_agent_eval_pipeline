import Foundation
import Vision
import AppKit

// Apple Vision OCR (accurate) over one PNG or every PNG in a directory: one JSON object per image with its
// text lines, their boxes ([x0, y0, x1, y1] in pixels, origin top left) and top-3 candidates.
//
//   swift ocr.swift <png or dir> <language correction: 0|1> [languages, e.g. zh-Hans,en-US]
//
// Without languages Vision reads English only, and Chinese text comes out as noise.
let path = URL(fileURLWithPath: CommandLine.arguments[1])
let correction = CommandLine.arguments[2] == "1"
let languages = CommandLine.arguments.count > 3 ? CommandLine.arguments[3].split(separator: ",").map(String.init) : []
var isDir: ObjCBool = false
FileManager.default.fileExists(atPath: path.path, isDirectory: &isDir)
let dir = isDir.boolValue ? path : path.deletingLastPathComponent()
let files = isDir.boolValue
    ? try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".png") }.sorted()
    : [path.lastPathComponent]
for f in files {
    guard let img = NSImage(contentsOf: dir.appendingPathComponent(f)),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = correction
    if !languages.isEmpty { req.recognitionLanguages = languages }
    try VNImageRequestHandler(cgImage: cg).perform([req])
    var lines: [[String: Any]] = []
    for obs in req.results ?? [] {
        let c = obs.topCandidates(3).map { ["text": $0.string, "conf": Double($0.confidence)] as [String: Any] }
        let b = obs.boundingBox  // normalized, origin bottom-left
        let box = [b.minX * Double(cg.width), (1 - b.maxY) * Double(cg.height), b.maxX * Double(cg.width), (1 - b.minY) * Double(cg.height)]
        lines.append(["candidates": c, "box": box.map { Int($0) }])
    }
    let data = try JSONSerialization.data(withJSONObject: ["file": f, "lines": lines])
    print(String(data: data, encoding: .utf8)!)
}
