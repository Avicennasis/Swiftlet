import Foundation
import Testing
@testable import SwiftletCore

/// The Metal runtime must reproduce the CPU reference: same greedy tokens,
/// near-identical logits, on both quantized (int4) and plain (f32) tiny models
/// and both DeltaNet layouts.
@Suite struct MetalModelTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    static func compare(_ modelName: String) throws {
        let dir = fixturesDir.appendingPathComponent(modelName)
        let cpu = try QwenCPUModel(modelDir: dir)
        cpu.retainAllLayers = true
        let gpu = try QwenMetalModel(modelDir: dir)
        let tokens = [1, 5, 9, 42, 7]

        let cpuState = QwenCPUModel.DecodeState()
        let gpuState = QwenCPUModel.DecodeState()
        var cpuLogits: [Float] = []
        var gpuLogits: [Float] = []
        for t in tokens {
            cpuLogits = try cpu.step([t], state: cpuState)
            gpuLogits = try gpu.step([t], state: gpuState)
        }

        var maxDiff: Float = 0
        for i in 0..<cpuLogits.count { maxDiff = max(maxDiff, abs(cpuLogits[i] - gpuLogits[i])) }
        #expect(maxDiff < 2e-3, "\(modelName): GPU vs CPU logits maxAbsDiff \(maxDiff)")

        func argmax(_ v: [Float]) -> Int {
            var b = 0
            for i in 1..<v.count where v[i] > v[b] { b = i }
            return b
        }
        #expect(argmax(cpuLogits) == argmax(gpuLogits), "\(modelName): greedy diverged")
    }

    @Test func gpuMatchesCPUOnQuantizedTiny() throws {
        try Self.compare("tiny-model-q4")
    }

    /// Full streaming path: repack tiny model to .qpack, run the GPU model in
    /// cache/pread mode, compare against the CPU reference on the raw dir.
    @Test func gpuQpackStreamingMatchesCPU() throws {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-q4-gpu-\(UUID().uuidString).qpack")
        defer { try? FileManager.default.removeItem(at: out) }
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()

        let cpu = try QwenCPUModel(modelDir: src)
        cpu.retainAllLayers = true
        let gpu = try QwenMetalModel(modelDir: out, cacheBudgetGB: 0.05)
        #expect(gpu.expertCache != nil, "qpack mode not engaged")

        let tokens = [1, 5, 9, 42, 7, 99]
        let cpuState = QwenCPUModel.DecodeState()
        let gpuState = QwenCPUModel.DecodeState()
        var cpuLogits: [Float] = []
        var gpuLogits: [Float] = []
        for t in tokens {
            cpuLogits = try cpu.step([t], state: cpuState)
            gpuLogits = try gpu.step([t], state: gpuState)
        }
        var maxDiff: Float = 0
        for i in 0..<cpuLogits.count { maxDiff = max(maxDiff, abs(cpuLogits[i] - gpuLogits[i])) }
        #expect(maxDiff < 2e-3, "qpack GPU vs CPU logits maxAbsDiff \(maxDiff)")
        let cache = gpu.expertCache!
        #expect(cache.hits + cache.misses > 0)
    }

    @Test func gpuMatchesCPUOnQwen35Tiny() throws {
        try Self.compare("tiny-model-q35")
    }
}
