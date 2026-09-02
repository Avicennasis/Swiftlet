import Foundation
import Testing
@testable import SwiftletCore

/// The Metal runtime must reproduce the CPU reference: same greedy tokens,
/// near-identical logits, on both quantized (int4) and plain (f32) tiny models
/// and both DeltaNet layouts.
@Suite struct MetalModelTests {
    /// S3a whole-step aggregate regression baseline. These counts intentionally
    /// do not assign work to phases or make timeline/performance claims.
    struct FastPathBaseline {
        let commandBuffersPerToken: Int
        let intermediateDispatches: Int
        let finalDispatches: Int

        func commandBuffers(tokens: Int) -> Int {
            commandBuffersPerToken * tokens
        }

        func dispatches(tokens: Int) -> Int {
            guard tokens > 0 else { return 0 }
            return intermediateDispatches * (tokens - 1) + finalDispatches
        }
    }

    static let commandBuffersPerToken = 11
    static let q4Baseline = FastPathBaseline(
        commandBuffersPerToken: commandBuffersPerToken,
        intermediateDispatches: 212,
        finalDispatches: 214
    )
    static let q35Baseline = FastPathBaseline(
        commandBuffersPerToken: commandBuffersPerToken,
        intermediateDispatches: 218,
        finalDispatches: 220
    )

    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    static func maxAbsDiff(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return .infinity }
        var result: Float = 0
        for i in lhs.indices { result = max(result, abs(lhs[i] - rhs[i])) }
        return result
    }

    static func expectMatchingKV(
        _ lhs: QwenCPUModel.DecodeState, _ rhs: QwenCPUModel.DecodeState,
        label: String
    ) {
        #expect(lhs.position == rhs.position, "\(label): positions diverged")
        #expect(Set(lhs.kv.keys) == Set(rhs.kv.keys), "\(label): KV layers diverged")
        for layer in lhs.kv.keys {
            guard let l = lhs.kv[layer], let r = rhs.kv[layer] else { continue }
            #expect(l.k.count == r.k.count, "\(label): layer \(layer) K size diverged")
            #expect(l.v.count == r.v.count, "\(label): layer \(layer) V size diverged")
            #expect(Self.maxAbsDiff(l.k, r.k) < 1e-6, "\(label): layer \(layer) K diverged")
            #expect(Self.maxAbsDiff(l.v, r.v) < 1e-6, "\(label): layer \(layer) V diverged")
        }
    }

    static func expectInstrumentation(
        _ metrics: QwenMetalModel.StepMetrics, tokens: Int, label: String
    ) {
        #expect(metrics.completedWithoutThrow, "\(label): step threw")
        #expect(metrics.tokensProcessed == tokens, "\(label): token count")
        #expect(metrics.logitProjections == 1, "\(label): LM-head count")
        #expect(metrics.commandBuffersCommitted > 0, "\(label): no command buffers")
        #expect(metrics.blockingWaits == metrics.commandBuffersCommitted,
                "\(label): commit/wait mismatch")
        #expect(metrics.commandBufferErrors == 0, "\(label): command-buffer error")
        #expect(metrics.computeDispatchesEncoded > 0, "\(label): no compute dispatches")
        #expect(metrics.gpuTimedCommandBuffers + metrics.gpuUntimedCommandBuffers
                + metrics.commandBufferErrors == metrics.commandBuffersCommitted,
                "\(label): GPU timing samples")
    }

    static func expectFastPathBaseline(
        _ metrics: QwenMetalModel.StepMetrics,
        tokens: Int,
        baseline: FastPathBaseline,
        label: String
    ) {
        #expect(metrics.commandBuffersCommitted == baseline.commandBuffers(tokens: tokens),
                "\(label): fast-path command-buffer baseline changed")
        #expect(metrics.computeDispatchesEncoded == baseline.dispatches(tokens: tokens),
                "\(label): fast-path dispatch baseline changed")
    }

    static func compare(_ modelName: String, baseline: FastPathBaseline) throws {
        let dir = fixturesDir.appendingPathComponent(modelName)
        let cpu = try QwenCPUModel(modelDir: dir)
        cpu.retainAllLayers = true
        let sequentialGPU = try QwenMetalModel(modelDir: dir)
        let elidingGPU = try QwenMetalModel(modelDir: dir)
        let tokens = [1, 5, 9, 42, 7]

        let cpuState = QwenCPUModel.DecodeState()
        let sequentialState = QwenCPUModel.DecodeState()
        var cpuLogits: [Float] = []
        var sequentialLogits: [Float] = []
        for t in tokens {
            cpuLogits = try cpu.step([t], state: cpuState)
            sequentialLogits = try sequentialGPU.step([t], state: sequentialState)
            #expect(sequentialGPU.lastStepMetrics.tokensProcessed == 1)
            #expect(sequentialGPU.lastStepMetrics.logitProjections == 1)
            #expect(sequentialGPU.lastStepMetrics.avoidedLogitProjections == 0)
        }
        let singleMetrics = sequentialGPU.lastStepMetrics
        Self.expectInstrumentation(singleMetrics, tokens: 1, label: "\(modelName) one token")
        Self.expectFastPathBaseline(
            singleMetrics, tokens: 1, baseline: baseline, label: "\(modelName) one token"
        )

        let maxDiff = Self.maxAbsDiff(cpuLogits, sequentialLogits)
        #expect(maxDiff < 2e-3, "\(modelName): GPU vs CPU logits maxAbsDiff \(maxDiff)")

        // S1a intermediate LM-head elision must retain token-at-a-time output
        // and state. Separate instances isolate GPU-resident recurrence.
        let elidingState = QwenCPUModel.DecodeState()
        let elidingLogits = try elidingGPU.step(tokens, state: elidingState)
        let multiMetrics = elidingGPU.lastStepMetrics
        let elisionDiff = Self.maxAbsDiff(sequentialLogits, elidingLogits)
        #expect(elisionDiff < 2e-3, "\(modelName): LM-head elision logits maxAbsDiff \(elisionDiff)")
        #expect(elidingGPU.lastStepMetrics.tokensProcessed == tokens.count)
        #expect(elidingGPU.lastStepMetrics.logitProjections == 1)
        #expect(elidingGPU.lastStepMetrics.avoidedLogitProjections == tokens.count - 1)
        Self.expectInstrumentation(multiMetrics, tokens: tokens.count, label: "\(modelName) multi token")
        Self.expectFastPathBaseline(
            multiMetrics, tokens: tokens.count, baseline: baseline, label: "\(modelName) multi token"
        )
        Self.expectMatchingKV(sequentialState, elidingState, label: "\(modelName) elision input")

        let continuation = 11
        let sequentialContinuation = try sequentialGPU.step([continuation], state: sequentialState)
        let elidingContinuation = try elidingGPU.step([continuation], state: elidingState)
        let continuationDiff = Self.maxAbsDiff(sequentialContinuation, elidingContinuation)
        #expect(continuationDiff < 2e-3, "\(modelName): continuation maxAbsDiff \(continuationDiff)")
        #expect(elidingGPU.lastStepMetrics.tokensProcessed == 1)
        #expect(elidingGPU.lastStepMetrics.logitProjections == 1)
        #expect(elidingGPU.lastStepMetrics.avoidedLogitProjections == 0)
        Self.expectInstrumentation(
            elidingGPU.lastStepMetrics, tokens: 1, label: "\(modelName) continuation"
        )
        Self.expectFastPathBaseline(
            elidingGPU.lastStepMetrics, tokens: 1,
            baseline: baseline, label: "\(modelName) continuation"
        )
        Self.expectMatchingKV(sequentialState, elidingState, label: "\(modelName) continuation")

        func argmax(_ v: [Float]) -> Int {
            var b = 0
            for i in 1..<v.count where v[i] > v[b] { b = i }
            return b
        }
        #expect(argmax(cpuLogits) == argmax(sequentialLogits), "\(modelName): greedy diverged")
    }

    @Test func gpuMatchesCPUOnQuantizedTiny() throws {
        try Self.compare("tiny-model-q4", baseline: Self.q4Baseline)
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
        let sequentialGPU = try QwenMetalModel(modelDir: out, cacheBudgetGB: 0.05)
        let elidingGPU = try QwenMetalModel(modelDir: out, cacheBudgetGB: 0.05)
        #expect(sequentialGPU.expertCache != nil, "qpack mode not engaged")
        #expect(elidingGPU.expertCache != nil, "qpack mode not engaged")

        let tokens = [1, 5, 9, 42, 7, 99]
        let cpuState = QwenCPUModel.DecodeState()
        let sequentialState = QwenCPUModel.DecodeState()
        var cpuLogits: [Float] = []
        var sequentialLogits: [Float] = []
        for t in tokens {
            cpuLogits = try cpu.step([t], state: cpuState)
            sequentialLogits = try sequentialGPU.step([t], state: sequentialState)
            #expect(sequentialGPU.lastStepMetrics.tokensProcessed == 1)
            #expect(sequentialGPU.lastStepMetrics.logitProjections == 1)
            #expect(sequentialGPU.lastStepMetrics.avoidedLogitProjections == 0)
        }
        let singleMetrics = sequentialGPU.lastStepMetrics
        Self.expectInstrumentation(singleMetrics, tokens: 1, label: "qpack one token")
        Self.expectFastPathBaseline(
            singleMetrics, tokens: 1, baseline: Self.q4Baseline, label: "qpack one token"
        )
        let maxDiff = Self.maxAbsDiff(cpuLogits, sequentialLogits)
        #expect(maxDiff < 2e-3, "qpack GPU vs CPU logits maxAbsDiff \(maxDiff)")

        let elidingState = QwenCPUModel.DecodeState()
        let elidingLogits = try elidingGPU.step(tokens, state: elidingState)
        let multiMetrics = elidingGPU.lastStepMetrics
        let elisionDiff = Self.maxAbsDiff(sequentialLogits, elidingLogits)
        #expect(elisionDiff < 2e-3, "qpack LM-head elision maxAbsDiff \(elisionDiff)")
        #expect(elidingGPU.lastStepMetrics.tokensProcessed == tokens.count)
        #expect(elidingGPU.lastStepMetrics.logitProjections == 1)
        #expect(elidingGPU.lastStepMetrics.avoidedLogitProjections == tokens.count - 1)
        Self.expectInstrumentation(multiMetrics, tokens: tokens.count, label: "qpack multi token")
        Self.expectFastPathBaseline(
            multiMetrics, tokens: tokens.count,
            baseline: Self.q4Baseline, label: "qpack multi token"
        )
        Self.expectMatchingKV(sequentialState, elidingState, label: "qpack elision input")

        let continuation = 11
        let sequentialContinuation = try sequentialGPU.step([continuation], state: sequentialState)
        let elidingContinuation = try elidingGPU.step([continuation], state: elidingState)
        let continuationDiff = Self.maxAbsDiff(sequentialContinuation, elidingContinuation)
        #expect(continuationDiff < 2e-3, "qpack continuation maxAbsDiff \(continuationDiff)")
        #expect(elidingGPU.lastStepMetrics.logitProjections == 1)
        Self.expectInstrumentation(
            elidingGPU.lastStepMetrics, tokens: 1, label: "qpack continuation"
        )
        Self.expectFastPathBaseline(
            elidingGPU.lastStepMetrics, tokens: 1,
            baseline: Self.q4Baseline, label: "qpack continuation"
        )
        Self.expectMatchingKV(sequentialState, elidingState, label: "qpack continuation")

        for cache in [sequentialGPU.expertCache!, elidingGPU.expertCache!] {
            #expect(cache.hits + cache.misses > 0)
            #expect(cache.allocatedSlots > 0)
            #expect(cache.allocatedBytes >= cache.logicalBytes)
            #expect(cache.allocatedBytes <= cache.budgetBytes)
        }

        // A pressure request below the minimum working set must not replace a
        // functioning qpack cache with nil.
        let cacheBeforeImpossibleShrink = sequentialGPU.expertCache!
        sequentialGPU.shrinkCache(toGB: 0)
        #expect(sequentialGPU.expertCache === cacheBeforeImpossibleShrink)
    }

    @Test func gpuMatchesCPUOnQwen35Tiny() throws {
        try Self.compare("tiny-model-q35", baseline: Self.q35Baseline)
    }
}
