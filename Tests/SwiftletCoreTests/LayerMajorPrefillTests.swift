import Foundation
import Testing
@testable import SwiftletCore

/// S1b: layer-major, chunked prefill with expert-union routing. The bar:
/// final logits, greedy continuation, and KV/recurrent state must match the
/// legacy token-major path under the S1a tolerance discipline; the experts
/// touched per layer must be exactly the S1b-a oracle's planned unions; and
/// the new schedule's buffer/dispatch shape is pinned as exactly as the S3a
/// baselines pin the old one. Decode steps never take this path.
@Suite struct LayerMajorPrefillTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    /// Layer-major fast-path baseline: one decode-token-shaped buffer
    /// sequence per chunk (the S2 9-buffer shape), per-token dispatch cost
    /// identical to the decode schedule, and the LM head only after the
    /// final chunk. Exact assertions, S3a style.
    struct LayerMajorBaseline {
        let commandBuffersPerChunk: Int
        let dispatchesPerToken: Int
        let lmHeadDispatches: Int

        func chunkCount(tokens: Int, chunkTokens: Int) -> Int {
            (tokens + chunkTokens - 1) / chunkTokens
        }

        func commandBuffers(tokens: Int, chunkTokens: Int) -> Int {
            commandBuffersPerChunk * chunkCount(tokens: tokens, chunkTokens: chunkTokens)
        }

        func dispatches(tokens: Int) -> Int {
            dispatchesPerToken * tokens + lmHeadDispatches
        }
    }

    /// S2: the chunk keeps the decode shape — 9 buffers per chunk (was 11
    /// around the CPU attention core) and +3 dispatches per attention layer
    /// per token (q prep, KV append, causal attention), so 212 -> 218 (q4)
    /// and 218 -> 224 (q35) dispatches per token.
    static let q4Baseline = LayerMajorBaseline(
        commandBuffersPerChunk: MetalModelTests.commandBuffersPerToken,
        dispatchesPerToken: 218,
        lmHeadDispatches: 2
    )
    static let q35Baseline = LayerMajorBaseline(
        commandBuffersPerChunk: MetalModelTests.commandBuffersPerToken,
        dispatchesPerToken: 224,
        lmHeadDispatches: 2
    )

    static func argmax(_ v: [Float]) -> Int {
        var b = 0
        for i in 1..<v.count where v[i] > v[b] { b = i }
        return b
    }

    /// The shipped default is the S1b schedule with a bounded chunk.
    @Test func defaultPrefillModeIsLayerMajor() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let model = try QwenMetalModel(modelDir: dir)
        #expect(model.prefillMode == .layerMajor(chunkTokens: 32))
    }

    static func compare(
        _ modelName: String,
        chunkTokens: Int,
        baseline: LayerMajorBaseline,
        decodeBaseline: MetalModelTests.FastPathBaseline
    ) throws {
        let dir = fixturesDir.appendingPathComponent(modelName)
        let label = "\(modelName) chunk=\(chunkTokens)"
        let tokens = [1, 5, 9, 42, 7]

        let cpu = try QwenCPUModel(modelDir: dir)
        cpu.retainAllLayers = true
        let cpuState = QwenCPUModel.DecodeState()
        var cpuLogits: [Float] = []
        for t in tokens { cpuLogits = try cpu.step([t], state: cpuState) }

        // Reference: the decode loop, token at a time (never layer-major).
        let sequentialGPU = try QwenMetalModel(modelDir: dir)
        let sequentialState = QwenCPUModel.DecodeState()
        var sequentialLogits: [Float] = []
        for t in tokens { sequentialLogits = try sequentialGPU.step([t], state: sequentialState) }

        let layerMajorGPU = try QwenMetalModel(modelDir: dir)
        layerMajorGPU.prefillMode = .layerMajor(chunkTokens: chunkTokens)
        let layerMajorState = QwenCPUModel.DecodeState()
        let layerMajorLogits = try layerMajorGPU.step(tokens, state: layerMajorState)
        let metrics = layerMajorGPU.lastStepMetrics

        // Correctness bar: same logits, same greedy pick, same KV state.
        let cpuDiff = MetalModelTests.maxAbsDiff(cpuLogits, layerMajorLogits)
        #expect(cpuDiff < 2e-3, "\(label): CPU vs layer-major maxAbsDiff \(cpuDiff)")
        let seqDiff = MetalModelTests.maxAbsDiff(sequentialLogits, layerMajorLogits)
        #expect(seqDiff < 2e-3, "\(label): sequential vs layer-major maxAbsDiff \(seqDiff)")
        MetalModelTests.expectMatchingKV(
            sequentialState, layerMajorState, label: "\(label) prompt")
        #expect(argmax(cpuLogits) == argmax(layerMajorLogits), "\(label): greedy diverged")

        // S1a property preserved: one LM head, intermediates elided.
        #expect(metrics.tokensProcessed == tokens.count, "\(label): token count")
        #expect(metrics.logitProjections == 1, "\(label): LM-head count")
        #expect(metrics.avoidedLogitProjections == tokens.count - 1, "\(label): elision count")
        MetalModelTests.expectInstrumentation(metrics, tokens: tokens.count, label: label)

        // S1b baselines, pinned exactly: buffers per chunk, dispatches per
        // token, and the chunk-shaped phase timeline.
        let chunks = baseline.chunkCount(tokens: tokens.count, chunkTokens: chunkTokens)
        #expect(metrics.commandBuffersCommitted
                == baseline.commandBuffers(tokens: tokens.count, chunkTokens: chunkTokens),
                "\(label): layer-major command-buffer baseline changed")
        #expect(metrics.computeDispatchesEncoded == baseline.dispatches(tokens: tokens.count),
                "\(label): layer-major dispatch baseline changed")
        MetalModelTests.expectPhaseTimeline(
            metrics,
            expectedPhases: MetalModelTests.expectedTimelinePhases(
                config: layerMajorGPU.config, tokens: chunks),
            label: label
        )

        // Continuation decode after the layer-major prompt: same next logits
        // as the sequential state, and the decode baselines untouched.
        let continuation = 11
        let seqCont = try sequentialGPU.step([continuation], state: sequentialState)
        let lmCont = try layerMajorGPU.step([continuation], state: layerMajorState)
        let contDiff = MetalModelTests.maxAbsDiff(seqCont, lmCont)
        #expect(contDiff < 2e-3, "\(label): continuation maxAbsDiff \(contDiff)")
        #expect(argmax(seqCont) == argmax(lmCont), "\(label): continuation greedy diverged")
        MetalModelTests.expectMatchingKV(
            sequentialState, layerMajorState, label: "\(label) continuation")
        MetalModelTests.expectFastPathBaseline(
            layerMajorGPU.lastStepMetrics, tokens: 1,
            baseline: decodeBaseline, label: "\(label) continuation"
        )
    }

    @Test func layerMajorMatchesSequentialOnQuantizedTiny() throws {
        try Self.compare("tiny-model-q4", chunkTokens: 32,
                         baseline: Self.q4Baseline, decodeBaseline: MetalModelTests.q4Baseline)
    }

    @Test func layerMajorMatchesSequentialOnQwen35Tiny() throws {
        try Self.compare("tiny-model-q35", chunkTokens: 32,
                         baseline: Self.q35Baseline, decodeBaseline: MetalModelTests.q35Baseline)
    }

    /// A prompt longer than the chunk crosses chunk boundaries with the same
    /// math: chunks of 2/2/1 must reproduce the sequential state exactly.
    @Test func chunkedPrefillSpansThePrompt() throws {
        try Self.compare("tiny-model-q4", chunkTokens: 2,
                         baseline: Self.q4Baseline, decodeBaseline: MetalModelTests.q4Baseline)
    }

    /// chunkTokens=1 degenerates to token order — the boundary case where
    /// layer-major and token-major visit (token, layer) identically.
    @Test func singleTokenChunksDegenerateToTokenOrder() throws {
        try Self.compare("tiny-model-q4", chunkTokens: 1,
                         baseline: Self.q4Baseline, decodeBaseline: MetalModelTests.q4Baseline)
    }

    // MARK: - Expert unions vs the S1b-a oracle

    /// The layer-major path must route every (token, layer) exactly as the
    /// token loop does, and fetch per layer exactly the union the
    /// PrefillExpertUnionPlan oracle computes from those routes — per chunk.
    static func expectUnionsMatchPlan(_ modelName: String, chunkTokens: Int) throws {
        let dir = fixturesDir.appendingPathComponent(modelName)
        let label = "\(modelName) chunk=\(chunkTokens)"
        let tokens = [1, 5, 9, 42, 7]

        // Oracle input: token-major routes recorded from the decode loop.
        let loopModel = try QwenMetalModel(modelDir: dir)
        let layerCount = loopModel.config.numHiddenLayers
        var loopFlat: [(layer: Int, experts: [Int])] = []
        loopModel.routedExpertObserver = { loopFlat.append(($0, $1)) }
        let loopState = QwenCPUModel.DecodeState()
        for t in tokens { _ = try loopModel.step([t], state: loopState) }
        loopModel.routedExpertObserver = nil
        #expect(loopFlat.count == tokens.count * layerCount, "\(label): observation shape")
        var loopRoutes: [[[Int]]] = []
        for t in 0..<tokens.count {
            var perLayer: [[Int]] = []
            for l in 0..<layerCount {
                let entry = loopFlat[t * layerCount + l]
                #expect(entry.layer == l, "\(label): loop layer order")
                perLayer.append(entry.experts)
            }
            loopRoutes.append(perLayer)
        }

        // The layer-major prefill records what it routed and what it fetched.
        let model = try QwenMetalModel(modelDir: dir)
        model.prefillMode = .layerMajor(chunkTokens: chunkTokens)
        var routed: [(layer: Int, experts: [Int])] = []
        var unions: [(layer: Int, experts: [Int])] = []
        model.routedExpertObserver = { routed.append(($0, $1)) }
        model.prefillExpertUnionObserver = { unions.append(($0, $1)) }
        let state = QwenCPUModel.DecodeState()
        _ = try model.step(tokens, state: state)
        model.routedExpertObserver = nil
        model.prefillExpertUnionObserver = nil

        // Expected order: per chunk, per layer — every token's route (in
        // token order), then that layer's planned union.
        var expectedRouted: [(Int, [Int])] = []
        var expectedUnions: [(Int, [Int])] = []
        var start = 0
        while start < tokens.count {
            let end = min(start + chunkTokens, tokens.count)
            let chunkPlan = try PrefillExpertUnionPlan(
                tokenRoutes: Array(loopRoutes[start..<end]),
                expertCount: model.config.numExperts
            )
            for l in 0..<layerCount {
                for t in start..<end { expectedRouted.append((l, loopRoutes[t][l])) }
                expectedUnions.append((l, chunkPlan.layers[l].experts))
            }
            start = end
        }
        #expect(routed.map(\.layer) == expectedRouted.map(\.0),
                "\(label): layer-major routing order diverged")
        #expect(routed.map(\.experts) == expectedRouted.map(\.1),
                "\(label): layer-major routing diverged from the token loop")
        #expect(unions.map(\.layer) == expectedUnions.map(\.0),
                "\(label): union layer order diverged")
        #expect(unions.map(\.experts) == expectedUnions.map(\.1),
                "\(label): fetched unions diverge from the oracle plan")

        // Single chunk: the fetched unions are exactly the whole-prompt plan
        // replayed layer-major (the S1b-a equivalence pattern, now consumed
        // by the real schedule).
        if chunkTokens >= tokens.count {
            let plan = try PrefillExpertUnionPlan(
                tokenRoutes: loopRoutes, expertCount: model.config.numExperts)
            var replayed: [(Int, [Int])] = []
            plan.replayLayerMajor { replayed.append(($0, $1)) }
            #expect(unions.map(\.layer) == replayed.map(\.0)
                    && unions.map(\.experts) == replayed.map(\.1),
                    "\(label): schedule does not consume the plan verbatim")
        }
    }

    @Test func unionsMatchThePlanOnQuantizedTiny() throws {
        try Self.expectUnionsMatchPlan("tiny-model-q4", chunkTokens: 32)
    }

    @Test func unionsMatchThePlanOnQwen35Tiny() throws {
        try Self.expectUnionsMatchPlan("tiny-model-q35", chunkTokens: 32)
    }

    @Test func chunkedUnionsMatchPerChunkPlans() throws {
        try Self.expectUnionsMatchPlan("tiny-model-q4", chunkTokens: 2)
    }

    // MARK: - Streaming (qpack) path

    /// On the streaming container the layer-major prefill must stay exact and
    /// its expert-cache traffic must be exactly one union fetch per layer:
    /// the cache sees sum(union sizes) requests for the whole prompt, not
    /// tokens x top-k.
    @Test func layerMajorQpackStreamingMatchesSequential() throws {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-q4-s1b-\(UUID().uuidString).qpack")
        defer { try? FileManager.default.removeItem(at: out) }
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()

        let tokens = [1, 5, 9, 42, 7, 99]
        let sequentialGPU = try QwenMetalModel(modelDir: out, cacheBudgetGB: 0.05)
        let layerMajorGPU = try QwenMetalModel(modelDir: out, cacheBudgetGB: 0.05)
        #expect(sequentialGPU.expertCache != nil, "qpack mode not engaged")
        #expect(layerMajorGPU.expertCache != nil, "qpack mode not engaged")

        let sequentialState = QwenCPUModel.DecodeState()
        var sequentialLogits: [Float] = []
        for t in tokens { sequentialLogits = try sequentialGPU.step([t], state: sequentialState) }

        var unionSizes: [Int] = []
        layerMajorGPU.prefillExpertUnionObserver = { unionSizes.append($1.count) }
        let cache = layerMajorGPU.expertCache!
        let requestsBefore = cache.hits + cache.misses
        let layerMajorState = QwenCPUModel.DecodeState()
        let layerMajorLogits = try layerMajorGPU.step(tokens, state: layerMajorState)
        layerMajorGPU.prefillExpertUnionObserver = nil
        let requests = cache.hits + cache.misses - requestsBefore

        let diff = MetalModelTests.maxAbsDiff(sequentialLogits, layerMajorLogits)
        #expect(diff < 2e-3, "qpack layer-major maxAbsDiff \(diff)")
        MetalModelTests.expectMatchingKV(
            sequentialState, layerMajorState, label: "qpack layer-major prompt")
        #expect(layerMajorGPU.lastStepMetrics.logitProjections == 1)
        #expect(layerMajorGPU.lastStepMetrics.commandBuffersCommitted
                == Self.q4Baseline.commandBuffers(tokens: tokens.count, chunkTokens: 32),
                "qpack layer-major command-buffer baseline changed")
        #expect(layerMajorGPU.lastStepMetrics.computeDispatchesEncoded
                == Self.q4Baseline.dispatches(tokens: tokens.count),
                "qpack layer-major dispatch baseline changed")

        #expect(unionSizes.count == layerMajorGPU.config.numHiddenLayers,
                "one union fetch per layer expected")
        #expect(requests == unionSizes.reduce(0, +),
                "expert-cache traffic is not one union fetch per layer")
        let tokenMajorRequests = tokens.count
            * layerMajorGPU.config.numHiddenLayers
            * layerMajorGPU.config.numExpertsPerTok
        #expect(requests <= tokenMajorRequests,
                "union fetches exceed token-major traffic")

        let continuation = 11
        let seqCont = try sequentialGPU.step([continuation], state: sequentialState)
        let lmCont = try layerMajorGPU.step([continuation], state: layerMajorState)
        let contDiff = MetalModelTests.maxAbsDiff(seqCont, lmCont)
        #expect(contDiff < 2e-3, "qpack layer-major continuation maxAbsDiff \(contDiff)")
        MetalModelTests.expectMatchingKV(
            sequentialState, layerMajorState, label: "qpack layer-major continuation")
    }
}
