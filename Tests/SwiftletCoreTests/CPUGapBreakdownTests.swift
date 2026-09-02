import Foundation
import Testing
@testable import SwiftletCore

/// S3c: the CPU-gap sub-attribution must be conservative (never claim more
/// than the S3a/S3b residual), rebuilt per step, and count exactly the work
/// the schedule performs — so a change to the schedule shows up here before
/// it shows up as a throughput surprise.
@Suite struct CPUGapBreakdownTests {
    static let fixturesDir = MetalModelTests.fixturesDir

    static func expectConservation(
        _ m: QwenMetalModel.StepMetrics, tokens: Int, label: String
    ) {
        #expect(m.completedWithoutThrow, "\(label): step threw")
        let gap = m.cpuGap
        #expect(gap.embeddingSeconds >= 0, "\(label): negative embedding time")
        #expect(gap.routerSeconds >= 0, "\(label): negative router time")
        #expect(gap.expertFetchSeconds >= 0, "\(label): negative fetch time")
        #expect(gap.kvMirrorSeconds >= 0, "\(label): negative KV-mirror time")
        #expect(gap.commandBufferSetupSeconds >= 0, "\(label): negative setup time")
        #expect(gap.commitSeconds >= 0, "\(label): negative commit time")
        #expect(gap.logitsReadbackSeconds >= 0, "\(label): negative logits time")
        #expect(gap.expertFetchHits >= 0 && gap.expertFetchMisses >= 0,
                "\(label): negative fetch counts")
        // Every scope is disjoint from every wait and every encode window,
        // so the attributed sum can never exceed the residual (2 ms of
        // headroom for clock jitter across ~50 scope boundaries).
        #expect(gap.attributedSeconds <= m.cpuGapSeconds + 2e-3,
                "\(label): attributed \(gap.attributedSeconds) exceeds gap \(m.cpuGapSeconds)")
        #expect(m.cpuGapOtherSeconds >= -2e-3, "\(label): negative residual")
        #expect(abs(m.encodeSeconds + m.blockingWaitSeconds + m.cpuGapSeconds - m.stepWallSeconds)
                < 1e-6, "\(label): wall != encode + wait + gap")
        #expect(gap.embeddingLookups == tokens, "\(label): embedding lookups != tokens")
        // Scopes that always run must have taken real time — a scope that
        // reads zero on every step is a scope that was never entered.
        #expect(gap.embeddingSeconds > 0, "\(label): embedding scope never entered")
        #expect(gap.routerSeconds > 0, "\(label): router scope never entered")
        #expect(gap.commandBufferSetupSeconds > 0, "\(label): setup scope never entered")
        #expect(gap.commitSeconds > 0, "\(label): commit scope never entered")
        #expect(gap.logitsReadbackSeconds > 0, "\(label): logits scope never entered")
    }

    /// Raw checkpoint dir (no expert cache): every scope conserves, fetch
    /// counts are zero, and the counters rebuild per step.
    @Test func scopesConserveTheStepWallOnRawDir() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let model = try QwenMetalModel(modelDir: dir)
        #expect(model.lastStepMetrics.cpuGap == .zero, "fresh model reports a stale gap")
        let state = QwenCPUModel.DecodeState()

        _ = try model.step([1, 5, 9], state: state)
        let multi = model.lastStepMetrics
        Self.expectConservation(multi, tokens: 3, label: "raw multi")
        #expect(multi.cpuGap.expertFetchHits == 0 && multi.cpuGap.expertFetchMisses == 0,
                "raw dir has no expert cache to hit")
        #expect(multi.cpuGap.expertFetchSeconds == 0, "raw dir fetch scope entered")
        #expect(multi.cpuGap.kvMirrorSeconds > 0, "attention layers mirror KV")

        _ = try model.step([11], state: state)
        let single = model.lastStepMetrics
        Self.expectConservation(single, tokens: 1, label: "raw single")
        #expect(single.cpuGap.embeddingLookups == 1, "gap counters accumulated across steps")
    }

    /// The token-major prompt schedule (S1a) conserves too.
    @Test func scopesConserveOnTokenMajorPrompt() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q35")
        let model = try QwenMetalModel(modelDir: dir)
        model.prefillMode = .tokenMajor
        let state = QwenCPUModel.DecodeState()
        _ = try model.step([1, 5, 9, 42], state: state)
        Self.expectConservation(model.lastStepMetrics, tokens: 4, label: "token-major")
    }

    /// qpack: a single decode step fetches exactly layers x top-k experts, the
    /// counted hits/misses equal the cache's own delta, and the layer-major
    /// prompt fetches exactly the per-layer unions the plan predicts.
    @Test func qpackFetchCountsMatchTheSchedule() throws {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-q4-gap-\(UUID().uuidString).qpack")
        defer { try? FileManager.default.removeItem(at: out) }
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()

        let model = try QwenMetalModel(modelDir: out, cacheBudgetGB: 0.05)
        let cache = try #require(model.expertCache, "qpack mode not engaged")
        let cfg = model.config
        let state = QwenCPUModel.DecodeState()

        var unionTotal = 0
        model.prefillExpertUnionObserver = { _, union in unionTotal += union.count }
        let hits0 = cache.hits, misses0 = cache.misses
        _ = try model.step([1, 5, 9, 42, 7, 99], state: state)
        let prompt = model.lastStepMetrics
        Self.expectConservation(prompt, tokens: 6, label: "qpack prompt")
        #expect(prompt.cpuGap.expertFetchHits + prompt.cpuGap.expertFetchMisses == unionTotal,
                "prompt fetch count != sum of per-layer unions")
        #expect(prompt.cpuGap.expertFetchHits == cache.hits - hits0,
                "prompt hit count != cache delta")
        #expect(prompt.cpuGap.expertFetchMisses == cache.misses - misses0,
                "prompt miss count != cache delta")
        #expect(prompt.cpuGap.expertFetchSeconds > 0, "fetch scope never entered")

        let hits1 = cache.hits, misses1 = cache.misses
        _ = try model.step([3], state: state)
        let single = model.lastStepMetrics
        Self.expectConservation(single, tokens: 1, label: "qpack decode")
        #expect(single.cpuGap.expertFetchHits + single.cpuGap.expertFetchMisses
                == cfg.numHiddenLayers * cfg.numExpertsPerTok,
                "decode step fetch count != layers x top-k")
        #expect(single.cpuGap.expertFetchHits == cache.hits - hits1, "decode hit count != cache delta")
        #expect(single.cpuGap.expertFetchMisses == cache.misses - misses1,
                "decode miss count != cache delta")
    }
}
