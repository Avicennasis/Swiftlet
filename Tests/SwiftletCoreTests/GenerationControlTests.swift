import Foundation
import Testing
@testable import SwiftletCore

@Suite struct GenerationControlTests {
    private static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    private final class LegacyInferenceModel: InferenceModel {
        let config: QwenConfig
        let modelDir: URL
        var stepCalls = 0

        init(modelDir: URL) throws {
            self.modelDir = modelDir
            config = try QwenConfig(url: modelDir.appendingPathComponent("config.json"))
        }

        func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float] {
            stepCalls += 1
            return [0, 1, 2, 3]
        }
    }

    @Test func cancellationIsOneWayAndObservableAcrossReferences() {
        let cancellation = GenerationCancellation()
        let shared = cancellation
        #expect(!shared.isCancelled)
        cancellation.cancel()
        #expect(shared.isCancelled)
        cancellation.cancel()
        #expect(shared.isCancelled)
    }

    @Test func finishReasonsHaveStableWireValues() {
        #expect(GenerationFinishReason.stop.rawValue == "stop")
        #expect(GenerationFinishReason.length.rawValue == "length")
        #expect(GenerationFinishReason.cancelled.rawValue == "cancelled")
    }

    @Test func legacyInferenceConformerGetsCancellableDefault() throws {
        let model = try LegacyInferenceModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model")
        )
        let state = QwenCPUModel.DecodeState()
        let erased: any InferenceModel = model

        let logits = try erased.step([1], state: state, shouldCancel: { false })

        #expect(logits == [0, 1, 2, 3])
        #expect(model.stepCalls == 1)
    }

    @Test func cancellableDefaultChecksBeforeAndAfterLegacyStep() throws {
        let model = try LegacyInferenceModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model")
        )
        let state = QwenCPUModel.DecodeState()
        let erased: any InferenceModel = model

        #expect(throws: GenerationInterruption.self) {
            _ = try erased.step([1], state: state, shouldCancel: { true })
        }
        #expect(model.stepCalls == 0)

        var checks = 0
        #expect(throws: GenerationInterruption.self) {
            _ = try erased.step([1], state: state, shouldCancel: {
                checks += 1
                return checks == 2
            })
        }
        #expect(model.stepCalls == 1)
        #expect(checks == 2)
    }

    @Test func cpuReferenceCancelsOnlyBetweenCompleteLayers() throws {
        let model = try QwenCPUModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model")
        )
        let interruptedState = QwenCPUModel.DecodeState()
        var checks = 0

        #expect(throws: GenerationInterruption.self) {
            _ = try model.step([1, 5, 9], state: interruptedState, shouldCancel: {
                checks += 1
                // Entry, before layer 0, then between layers 0 and 1.
                return checks == 3
            })
        }
        #expect(checks == 3)
        #expect(interruptedState.position == 0)

        // A caller recovers by discarding the partial state, not by trying to
        // continue it. A fresh state still completes normally.
        let freshState = QwenCPUModel.DecodeState()
        let logits = try model.step(
            [1, 5, 9], state: freshState, shouldCancel: { false }
        )
        #expect(freshState.position == 3)
        #expect(logits.count == model.config.vocabSize)
    }
}
