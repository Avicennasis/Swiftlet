import Foundation
import Testing
@testable import SwiftletCore

@Suite struct ContextWindowTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    @Test func clampsGenerationToRemainingCapacity() throws {
        let window = try ContextWindow(maximumTokens: 100)

        #expect(try window.admittedMaxNew(
            processedTokens: 0, incomingTokens: 90, requestedMaxNew: 20
        ) == 10)
        #expect(try window.admittedMaxNew(
            processedTokens: 40, incomingTokens: 20, requestedMaxNew: 12
        ) == 12)
    }

    @Test func rejectsPromptWithNoGenerationRoom() throws {
        let window = try ContextWindow(maximumTokens: 100)

        #expect(throws: ContextWindowError.self) {
            _ = try window.admittedMaxNew(
                processedTokens: 40, incomingTokens: 60, requestedMaxNew: 1
            )
        }
        // A caller explicitly requesting prefill only may consume the window.
        #expect(try window.admittedMaxNew(
            processedTokens: 40, incomingTokens: 60, requestedMaxNew: 0
        ) == 0)
    }

    @Test func rejectsOversizedStepBeforeOverflow() throws {
        let window = try ContextWindow(maximumTokens: .max)

        #expect(throws: ContextWindowError.self) {
            try window.validateStep(processedTokens: Int.max - 1, incomingTokens: 2)
        }
        #expect(throws: ContextWindowError.self) {
            _ = try window.admittedMaxNew(
                processedTokens: 0, incomingTokens: 1, requestedMaxNew: -1
            )
        }
    }

    @Test func rejectsEmptyModelStep() throws {
        let window = try ContextWindow(maximumTokens: 100)
        #expect(throws: ContextWindowError.self) {
            try window.validateStep(processedTokens: 0, incomingTokens: 0)
        }
    }

    @Test func realModelStepsRejectBeforeStateMutation() throws {
        let modelDir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let oversized = [Int](repeating: 1, count: 513)

        let cpu = try QwenCPUModel(modelDir: modelDir)
        #expect(cpu.config.maxPositionEmbeddings == 512)
        let cpuState = QwenCPUModel.DecodeState()
        #expect(throws: ContextWindowError.self) {
            _ = try cpu.step(oversized, state: cpuState)
        }
        #expect(cpuState.position == 0)

        let metal = try QwenMetalModel(modelDir: modelDir)
        let metalState = QwenCPUModel.DecodeState()
        #expect(throws: ContextWindowError.self) {
            _ = try metal.step(oversized, state: metalState)
        }
        #expect(metalState.position == 0)
        #expect(metal.lastStepMetrics.tokensProcessed == 0)
        #expect(!metal.lastStepMetrics.completedWithoutThrow)
    }
}
