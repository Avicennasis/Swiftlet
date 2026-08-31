import Foundation
import Testing
@testable import SwiftletCore

@Suite struct TextGeneratorTests {
    private final class CancellationProbeModel: InferenceModel, @unchecked Sendable {
        enum ProbeError: Swift.Error { case timedOutWaitingForCancellation }

        let config: QwenConfig
        let modelDir: URL
        let waitForCancellation: Bool
        private let lock = NSLock()
        private var _legacyCalls = 0
        private var _cancellableCalls = 0
        private var _observedCancellation = false

        init(modelDir: URL, waitForCancellation: Bool) throws {
            self.modelDir = modelDir
            self.waitForCancellation = waitForCancellation
            config = try QwenConfig(url: modelDir.appendingPathComponent("config.json"))
        }

        var legacyCalls: Int { locked { _legacyCalls } }
        var cancellableCalls: Int { locked { _cancellableCalls } }
        var observedCancellation: Bool { locked { _observedCancellation } }

        func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float] {
            locked { _legacyCalls += 1 }
            return [Float](repeating: 0, count: config.vocabSize)
        }

        func step(
            _ tokens: [Int],
            state: QwenCPUModel.DecodeState,
            shouldCancel: () -> Bool
        ) throws -> [Float] {
            locked { _cancellableCalls += 1 }
            if waitForCancellation {
                let deadline = Date().addingTimeInterval(2)
                while !shouldCancel() {
                    guard Date() < deadline else {
                        throw ProbeError.timedOutWaitingForCancellation
                    }
                    Thread.sleep(forTimeInterval: 0.001)
                }
            }
            if shouldCancel() {
                locked { _observedCancellation = true }
                throw GenerationInterruption.cancelled
            }
            return [Float](repeating: 0, count: config.vocabSize)
        }

        private func locked<T>(_ body: () -> T) -> T {
            lock.lock()
            defer { lock.unlock() }
            return body()
        }
    }

    private final class SingleFlightProbeModel: InferenceModel, @unchecked Sendable {
        enum ProbeError: Swift.Error { case timedOutWaitingForRelease }

        let config: QwenConfig
        let modelDir: URL
        private let lock = NSLock()
        private let releaseFirstCall = DispatchSemaphore(value: 0)
        private var _callCount = 0
        private var _activeCalls = 0
        private var _maxActiveCalls = 0
        private var _firstCallEntered = false

        init(modelDir: URL) throws {
            self.modelDir = modelDir
            config = try QwenConfig(url: modelDir.appendingPathComponent("config.json"))
        }

        var callCount: Int { locked { _callCount } }
        var maxActiveCalls: Int { locked { _maxActiveCalls } }
        var firstCallEntered: Bool { locked { _firstCallEntered } }

        func releaseFirst() {
            releaseFirstCall.signal()
        }

        func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float] {
            try runStep(shouldCancel: { false })
        }

        func step(
            _ tokens: [Int],
            state: QwenCPUModel.DecodeState,
            shouldCancel: () -> Bool
        ) throws -> [Float] {
            try runStep(shouldCancel: shouldCancel)
        }

        private func runStep(shouldCancel: () -> Bool) throws -> [Float] {
            let blocks = locked { () -> Bool in
                _callCount += 1
                _activeCalls += 1
                _maxActiveCalls = max(_maxActiveCalls, _activeCalls)
                if _callCount == 1 {
                    _firstCallEntered = true
                    return true
                }
                return false
            }
            defer { locked { _activeCalls -= 1 } }

            if blocks,
               releaseFirstCall.wait(timeout: .now() + .seconds(2)) == .timedOut {
                throw ProbeError.timedOutWaitingForRelease
            }
            if shouldCancel() { throw GenerationInterruption.cancelled }
            return [Float](repeating: 0, count: config.vocabSize)
        }

        private func locked<T>(_ body: () -> T) -> T {
            lock.lock()
            defer { lock.unlock() }
            return body()
        }
    }

    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    @Test func tokenStreamMatchesDirectGenerate() async throws {
        let model = try QwenCPUModel(modelDir: Self.fixturesDir.appendingPathComponent("tiny-model-q4"))
        model.retainAllLayers = true
        let generator = TextGenerator(model: model)

        var direct: [Int] = []
        try generator.generate(promptIds: [1, 5, 9], maxNew: 6) { direct.append($0); return true }

        var streamed: [Int] = []
        for try await token in generator.tokenStream(promptIds: [1, 5, 9], maxNew: 6) {
            streamed.append(token)
        }
        #expect(streamed == direct)
        #expect(streamed.count == 6)
    }

    @Test func generateRejectsPreCancelledWorkBeforeModelStep() throws {
        let model = try CancellationProbeModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model"),
            waitForCancellation: false
        )
        let generator = TextGenerator(model: model)
        let cancellation = GenerationCancellation()
        cancellation.cancel()

        #expect(throws: GenerationInterruption.self) {
            _ = try generator.generate(
                promptIds: [1, 5, 9], maxNew: 2, cancellation: cancellation
            ) { _ in true }
        }
        #expect(model.cancellableCalls == 0)
        #expect(model.legacyCalls == 0)
        #expect(!model.observedCancellation)
    }

    @Test func concurrentTokenStreamsNeverOverlapModelCalls() async throws {
        let model = try SingleFlightProbeModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model")
        )
        let generator = TextGenerator(model: model)

        let firstStream = generator.tokenStream(promptIds: [1], maxNew: 0)
        let first = Task {
            for try await _ in firstStream {}
        }
        let deadline = Date().addingTimeInterval(1)
        while !model.firstCallEntered, Date() < deadline {
            try await Task<Never, Never>.sleep(nanoseconds: 1_000_000)
        }
        #expect(model.firstCallEntered)

        let secondStream = generator.tokenStream(promptIds: [2], maxNew: 0)
        let second = Task {
            for try await _ in secondStream {}
        }
        try await Task<Never, Never>.sleep(nanoseconds: 20_000_000)

        #expect(model.callCount == 1)
        #expect(model.maxActiveCalls == 1)
        model.releaseFirst()
        try await first.value
        try await second.value

        #expect(model.callCount == 2)
        #expect(model.maxActiveCalls == 1)
    }

    @Test func queuedCancelledTokenStreamNeverTouchesModel() async throws {
        let model = try SingleFlightProbeModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model")
        )
        let generator = TextGenerator(model: model)

        let firstStream = generator.tokenStream(promptIds: [1], maxNew: 0)
        let first = Task {
            for try await _ in firstStream {}
        }
        let deadline = Date().addingTimeInterval(1)
        while !model.firstCallEntered, Date() < deadline {
            try await Task<Never, Never>.sleep(nanoseconds: 1_000_000)
        }
        #expect(model.firstCallEntered)

        let cancellation = GenerationCancellation()
        let secondStream = generator.tokenStream(
            promptIds: [2], maxNew: 0, cancellation: cancellation
        )
        let second = Task {
            for try await _ in secondStream {}
        }
        cancellation.cancel()
        model.releaseFirst()

        try await first.value
        try await second.value
        #expect(model.callCount == 1)
        #expect(model.maxActiveCalls == 1)
    }

    @Test func tokenStreamOwnerCancelsBlockedPrefill() async throws {
        let model = try CancellationProbeModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model"),
            waitForCancellation: true
        )
        let generator = TextGenerator(model: model)
        let cancellation = GenerationCancellation()
        DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(20)) {
            cancellation.cancel()
        }

        var streamed: [Int] = []
        for try await token in generator.tokenStream(
            promptIds: [1, 5, 9], maxNew: 2, cancellation: cancellation
        ) {
            streamed.append(token)
        }

        #expect(streamed.isEmpty)
        #expect(model.cancellableCalls == 1)
        #expect(model.legacyCalls == 0)
        #expect(model.observedCancellation)
    }

    @Test func cancellingConsumerTaskCancelsBlockedPrefill() async throws {
        let model = try CancellationProbeModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model"),
            waitForCancellation: true
        )
        let generator = TextGenerator(model: model)
        let stream = generator.tokenStream(promptIds: [1, 5, 9], maxNew: 2)
        let consumer = Task {
            do {
                for try await _ in stream {}
            } catch {
                // Cancellation is normal termination for this API.
            }
        }

        try await Task<Never, Never>.sleep(nanoseconds: 20_000_000)
        consumer.cancel()
        await consumer.value
        let deadline = Date().addingTimeInterval(1)
        while !model.observedCancellation, Date() < deadline {
            try await Task<Never, Never>.sleep(nanoseconds: 1_000_000)
        }

        #expect(model.cancellableCalls == 1)
        #expect(model.legacyCalls == 0)
        #expect(model.observedCancellation)
    }
}
