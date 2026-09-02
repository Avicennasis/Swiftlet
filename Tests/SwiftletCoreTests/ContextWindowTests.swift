import Testing
@testable import SwiftletCore

@Suite struct ContextWindowTests {
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
}
