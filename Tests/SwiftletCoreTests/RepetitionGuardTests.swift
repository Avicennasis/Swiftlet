import Testing
import Foundation
@testable import SwiftletCore

/// Regression tests for issue #13: temperature-0 (greedy) decode collapsing into
/// verbatim repetition loops on the quantized 35B container. The fix keeps a
/// deterministic loop guard active in greedy mode and adds a no-repeat-n-gram
/// block that mathematically forbids any verbatim n-gram from recurring.
@Suite struct RepetitionGuardTests {

    // MARK: - Greedy preset keeps loop protection

    @Test func greedyIsDeterministicButLoopProtected() {
        let g = SwiftletSession.GenerationOptions.greedy
        // A temperature-0 pick stays deterministic (no sampling randomness)...
        #expect(g.temperature == 0)
        // ...but the anti-loop guards must NOT be disabled the way they were
        // before the fix (both penalties were zeroed, leaving pure argmax).
        #expect(g.noRepeatNGram >= 2)
        #expect(g.frequencyPenalty > 0)
        // The flat presence penalty is intentionally off for greedy: it would
        // distort legitimate repetition in code.
        #expect(g.presencePenalty == 0)
    }

    // MARK: - No-repeat-n-gram guard

    @Test func bansTheTokenThatWouldCloseARepeatedNGram() {
        // History "1 2 3 1 2": the 3-gram starting "1 2" was once followed by
        // "3", so re-emitting 3 after the trailing "1 2" is banned.
        let banned = SwiftletSession.noRepeatNGramBanned(generated: [1, 2, 3, 1, 2], n: 3)
        #expect(banned == [3])
    }

    @Test func breaksAReportedStyleNumericLoop() {
        // The issue's collapse was a fixed numeric cycle repeating forever.
        // Model it as tokens 7,8,9 cycling: once "7 8" -> 9 has been seen, the
        // guard bans 9 after the next "7 8", so argmax must diverge.
        let banned = SwiftletSession.noRepeatNGramBanned(generated: [7, 8, 9, 7, 8], n: 3)
        #expect(banned.contains(9))
    }

    @Test func allowsNovelContinuations() {
        // No (n-1)-gram repeats, so nothing is banned: normal generation is
        // untouched by the guard.
        #expect(SwiftletSession.noRepeatNGramBanned(generated: [1, 2, 3, 4, 5], n: 3).isEmpty)
    }

    @Test func shortHistoryBansNothing() {
        #expect(SwiftletSession.noRepeatNGramBanned(generated: [], n: 3).isEmpty)
        #expect(SwiftletSession.noRepeatNGramBanned(generated: [1, 2], n: 3).isEmpty)
    }

    @Test func disabledWhenNIsBelowTwo() {
        #expect(SwiftletSession.noRepeatNGramBanned(generated: [1, 2, 1, 2], n: 0).isEmpty)
        #expect(SwiftletSession.noRepeatNGramBanned(generated: [1, 2, 1, 2], n: 1).isEmpty)
    }

    @Test func collectsEveryDistinctSuccessorOfARepeatedPrefix() {
        // Prefix "1 2" was followed by both 3 and 9 earlier; the trailing
        // "1 2" must have both banned so neither branch of the loop can close.
        let banned = SwiftletSession.noRepeatNGramBanned(generated: [1, 2, 3, 1, 2, 9, 1, 2], n: 3)
        #expect(banned == [3, 9])
    }
}
