import Testing
@testable import SwiftletCore

/// A multi-byte character split across tokens must never leak a U+FFFD into
/// the stream, and the emitted prefix must stay valid once the character
/// completes (the observed failure: one printed "�" silenced the whole rest
/// of the turn because `hasPrefix` never matched again).
@Suite struct StreamingTextTests {
    @Test func plainGrowthEmitsDelta() {
        let r = StreamingText.delta(printed: "Hallo", decoded: "Hallo Welt")
        #expect(r?.delta == " Welt" && r?.printed == "Hallo Welt")
    }

    @Test func incompleteCharacterIsHeldBack() {
        // Emoji first byte(s) decoded -> trailing U+FFFD is not emitted.
        let r = StreamingText.delta(printed: "Danke!", decoded: "Danke! \u{FFFD}")
        #expect(r?.delta == " " && r?.printed == "Danke! ")
        // Nothing new while the character is still incomplete.
        #expect(StreamingText.delta(printed: "Danke! ", decoded: "Danke! \u{FFFD}") == nil)
    }

    @Test func completedCharacterEmitsCleanly() {
        // Next token completes the emoji: the stream continues seamlessly.
        let r = StreamingText.delta(printed: "Danke! ", decoded: "Danke! 😊 gerne")
        #expect(r?.delta == "😊 gerne")
    }

    @Test func fffdMidStringIsPreserved() {
        // Only a TRAILING replacement run is provisional.
        let r = StreamingText.delta(printed: "", decoded: "a\u{FFFD}b")
        #expect(r?.delta == "a\u{FFFD}b")
    }

    @Test func divergedPrintedStateDoesNotEmitGarbage() {
        // Legacy failure state (a "�" was already printed): no crash, no
        // bogus delta — the turn-final flush is responsible for recovery.
        #expect(StreamingText.delta(printed: "Hi \u{FFFD}", decoded: "Hi 😊") == nil)
    }

    @Test func finalFlushEmitsTrailingReplacement() {
        // EOS right after an incomplete character: final text is emitted
        // verbatim, replacement char included (it can never complete now).
        #expect(StreamingText.finalDelta(printed: "Ende ", decoded: "Ende \u{FFFD}") == "\u{FFFD}")
        #expect(StreamingText.finalDelta(printed: "gleich", decoded: "gleich") == nil)
    }

    @Test func multiTokenEmojiSequence() {
        // Simulated 4-byte emoji arriving over three decode steps.
        var printed = ""
        var out = ""
        for decoded in ["Wald", "Wald \u{FFFD}", "Wald \u{FFFD}\u{FFFD}", "Wald 🌲!"] {
            if let (d, p) = StreamingText.delta(printed: printed, decoded: decoded) {
                out += d
                printed = p
            }
        }
        #expect(out == "Wald 🌲!")
    }

    // MARK: Grapheme-extension (issue #15)
    //
    // A later token appends a combining scalar to a character that already
    // decoded cleanly, so no U+FFFD is ever involved. Under `Character`
    // comparison the grown cluster is not a prefix of itself, which silenced
    // the whole rest of the turn. Scalar comparison keeps it a valid prefix.

    @Test func variationSelectorExtendsEmittedCharacter() {
        // ✍ (U+270D) decoded first; the next token adds U+FE0F -> ✍️.
        let r = StreamingText.delta(printed: "\u{270D}", decoded: "\u{270D}\u{FE0F} Writing")
        #expect(r?.delta == "\u{FE0F} Writing")
        #expect(r?.printed == "\u{270D}\u{FE0F} Writing")
    }

    @Test func skinToneModifierExtendsEmittedCharacter() {
        // 👍 then a skin-tone modifier (U+1F3FD): the base is already on screen,
        // so only the modifier and what follows is the new delta.
        let r = StreamingText.delta(printed: "👍", decoded: "👍\u{1F3FD} nice")
        #expect(r?.delta == "\u{1F3FD} nice")
    }

    @Test func zwjSequenceGrowsWithoutSilencing() {
        // A ZWJ family emoji arriving scalar group by scalar group: every step
        // must keep emitting rather than going quiet, and the concatenation
        // must equal the final text.
        var printed = ""
        var out = ""
        let steps = [
            "family: \u{1F468}",                                   // 👨
            "family: \u{1F468}\u{200D}\u{1F469}",                  // 👨‍👩
            "family: \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}", // 👨‍👩‍👧
            "family: \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467} done",
        ]
        for decoded in steps {
            if let (d, p) = StreamingText.delta(printed: printed, decoded: decoded) {
                out += d
                printed = p
            }
        }
        #expect(out == steps.last)
    }

    @Test func combinerAfterHeldBackReplacementStillResolves() {
        // The two hazards back to back: an incomplete tail (U+FFFD) resolves
        // into a base character, and the following token extends it. Neither
        // step may go silent.
        var printed = ""
        var out = ""
        for decoded in ["ok \u{FFFD}", "ok \u{270D}", "ok \u{270D}\u{FE0F}!"] {
            if let (d, p) = StreamingText.delta(printed: printed, decoded: decoded) {
                out += d
                printed = p
            }
        }
        #expect(out == "ok \u{270D}\u{FE0F}!")
    }
}
