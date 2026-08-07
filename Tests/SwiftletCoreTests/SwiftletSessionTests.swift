import Foundation
import Testing
@testable import SwiftletCore

@Suite struct SwiftletSessionTests {
    /// Issue #9: a token boundary that splits a multi-byte character decodes
    /// to a trailing U+FFFD, which must be held back, never emitted or stored
    /// as printed text (it poisons the prefix check and the EOS gate).
    @Test func trimIncompleteUTF8HoldsUnstableTail() {
        #expect(SwiftletSession.trimIncompleteUTF8("hello") == "hello")
        #expect(SwiftletSession.trimIncompleteUTF8("") == "")
        #expect(SwiftletSession.trimIncompleteUTF8("~10\u{FFFD}") == "~10")
        // A partial 4-byte sequence can render as several replacement chars.
        #expect(SwiftletSession.trimIncompleteUTF8("x\u{FFFD}\u{FFFD}\u{FFFD}") == "x")
        // Interior replacement chars are genuine content and stay.
        #expect(SwiftletSession.trimIncompleteUTF8("a\u{FFFD}b") == "a\u{FFFD}b")
        #expect(SwiftletSession.trimIncompleteUTF8("a\u{FFFD}b\u{FFFD}") == "a\u{FFFD}b")
        // Completed characters pass through untouched.
        #expect(SwiftletSession.trimIncompleteUTF8("~10⁵⁰⁰ vacua") == "~10⁵⁰⁰ vacua")
    }
}
