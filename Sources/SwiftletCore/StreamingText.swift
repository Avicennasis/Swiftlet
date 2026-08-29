import Foundation

/// Streaming-safe text deltas over an incrementally decoded token sequence.
///
/// A BPE tokenizer breaks the stream in two ways that a grapheme-level
/// `hasPrefix` delta scheme cannot survive, both of which silence the rest of
/// the turn once they hit:
///
///  - It splits a character's UTF-8 bytes across tokens. The incomplete tail
///    decodes to a trailing U+FFFD that must not be emitted.
///  - A later token *extends* a character that already decoded cleanly. A
///    variation selector, skin-tone modifier, or ZWJ turns ✍ (U+270D) into
///    ✍️ (U+270D U+FE0F). Nothing is malformed, but the final grapheme cluster
///    changed identity, so `"✍️".hasPrefix("✍")` is false and the emitted
///    text is no longer a `Character`-prefix of the full decode.
///
/// Comparing on Unicode scalars fixes both. A trailing U+FFFD run is held back
/// until it completes, and an extended cluster stays a valid scalar-prefix, so
/// the combining scalar is emitted as its own delta and the consumer composes
/// it onto the character already on screen. The concatenation of every delta
/// always equals the final decoded text.
public enum StreamingText {
    /// `decoded`'s scalars with any trailing U+FFFD run removed (an incomplete
    /// multi-byte character mid-stream). U+FFFD elsewhere is preserved.
    public static func stablePrefix(_ decoded: String) -> String {
        var scalars = Array(decoded.unicodeScalars)
        while scalars.last == "\u{FFFD}" { scalars.removeLast() }
        return String(String.UnicodeScalarView(scalars))
    }

    /// The next emittable delta, or nil when nothing new can be emitted yet.
    /// On success, the caller should append `delta` to its output and replace
    /// its printed-state with `printed`. Prefix and slice are computed on
    /// Unicode scalars, so a character that a later token extends stays a
    /// valid prefix instead of silencing the stream.
    public static func delta(printed: String, decoded: String) -> (delta: String, printed: String)? {
        let stable = stablePrefix(decoded)
        let printedScalars = Array(printed.unicodeScalars)
        let stableScalars = Array(stable.unicodeScalars)
        guard stableScalars.count > printedScalars.count,
              stableScalars.starts(with: printedScalars) else { return nil }
        let deltaScalars = stableScalars[printedScalars.count...]
        return (String(String.UnicodeScalarView(deltaScalars)), stable)
    }

    /// Delta for the END of a turn: the sequence is final, so a trailing
    /// U+FFFD can never complete and is emitted as-is rather than held back.
    public static func finalDelta(printed: String, decoded: String) -> String? {
        let printedScalars = Array(printed.unicodeScalars)
        let decodedScalars = Array(decoded.unicodeScalars)
        guard decodedScalars.count > printedScalars.count,
              decodedScalars.starts(with: printedScalars) else { return nil }
        return String(String.UnicodeScalarView(decodedScalars[printedScalars.count...]))
    }
}
