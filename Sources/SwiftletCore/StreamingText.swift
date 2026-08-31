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

/// Incremental text-stop filtering for a cumulatively decoded token stream.
///
/// A suffix that could still become a stop string is held back instead of
/// being sent to the caller. This is what lets `"END"` match when `"E"`,
/// `"N"`, and `"D"` arrive in separate tokens without leaking `"EN"` first.
/// Matching and slicing use Unicode scalars for the same reason StreamingText
/// does: a later token may extend a previously complete grapheme cluster.
public struct StopSequenceFilter {
    public struct Update: Sendable {
        public let delta: String?
        public let matchedStop: String?
        /// False only when cumulative tokenizer output no longer has the text
        /// already exposed to the caller as a Unicode-scalar prefix.
        public let isConsistent: Bool

        public var didStop: Bool { matchedStop != nil }
    }

    private let stops: [(text: String, scalars: [Unicode.Scalar])]
    private var emitted: [Unicode.Scalar] = []
    private var stopped = false

    public init(stopSequences: [String]) {
        // Swift String equality is canonically equivalent, but stop matching
        // is deliberately scalar-exact. Keep precomposed and decomposed forms
        // as distinct entries so each can match its own wire representation.
        var seen = Set<[Unicode.Scalar]>()
        stops = stopSequences.compactMap { text in
            let scalars = Array(text.unicodeScalars)
            guard !scalars.isEmpty, seen.insert(scalars).inserted else { return nil }
            return (text, scalars)
        }
    }

    /// Text released to the caller so far. If a stop matched, this excludes
    /// both the stop itself and any text decoded after it in the same token.
    public var output: String {
        String(String.UnicodeScalarView(emitted))
    }

    /// Consumes the tokenizer's full cumulative decode for the current token.
    public mutating func consume(decoded: String) -> Update {
        guard !stopped else {
            return Update(delta: nil, matchedStop: nil, isConsistent: true)
        }
        let stable = Array(StreamingText.stablePrefix(decoded).unicodeScalars)

        // Tokenizer output is expected to grow by scalar prefix. If a custom
        // tokenizer violates that contract, fail closed instead of duplicating
        // or emitting text from an ambiguous offset.
        guard stable.starts(with: emitted) else {
            return Update(delta: nil, matchedStop: nil, isConsistent: false)
        }

        if let match = earliestMatch(in: stable) {
            let delta = append(stable[emitted.count..<match.index])
            stopped = true
            return Update(delta: delta, matchedStop: match.stop.text, isConsistent: true)
        }

        // Hold the longest suffix that is a proper prefix of any stop. A full
        // stop was handled above, so limiting to count-1 is sufficient.
        let uncommittedCount = stable.count - emitted.count
        var hold = 0
        for stop in stops where stop.scalars.count > 1 {
            let limit = min(stop.scalars.count - 1, uncommittedCount)
            guard limit > hold else { continue }
            for count in stride(from: limit, through: hold + 1, by: -1) {
                if stable.suffix(count).elementsEqual(stop.scalars.prefix(count)) {
                    hold = count
                    break
                }
            }
        }
        let safeEnd = stable.count - hold
        return Update(
            delta: append(stable[emitted.count..<safeEnd]),
            matchedStop: nil,
            isConsistent: true
        )
    }

    /// Releases a non-matching held prefix when EOS or the length limit ends
    /// the turn. For callers that also need to distinguish a final raw-text
    /// stop match, use `finishUpdate(decoded:)`.
    public mutating func finish(decoded: String) -> String? {
        finishUpdate(decoded: decoded).delta
    }

    /// Finalizes against the raw scalar stream. Unlike `consume`, this does
    /// not trim a trailing U+FFFD: at EOS/length it can no longer complete, and
    /// it may itself complete a caller-supplied stop sequence.
    public mutating func finishUpdate(decoded: String) -> Update {
        guard !stopped else {
            return Update(delta: nil, matchedStop: nil, isConsistent: true)
        }
        let scalars = Array(decoded.unicodeScalars)
        guard scalars.starts(with: emitted) else {
            return Update(delta: nil, matchedStop: nil, isConsistent: false)
        }
        if let match = earliestMatch(in: scalars) {
            let delta = append(scalars[emitted.count..<match.index])
            stopped = true
            return Update(delta: delta, matchedStop: match.stop.text, isConsistent: true)
        }
        return Update(
            delta: append(scalars[emitted.count..<scalars.count]),
            matchedStop: nil,
            isConsistent: true
        )
    }

    private func earliestMatch(
        in scalars: [Unicode.Scalar]
    ) -> (index: Int, stop: (text: String, scalars: [Unicode.Scalar]))? {
        guard !stops.isEmpty else { return nil }
        for index in emitted.count...scalars.count {
            for stop in stops where index + stop.scalars.count <= scalars.count {
                if scalars[index..<(index + stop.scalars.count)].elementsEqual(stop.scalars) {
                    return (index, stop)
                }
            }
        }
        return nil
    }

    private mutating func append<C: Collection>(_ scalars: C) -> String?
    where C.Element == Unicode.Scalar {
        guard !scalars.isEmpty else { return nil }
        let delta = String(String.UnicodeScalarView(Array(scalars)))
        emitted.append(contentsOf: scalars)
        return delta
    }
}
