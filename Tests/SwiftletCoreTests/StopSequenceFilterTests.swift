import Testing
@testable import SwiftletCore

@Suite struct StopSequenceFilterTests {
    @Test func splitStopPrefixIsHeldAcrossTokens() {
        var filter = StopSequenceFilter(stopSequences: ["END"])

        let first = filter.consume(decoded: "hello E")
        #expect(first.delta == "hello ")
        #expect(!first.didStop)

        let second = filter.consume(decoded: "hello EN")
        #expect(second.delta == nil)
        #expect(!second.didStop)

        let third = filter.consume(decoded: "hello END ignored")
        #expect(third.delta == nil)
        #expect(third.matchedStop == "END")
        #expect(filter.output == "hello ")
    }

    @Test func textBeforeAndAfterStopInOneTokenIsFiltered() {
        var filter = StopSequenceFilter(stopSequences: ["<stop>"])
        let update = filter.consume(decoded: "answer<stop>not part of the answer")
        #expect(update.delta == "answer")
        #expect(update.matchedStop == "<stop>")
        #expect(filter.output == "answer")
        #expect(filter.finish(decoded: "answer<stop>not part of the answer") == nil)
    }

    @Test func earliestOfSeveralStopsWins() {
        var filter = StopSequenceFilter(stopSequences: ["SECOND", "FIRST"])
        let update = filter.consume(decoded: "a FIRST b SECOND")
        #expect(update.delta == "a ")
        #expect(update.matchedStop == "FIRST")
    }

    @Test func overlappingStopsDoNotLeakTheirSharedPrefix() {
        var filter = StopSequenceFilter(stopSequences: ["ABC", "AB"])
        #expect(filter.consume(decoded: "xA").delta == "x")
        let update = filter.consume(decoded: "xAB")
        #expect(update.matchedStop == "AB")
        #expect(filter.output == "x")
    }

    @Test func falsePrefixIsReleasedWhenItCanNoLongerMatch() {
        var filter = StopSequenceFilter(stopSequences: ["END"])
        #expect(filter.consume(decoded: "hello E").delta == "hello ")

        let update = filter.consume(decoded: "hello EX then")
        #expect(update.delta == "EX then")
        #expect(!update.didStop)
        #expect(filter.output == "hello EX then")
    }

    @Test func longestViablePrefixIsHeldAcrossOverlappingStops() {
        var filter = StopSequenceFilter(stopSequences: ["ABCD", "BCD"])
        #expect(filter.consume(decoded: "xABC").delta == "x")

        let update = filter.consume(decoded: "xABCE")
        #expect(update.delta == "ABCE")
        #expect(!update.didStop)
    }

    @Test func unicodeStopCanCrossScalarFragments() {
        var filter = StopSequenceFilter(stopSequences: ["🌲!"])
        #expect(filter.consume(decoded: "woods 🌲").delta == "woods ")
        let update = filter.consume(decoded: "woods 🌲! trailing")
        #expect(update.matchedStop == "🌲!")
        #expect(filter.output == "woods ")
    }

    @Test func lengthOrEOSFlushesNonMatchingPartialStop() {
        var filter = StopSequenceFilter(stopSequences: ["END"])
        #expect(filter.consume(decoded: "hello E").delta == "hello ")
        #expect(filter.finish(decoded: "hello E") == "E")
        #expect(filter.output == "hello E")
    }

    @Test func cancellationCanDropHeldTextWithoutFlushing() {
        var filter = StopSequenceFilter(stopSequences: ["END"])
        #expect(filter.consume(decoded: "visible E").delta == "visible ")

        // Cancellation deliberately does not call finish(decoded:). The only
        // text exposed to the consumer is therefore the committed prefix.
        #expect(filter.output == "visible ")
    }

    @Test func emptyAndDuplicateStopsAreHarmless() {
        var filter = StopSequenceFilter(stopSequences: ["", "END", "END"])
        let update = filter.consume(decoded: "plain text")
        #expect(update.delta == "plain text")
        #expect(!update.didStop)
    }

    @Test func canonicallyEquivalentStopsRemainScalarDistinct() {
        let precomposed = "\u{00E9}"
        let decomposed = "e\u{0301}"
        #expect(precomposed == decomposed) // Swift String canonical equality.

        var filter = StopSequenceFilter(stopSequences: [precomposed, decomposed])
        let update = filter.consume(decoded: "before \(decomposed) after")

        #expect(update.didStop)
        #expect(update.delta == "before ")
        #expect(update.matchedStop?.unicodeScalars.count == 2)
        #expect(filter.output == "before ")
    }

    @Test func utf8ReplacementTailRemainsProvisional() {
        var filter = StopSequenceFilter(stopSequences: ["END"])
        #expect(filter.consume(decoded: "ok \u{FFFD}").delta == "ok ")
        #expect(filter.consume(decoded: "ok 😊").delta == "😊")
        #expect(filter.finish(decoded: "ok 😊") == nil)
    }

    @Test func lengthFlushesFinalReplacementButCancellationNeedNot() {
        var lengthFilter = StopSequenceFilter(stopSequences: ["END"])
        #expect(lengthFilter.consume(decoded: "ok \u{FFFD}").delta == "ok ")
        #expect(lengthFilter.finish(decoded: "ok \u{FFFD}") == "\u{FFFD}")

        var cancelledFilter = StopSequenceFilter(stopSequences: ["END"])
        #expect(cancelledFilter.consume(decoded: "ok \u{FFFD}").delta == "ok ")
        #expect(cancelledFilter.output == "ok ")
    }

    @Test func finalRawReplacementCanCompleteStopWithoutLeaking() {
        let stop = "X\u{FFFD}"
        var filter = StopSequenceFilter(stopSequences: [stop])

        // Model a terminal replacement at the end of the cumulative decode;
        // stablePrefix withholds it, leaving X as a possible stop prefix.
        #expect(filter.consume(decoded: "answer \(stop)").delta == "answer ")
        let final = filter.finishUpdate(decoded: "answer \(stop)")

        #expect(final.didStop)
        #expect(final.delta == nil)
        #expect(filter.output == "answer ")
    }

    @Test func divergentCumulativeDecodeFailsClosed() {
        var filter = StopSequenceFilter(stopSequences: ["END"])
        #expect(filter.consume(decoded: "already visible").delta == "already visible")

        let update = filter.consume(decoded: "a rewritten decode")
        #expect(!update.isConsistent)
        #expect(update.delta == nil)
        #expect(filter.output == "already visible")
    }
}
