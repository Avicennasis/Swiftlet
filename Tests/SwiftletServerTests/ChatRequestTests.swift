import Foundation
import Testing
import SwiftletCore
@testable import SwiftletServer

@Suite struct ChatRequestTests {
    private func decode(_ json: String) throws -> ChatRequest {
        try JSONDecoder().decode(ChatRequest.self, from: Data(json.utf8))
    }

    @Test func plainStringContent() throws {
        let r = try decode(
            #"{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"hi"}],"stream":true,"max_tokens":512,"temperature":0.7}"#
        )
        #expect(r.messages.map(\.role) == ["system", "user"])
        #expect(r.messages.map(\.content.text) == ["sys", "hi"])
        #expect(r.stream == true)
        #expect(r.max_tokens == 512)
    }

    @Test func arrayOfPartsContent() throws {
        let r = try decode(
            #"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#
        )
        #expect(r.messages[0].content.text == "hi")
    }

    @Test func multipleTextPartsJoined() throws {
        let r = try decode(
            #"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}"#
        )
        #expect(r.messages[0].content.text == "ab")
    }

    private func expectRefused(type: String, _ json: String) {
        do {
            _ = try decode(json)
            Issue.record("a \"\(type)\" content part was accepted")
        } catch let part as ChatContent.UnsupportedContentPart {
            #expect(part.type == type)
            #expect(part.description.contains("\"\(type)\""), "the refusal must name the part type")
        } catch {
            Issue.record("refused with \(error), not by content part type")
        }
    }

    /// A non-text part is refused by name, not dropped. Before this the part
    /// vanished at decode and the model answered the empty text around it.
    @Test func imagePartIsRefusedByName() {
        expectRefused(type: "image_url",
            #"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,xxx"}}]}]}"#)
    }

    /// Refusal is whole-request: the text beside the image would have been
    /// answered as if the image had never been sent.
    @Test func textBesideAnImagePartIsStillRefused() {
        expectRefused(type: "image_url",
            #"{"messages":[{"role":"user","content":[{"type":"text","text":"what is in this picture?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,xxx"}}]}]}"#)
    }

    /// Every foreign type is named as itself, so a client sees which one.
    @Test func otherPartTypesAreRefusedByTheirOwnName() {
        expectRefused(type: "input_audio",
            #"{"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"xx","format":"wav"}}]}]}"#)
        expectRefused(type: "input_image",
            #"{"messages":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,xxx"}]}]}"#)
    }

    /// Only the part type decides: a text part in a later message after an
    /// image part in an earlier one is still refused, and a request with
    /// text parts alone still decodes.
    @Test func imagePartInAnEarlierMessageIsRefused() {
        expectRefused(type: "image_url",
            #"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,xxx"}}]},{"role":"assistant","content":"?"},{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#)
    }

    /// A part that names no type but carries text is text (unchanged
    /// leniency); a `text` part with no text is a malformed body.
    @Test func typelessTextPartIsText() throws {
        let r = try decode(
            #"{"messages":[{"role":"user","content":[{"text":"hi"}]}]}"#
        )
        #expect(r.messages[0].content.text == "hi")
    }

    @Test func textPartWithoutTextIsMalformed() {
        #expect(throws: DecodingError.self) {
            _ = try decode(
                #"{"messages":[{"role":"user","content":[{"type":"text"}]}]}"#
            )
        }
    }

    @Test func nullContentWithToolCalls() throws {
        let r = try decode(
            #"{"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"}}]}]}"#
        )
        #expect(r.messages[0].role == "assistant")
        #expect(r.messages[0].content.text == "")
    }

    @Test func unknownTopLevelKeysIgnored() throws {
        let r = try decode(
            #"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"bash"}}],"stream_options":{},"store":false,"max_completion_tokens":256}"#
        )
        #expect(r.messages.count == 1)
        #expect(r.max_tokens == nil)
        #expect(r.max_completion_tokens == 256)
    }

    @Test func stopAcceptsStringOrArray() throws {
        let one = try decode(
            #"{"messages":[{"role":"user","content":"hi"}],"stop":"END"}"#
        )
        #expect(one.stop?.values == ["END"])

        let many = try decode(
            #"{"messages":[{"role":"user","content":"hi"}],"stop":["END","USER:"]}"#
        )
        #expect(many.stop?.values == ["END", "USER:"])
    }

    @Test func nonStringStopThrows() {
        #expect(throws: DecodingError.self) {
            _ = try decode(
                #"{"messages":[{"role":"user","content":"hi"}],"stop":42}"#
            )
        }
    }

    @Test func finishReasonMappingIsTruthful() {
        #expect(openAIFinishReason(.stop) == "stop")
        #expect(openAIFinishReason(.length) == "length")
        #expect(openAIFinishReason(.cancelled) == nil)
        #expect(openAIFinishReason(nil) == nil)
    }

    @Test func nonterminalStreamingPayloadCarriesNullFinishReason() throws {
        let payload = completionPayload(
            id: "chatcmpl-test", text: nil, delta: "piece", finish: nil,
            model: "swiftlet-test"
        )
        let choices = try #require(payload["choices"] as? [[String: Any]])
        let choice = try #require(choices.first)
        #expect(choice["finish_reason"] is NSNull)

        let encoded = try JSONSerialization.data(withJSONObject: payload)
        let json = try #require(String(data: encoded, encoding: .utf8))
        #expect(json.contains(#""finish_reason":null"#))
    }

    @Test func terminalStreamingPayloadCarriesConcreteFinishReason() throws {
        let payload = completionPayload(
            id: "chatcmpl-test", text: nil, delta: "", finish: "length"
        )
        let choices = try #require(payload["choices"] as? [[String: Any]])
        let choice = try #require(choices.first)
        #expect(choice["finish_reason"] as? String == "length")
    }

    @Test func connectionOwnerCancelsQueuedAndActiveWork() {
        let owner = ConnectionGenerationOwner()
        let first = GenerationCancellation()
        let second = GenerationCancellation()
        owner.register(id: "first", cancellation: first)
        owner.register(id: "second", cancellation: second)

        owner.cancelAll()

        #expect(first.isCancelled)
        #expect(second.isCancelled)
    }

    @Test func completedWorkIsNoLongerOwnedByConnection() {
        let owner = ConnectionGenerationOwner()
        let completed = GenerationCancellation()
        owner.register(id: "done", cancellation: completed)
        owner.finish(id: "done")

        owner.cancelAll()

        #expect(!completed.isCancelled)
    }

    @Test func disconnectCancellationIsScopedToItsConnection() {
        let disconnectedOwner = ConnectionGenerationOwner()
        let liveOwner = ConnectionGenerationOwner()
        let disconnectedRequest = GenerationCancellation()
        let liveRequest = GenerationCancellation()
        disconnectedOwner.register(id: "queued", cancellation: disconnectedRequest)
        liveOwner.register(id: "active", cancellation: liveRequest)

        disconnectedOwner.cancelAll()

        #expect(disconnectedRequest.isCancelled)
        #expect(!liveRequest.isCancelled)
    }

    @Test func writeFailureCancelsOnlyItsOwnedRequest() {
        let owner = ConnectionGenerationOwner()
        let failedRequest = GenerationCancellation()
        let neighboringRequest = GenerationCancellation()
        owner.register(id: "failed", cancellation: failedRequest)
        owner.register(id: "neighbor", cancellation: neighboringRequest)

        #expect(owner.failWrite(id: "failed"))
        #expect(failedRequest.isCancelled)
        #expect(!neighboringRequest.isCancelled)
        #expect(!owner.failWrite(id: "unknown"))

        owner.cancelAll()
        #expect(neighboringRequest.isCancelled)
    }

    @Test func malformedInputThrows() {
        #expect(throws: DecodingError.self) {
            _ = try decode(#"{"messages":"nope"}"#)
        }
    }

    @Test func nonSpecContentTypeThrows() {
        #expect(throws: DecodingError.self) {
            _ = try decode(#"{"messages":[{"role":"user","content":42}]}"#)
        }
    }
}
