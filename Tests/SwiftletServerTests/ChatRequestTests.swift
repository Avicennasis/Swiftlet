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

    @Test func imageOnlyPartsDropped() throws {
        let r = try decode(
            #"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,xxx"}}]}]}"#
        )
        #expect(r.messages[0].content.text == "")
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
