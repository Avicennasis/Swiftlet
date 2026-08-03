import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import SwiftletCore
import Tokenizers

// Loopback OpenAI-compatible Chat Completions server (modeled on
// TurboFieldfare's). One warm model, requests serialized. No auth/TLS: keep it
// on 127.0.0.1.
//
//   swiftlet-server --model <dir> [--port 8080]

struct ChatRequest: Decodable {
    struct Message: Decodable { let role: String; let content: String }
    let messages: [Message]
    let stream: Bool?
    let max_tokens: Int?
    let max_completion_tokens: Int?
}

let cliArgs = CommandLine.arguments
func flag(_ name: String) -> String? {
    guard let i = cliArgs.firstIndex(of: name), i + 1 < cliArgs.count else { return nil }
    return cliArgs[i + 1]
}
guard let modelPath = flag("--model") else {
    print("usage: swiftlet-server --model <dir> [--port 8080]")
    exit(2)
}
let port = Int(flag("--port") ?? "8080") ?? 8080
let modelURL = URL(fileURLWithPath: modelPath)

FileHandle.standardError.write(Data("loading model + tokenizer...\n".utf8))
let tokenizer = try await AutoTokenizer.from(modelFolder: modelURL)
let cpuModel = try QwenCPUModel(modelDir: modelURL)
cpuModel.retainAllLayers = true
let generator = TextGenerator(model: cpuModel)
let modelName = cpuModel.config.modelType
// One request at a time: the generator mutates shared per-layer caches.
let generationQueue = DispatchQueue(label: "swiftlet.generation")

@Sendable func jsonData(_ obj: [String: Any]) -> Data {
    (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{}".utf8)
}

@Sendable func completionPayload(id: String, text: String?, delta: String?, finish: String?) -> [String: Any] {
    var choice: [String: Any] = ["index": 0]
    if let text { choice["message"] = ["role": "assistant", "content": text] }
    if let delta { choice["delta"] = ["content": delta] }
    if let finish { choice["finish_reason"] = finish }
    return [
        "id": id,
        "object": delta != nil ? "chat.completion.chunk" : "chat.completion",
        "created": Int(Date().timeIntervalSince1970),
        "model": modelName,
        "choices": [choice],
    ]
}

final class HTTPHandler: ChannelInboundHandler {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private var requestHead: HTTPRequestHead?
    private var body = ByteBuffer()

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            requestHead = head
            body = ByteBuffer()
        case .body(var chunk):
            body.writeBuffer(&chunk)
        case .end:
            guard let head = requestHead else { return }
            route(context: context, head: head, body: body)
            requestHead = nil
        }
    }

    private func route(context: ChannelHandlerContext, head: HTTPRequestHead, body: ByteBuffer) {
        switch (head.method, head.uri) {
        case (.GET, "/v1/models"):
            respondJSON(context, status: .ok, data: jsonData([
                "object": "list",
                "data": [["id": modelName, "object": "model", "owned_by": "swiftlet"]],
            ]))
        case (.POST, "/v1/chat/completions"):
            handleChat(context: context, body: body)
        default:
            respondJSON(context, status: .notFound, data: jsonData(["error": "not found"]))
        }
    }

    private func handleChat(context: ChannelHandlerContext, body: ByteBuffer) {
        let bytes: [UInt8] = body.getBytes(at: 0, length: body.readableBytes) ?? []
        let data = Data(bytes)
        guard let request = try? JSONDecoder().decode(ChatRequest.self, from: data) else {
            respondJSON(context, status: .badRequest, data: jsonData(["error": "malformed request"]))
            return
        }
        let messages = request.messages.map { ["role": $0.role, "content": $0.content] }
        let maxNew = request.max_tokens ?? request.max_completion_tokens ?? 512
        let streaming = request.stream ?? false
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        let eventLoop = context.eventLoop
        let channel = context.channel

        if streaming {
            var head = HTTPResponseHead(version: .http1_1, status: .ok)
            head.headers.add(name: "Content-Type", value: "text/event-stream")
            head.headers.add(name: "Cache-Control", value: "no-cache")
            head.headers.add(name: "Transfer-Encoding", value: "chunked")
            context.writeAndFlush(wrapOutboundOut(.head(head)), promise: nil)
        }

        @Sendable func writeSSE(_ obj: [String: Any]) {
            let payload = "data: " + (String(data: jsonData(obj), encoding: .utf8) ?? "{}") + "\n\n"
            eventLoop.execute {
                var buf = channel.allocator.buffer(capacity: payload.utf8.count)
                buf.writeString(payload)
                _ = channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buf)))
            }
        }

        generationQueue.async { [wrapOutboundOut] in
            do {
                let promptIds = try tokenizer.applyChatTemplate(messages: messages)
                var generated: [Int] = []
                var printedText = ""
                let stats = try generator.generate(promptIds: promptIds, maxNew: maxNew) { token in
                    generated.append(token)
                    if streaming {
                        let text = tokenizer.decode(tokens: generated)
                        if text.hasPrefix(printedText) {
                            let delta = String(text.dropFirst(printedText.count))
                            if !delta.isEmpty {
                                writeSSE(completionPayload(id: id, text: nil, delta: delta, finish: nil))
                                printedText = text
                            }
                        }
                    }
                    return true
                }
                let fullText = tokenizer.decode(tokens: generated)
                FileHandle.standardError.write(Data(String(
                    format: "[%@] %d prompt + %d generated, prefill %.1fs, %.2f tok/s\n",
                    id as NSString, stats.promptTokens, stats.generatedTokens,
                    stats.prefillSeconds, Double(stats.generatedTokens) / max(stats.decodeSeconds, 0.001)
                ).utf8))

                eventLoop.execute {
                    if streaming {
                        writeSSE(completionPayload(id: id, text: nil, delta: "", finish: "stop"))
                        var buf = channel.allocator.buffer(capacity: 16)
                        buf.writeString("data: [DONE]\n\n")
                        _ = channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buf)))
                        _ = channel.writeAndFlush(HTTPServerResponsePart.end(nil))
                    } else {
                        let data = jsonData(completionPayload(id: id, text: fullText, delta: nil, finish: "stop"))
                        var head = HTTPResponseHead(version: .http1_1, status: .ok)
                        head.headers.add(name: "Content-Type", value: "application/json")
                        head.headers.add(name: "Content-Length", value: String(data.count))
                        var buf = channel.allocator.buffer(capacity: data.count)
                        buf.writeBytes(data)
                        _ = channel.write(HTTPServerResponsePart.head(head))
                        _ = channel.write(HTTPServerResponsePart.body(.byteBuffer(buf)))
                        _ = channel.writeAndFlush(HTTPServerResponsePart.end(nil))
                    }
                }
            } catch {
                eventLoop.execute {
                    let data = jsonData(["error": "\(error)"])
                    var head = HTTPResponseHead(version: .http1_1, status: .internalServerError)
                    head.headers.add(name: "Content-Length", value: String(data.count))
                    var buf = channel.allocator.buffer(capacity: data.count)
                    buf.writeBytes(data)
                    _ = channel.write(HTTPServerResponsePart.head(head))
                    _ = channel.write(HTTPServerResponsePart.body(.byteBuffer(buf)))
                    _ = channel.writeAndFlush(HTTPServerResponsePart.end(nil))
                }
            }
        }
    }

    private func respondJSON(_ context: ChannelHandlerContext, status: HTTPResponseStatus, data: Data) {
        var head = HTTPResponseHead(version: .http1_1, status: status)
        head.headers.add(name: "Content-Type", value: "application/json")
        head.headers.add(name: "Content-Length", value: String(data.count))
        var buf = context.channel.allocator.buffer(capacity: data.count)
        buf.writeBytes(data)
        context.write(wrapOutboundOut(.head(head)), promise: nil)
        context.write(wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
        context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
    }
}

let group = MultiThreadedEventLoopGroup(numberOfThreads: 2)
let bootstrap = ServerBootstrap(group: group)
    .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
    .childChannelInitializer { channel in
        channel.pipeline.configureHTTPServerPipeline().flatMap {
            channel.pipeline.addHandler(HTTPHandler())
        }
    }

let channel = try await bootstrap.bind(host: "127.0.0.1", port: port).get()
print("swiftlet-server listening on http://127.0.0.1:\(port)/v1 (model: \(modelName))")
try await channel.closeFuture.get()
