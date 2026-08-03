import Foundation

/// A model that can decode incrementally: CPU reference or Metal runtime.
public protocol InferenceModel: AnyObject {
    var config: QwenConfig { get }
    var modelDir: URL { get }
    func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float]
}

extension QwenCPUModel: InferenceModel {
    public var modelDir: URL { ckpt.dir }
}

extension QwenMetalModel: InferenceModel {
    public var modelDir: URL { ckpt.dir }
}

/// Engine-agnostic greedy generation loop shared by the CLI and server.
public final class TextGenerator {
    public let model: any InferenceModel
    public let eosTokens: Set<Int>

    public init(model: any InferenceModel) {
        self.model = model
        eosTokens = Self.eosTokenIds(modelDir: model.modelDir)
    }

    /// EOS ids from generation_config.json / config.json (int or int array).
    public static func eosTokenIds(modelDir: URL) -> Set<Int> {
        var eos = Set<Int>()
        for name in ["generation_config.json", "config.json"] {
            guard let data = FileManager.default.contents(atPath: modelDir.appendingPathComponent(name).path),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            if let v = obj["eos_token_id"] as? Int { eos.insert(v) }
            if let vs = obj["eos_token_id"] as? [Int] { eos.formUnion(vs) }
        }
        return eos
    }

    public struct Stats: Sendable {
        public var promptTokens = 0
        public var generatedTokens = 0
        public var prefillSeconds = 0.0
        public var decodeSeconds = 0.0
    }

    /// Token-id stream over `generate`, in the `AsyncStream` shape iOS chat
    /// UIs consume (Priv AI's engines all expose `-> AsyncStream`). Runs the
    /// blocking decode loop on a utility queue; cancelling the consuming task
    /// stops generation at the next token.
    public func tokenStream(promptIds: [Int], maxNew: Int) -> AsyncThrowingStream<Int, Swift.Error> {
        AsyncThrowingStream { continuation in
            let work = DispatchWorkItem {
                do {
                    var cancelled = false
                    try self.generate(promptIds: promptIds, maxNew: maxNew) { token in
                        if case .terminated = continuation.yield(token) {
                            cancelled = true
                            return false
                        }
                        return true
                    }
                    _ = cancelled
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in }
            DispatchQueue.global(qos: .userInitiated).async(execute: work)
        }
    }

    /// Greedy-decode up to `maxNew` tokens after `promptIds`. Calls `onToken`
    /// for each generated token; return false from it to stop early. EOS is
    /// consumed as a stop signal and not reported.
    @discardableResult
    public func generate(promptIds: [Int], maxNew: Int, onToken: (Int) -> Bool) throws -> Stats {
        var stats = Stats()
        stats.promptTokens = promptIds.count
        let state = QwenCPUModel.DecodeState()

        let prefillStart = Date()
        var logits = try model.step(promptIds, state: state)
        stats.prefillSeconds = -prefillStart.timeIntervalSinceNow

        let decodeStart = Date()
        for _ in 0..<maxNew {
            var best = 0
            for v in 1..<model.config.vocabSize where logits[v] > logits[best] { best = v }
            if eosTokens.contains(best) { break }
            stats.generatedTokens += 1
            if !onToken(best) { break }
            logits = try model.step([best], state: state)
        }
        stats.decodeSeconds = -decodeStart.timeIntervalSinceNow
        return stats
    }
}
