import Foundation

/// Cooperative cancellation shared by an embedding app, the loopback server,
/// and the blocking model loop. Cancellation is observed only at model-safe
/// boundaries: before/after a CPU step and between Metal token/layer command
/// buffers. It never abandons a command buffer while GPU work is in flight.
public final class GenerationCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    public init() {}

    public func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }

    public var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }
}

/// Why generation ended. `stop` covers model EOS and caller-supplied stop
/// strings; `length` means the requested token budget was exhausted.
public enum GenerationFinishReason: String, Sendable {
    case stop
    case length
    case cancelled
}

/// Internal control-flow error used to unwind a partially processed prompt or
/// token. Callers must discard the associated DecodeState when this is thrown.
enum GenerationInterruption: Swift.Error {
    case cancelled
    case invalidTextStream
}

@inline(__always)
func checkGenerationCancellation(_ shouldCancel: () -> Bool) throws {
    if shouldCancel() { throw GenerationInterruption.cancelled }
}
