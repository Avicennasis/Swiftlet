import Foundation

/// Qwen's model-owned inference context: the per-session state both Qwen
/// engines carry across incremental steps. Grows KV rows for the GQA layers;
/// keeps a fixed-size conv tail and delta recurrence for the DeltaNet layers.
///
/// Ownership: a context belongs to the model instance that created it
/// (`makeContext`), and that instance is the only one allowed to step it.
/// On the Metal fast path the live recurrent state of the currently bound
/// context sits in the model's GPU buffers; the CPU fields here hold the
/// last captured copy, refreshed whenever the model unbinds or snapshots the
/// context (see QwenMetalModel.bindContext).
public final class QwenInferenceContext: InferenceContext {
    /// Tokens already processed (RoPE offset for the next step).
    public internal(set) var position = 0
    /// layerIndex -> appended K/V rows, laid out [pos][kvHead][headDim].
    var kv: [Int: (k: [Float], v: [Float])] = [:]
    /// layerIndex -> last (convKernel-1) rows of mixed qkv, [row][convDim].
    var convTail: [Int: [Float]] = [:]
    /// layerIndex -> delta recurrence state, [vHead][vDim][kDim].
    var deltaState: [Int: [Float]] = [:]

    /// The model instance that created this context. Weak so a context can
    /// outlive its model without keeping it alive; a context whose owner is
    /// gone is refused by every model (its live GPU state died with the
    /// owner, so continuing it anywhere else would be silently wrong).
    private weak var owner: AnyObject?

    init(owner: AnyObject) {
        self.owner = owner
    }

    public func reset() {
        position = 0
        kv.removeAll()
        convTail.removeAll()
        deltaState.removeAll()
    }

    /// Refuses use by anything but the creating model instance.
    func checkOwner(_ model: AnyObject) throws {
        guard let owner, owner === model else { throw InferenceContextError.foreignContext }
    }
}

extension QwenCPUModel {
    /// Internal spelling kept for the engines' private signatures, so the
    /// Metal file's per-layer helpers did not have to churn for the S7a
    /// seam. Public callers use `QwenInferenceContext`.
    typealias DecodeState = QwenInferenceContext
}

extension QwenCPUModel: InferenceModel {
    public var modelDir: URL { ckpt.dir }
    public var vocabSize: Int { config.vocabSize }

    public func makeContext() -> any InferenceContext { makeQwenContext() }

    /// Typed factory for callers that hold the concrete model.
    public func makeQwenContext() -> QwenInferenceContext {
        QwenInferenceContext(owner: self)
    }

    public func step(_ tokens: [Int], context: any InferenceContext) throws -> [Float] {
        guard let ctx = context as? QwenInferenceContext else {
            throw InferenceContextError.foreignContext
        }
        return try step(tokens, context: ctx)
    }
}

extension QwenMetalModel: InferenceModel {
    public var modelDir: URL { ckpt.dir }
    public var vocabSize: Int { config.vocabSize }

    public func makeContext() -> any InferenceContext { makeQwenContext() }

    /// Typed factory for callers that hold the concrete model.
    public func makeQwenContext() -> QwenInferenceContext {
        QwenInferenceContext(owner: self)
    }

    public func step(_ tokens: [Int], context: any InferenceContext) throws -> [Float] {
        guard let ctx = context as? QwenInferenceContext else {
            throw InferenceContextError.foreignContext
        }
        return try step(tokens, context: ctx)
    }
}
