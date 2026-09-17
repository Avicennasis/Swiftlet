import Foundation

/// Runtime model configuration decoded from a checkpoint's `config.json`.
/// Handles both layouts of the family:
///  - qwen3_next (80B): flat fields, `rope_theta` / `partial_rotary_factor`
///    at top level, fused DeltaNet projections.
///  - qwen3_5_moe (Qwen3.5/3.6): multimodal checkpoint with text fields under
///    `text_config`, rope under `rope_parameters`, split DeltaNet projections,
///    weights prefixed `language_model.`.
public struct QwenConfig: Sendable {
    public enum DeltaProjectionLayout: Sendable {
        /// qwen3_next: in_proj_qkvz (interleaved per k-head) + in_proj_ba.
        case fusedInterleaved
        /// qwen3_5: in_proj_qkv (flat [q|k|v]) + in_proj_z + in_proj_b + in_proj_a.
        case split
    }

    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var partialRotaryFactor: Double
    public var ropeTheta: Double
    public var rmsNormEps: Double
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var tieWordEmbeddings: Bool
    public var fullAttentionInterval: Int

    public var linearNumValueHeads: Int
    public var linearNumKeyHeads: Int
    public var linearKeyHeadDim: Int
    public var linearValueHeadDim: Int
    public var linearConvKernelDim: Int

    public var numExperts: Int
    public var numExpertsPerTok: Int
    public var moeIntermediateSize: Int
    public var sharedExpertIntermediateSize: Int
    public var normTopkProb: Bool

    public var deltaLayout: DeltaProjectionLayout
    /// Prepended to every weight name ("" or "language_model.").
    public var weightPrefix: String

    public enum Error: Swift.Error, CustomStringConvertible {
        case missingField(String)
        case invalidField(String)
        /// `config.json` names a model family this engine does not implement.
        /// mlx-lm selects its model module by `model_type`, so a checkpoint
        /// that names another family has a different graph however familiar
        /// its field names look. Refused before any tensor is read; before
        /// this the field defaulted to `qwen3_next` and a foreign checkpoint
        /// failed later on whichever tensor differed first.
        case unsupportedModelType(String)
        /// `config.json` names no `model_type` at all. Every mlx-lm checkpoint
        /// records one, and guessing the family is the failure above.
        case missingModelType

        public var description: String {
            switch self {
            case .missingField(let key): return "config.json: missing field \(key)"
            case .invalidField(let why): return "config.json: \(why)"
            case .unsupportedModelType(let type):
                return "unsupported model_type \"\(type)\": " + QwenConfig.supportedFamilies
            case .missingModelType:
                return "config.json names no model_type; refusing to guess the family (" + QwenConfig.supportedFamilies + ")"
            }
        }
    }

    /// The `model_type` values this engine implements: the Qwen3.5/3.6 MoE
    /// spellings (`qwen3_5_moe`, `qwen3_6_moe`, their nested `text_config`
    /// and `*_text` variants) and Qwen3-Next (`qwen3_next`). Matched by
    /// prefix so a `_text` suffix or a point release still lands in its
    /// family; anything else is refused by name.
    public static let supportedModelTypePrefixes = ["qwen3_5", "qwen3_6", "qwen3_next"]
    static let supportedFamilies =
        "Swiftlet implements Qwen3.5/3.6 MoE (qwen3_5_moe, qwen3_6_moe and their *_text variants) and Qwen3-Next (qwen3_next) only"

    /// The single deciding function for the family. Checkpoint open, the
    /// repacker and the streaming installer (before its shard plan, so a
    /// foreign checkpoint is refused before the first weight byte) all decide
    /// here, so every entry point names the same type the same way. Returns
    /// the accepted `model_type`; multimodal checkpoints carry it under
    /// `text_config` as well as at the top level, and either spelling counts.
    public static func modelType(fromConfig top: [String: Any]) throws -> String {
        let text = (top["text_config"] as? [String: Any]) ?? top
        guard let modelType = (text["model_type"] as? String) ?? (top["model_type"] as? String) else {
            throw Error.missingModelType
        }
        guard supportedModelTypePrefixes.contains(where: { modelType.hasPrefix($0) }) else {
            throw Error.unsupportedModelType(modelType)
        }
        return modelType
    }

    public init(url: URL) throws {
        // Absent or non-object config.json is refused by name (the same
        // decision Checkpoint makes), not read as an empty config whose first
        // missing field then gets the blame.
        let top = try Checkpoint.readConfig(inDirectory: url.deletingLastPathComponent())
        // Multimodal checkpoints nest the text model under text_config.
        let text = (top["text_config"] as? [String: Any]) ?? top
        let isNested = top["text_config"] != nil

        func int(_ key: String) throws -> Int {
            guard let v = text[key] as? Int else { throw Error.missingField(key) }
            return v
        }
        func dbl(_ key: String, _ fallback: Double? = nil) throws -> Double {
            if let v = text[key] as? Double { return v }
            if let v = text[key] as? Int { return Double(v) }
            if let f = fallback { return f }
            throw Error.missingField(key)
        }

        modelType = try Self.modelType(fromConfig: top)
        let isQwen35Family = modelType.hasPrefix("qwen3_5") || modelType.hasPrefix("qwen3_6") || isNested
        hiddenSize = try int("hidden_size")
        numHiddenLayers = try int("num_hidden_layers")
        numAttentionHeads = try int("num_attention_heads")
        numKeyValueHeads = try int("num_key_value_heads")
        headDim = try int("head_dim")
        rmsNormEps = try dbl("rms_norm_eps", 1e-6)
        vocabSize = try int("vocab_size")
        maxPositionEmbeddings = try int("max_position_embeddings")
        guard maxPositionEmbeddings > 0 else {
            throw Error.invalidField("max_position_embeddings must be positive")
        }
        tieWordEmbeddings = (text["tie_word_embeddings"] as? Bool)
            ?? (top["tie_word_embeddings"] as? Bool) ?? false
        fullAttentionInterval = (text["full_attention_interval"] as? Int) ?? 4

        // Rope: qwen3_5 keeps theta/partial factor inside rope_parameters.
        let ropeParams = text["rope_parameters"] as? [String: Any]
        if let rp = ropeParams {
            ropeTheta = (rp["rope_theta"] as? Double) ?? Double(rp["rope_theta"] as? Int ?? 10_000_000)
            partialRotaryFactor = (rp["partial_rotary_factor"] as? Double) ?? 0.25
        } else {
            ropeTheta = try dbl("rope_theta")
            partialRotaryFactor = try dbl("partial_rotary_factor", 0.25)
        }

        linearNumValueHeads = try int("linear_num_value_heads")
        linearNumKeyHeads = try int("linear_num_key_heads")
        linearKeyHeadDim = try int("linear_key_head_dim")
        linearValueHeadDim = try int("linear_value_head_dim")
        linearConvKernelDim = try int("linear_conv_kernel_dim")

        numExperts = try int("num_experts")
        numExpertsPerTok = try int("num_experts_per_tok")
        moeIntermediateSize = try int("moe_intermediate_size")
        sharedExpertIntermediateSize = try int("shared_expert_intermediate_size")
        // Absent-key default is FAMILY-SPECIFIC in mlx-lm: qwen3_5's
        // TextModelArgs defaults to true, qwen3_next's ModelArgs to false.
        // Qwen3.5/3.6 checkpoints omit the key and rely on that default;
        // missing the renormalization scales every MoE output by the raw
        // top-k softmax mass (~0.2-0.6) and wrecks generation.
        normTopkProb = (text["norm_topk_prob"] as? Bool) ?? isQwen35Family

        deltaLayout = isQwen35Family ? .split : .fusedInterleaved
        weightPrefix = isNested ? "language_model." : ""
    }

    /// 0-based; matches mlx-lm: layer i is DeltaNet unless (i+1) % interval == 0.
    public func isLinearLayer(_ index: Int) -> Bool {
        (index + 1) % fullAttentionInterval != 0
    }

    public var keyDim: Int { linearNumKeyHeads * linearKeyHeadDim }
    public var valueDim: Int { linearNumValueHeads * linearValueHeadDim }
    public var convDim: Int { 2 * keyDim + valueDim }
    public var rotaryDims: Int { Int(Double(headDim) * partialRotaryFactor) }
}
