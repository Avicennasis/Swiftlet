import Foundation

/// Architecture description for the Qwen3-Next / Qwen3.5-MoE hybrid family.
///
/// The family interleaves Gated DeltaNet linear-attention layers with gated GQA
/// full-attention layers (`fullAttentionInterval`, 1-based: every Nth layer is
/// full attention), and every layer carries a high-sparsity routed MoE plus one
/// shared expert. Constants verified against the Hugging Face `config.json` of
/// each checkpoint and mlx-lm's `qwen3_next.py` (vendored in `references/`).
public struct ArchConfig: Sendable, Equatable {
    public enum Family: String, Sendable {
        case qwen3Next = "qwen3_next"
        case qwen3_5Moe = "qwen3_5_moe"
    }

    public let family: Family
    public let name: String
    public let repackSource: String

    public let hiddenSize: Int
    public let layerCount: Int
    public let fullAttentionInterval: Int
    public let vocabSize: Int
    public let tieWordEmbeddings: Bool
    public let rmsNormEps: Double
    /// Norm weights are stored zero-centered; apply as (1 + w). See qwen3_next.py sanitize.
    public let zeroCenteredNorms: Bool = true

    // Full attention (GQA) layers. q_proj emits 2x width: queries + output gate.
    public let attnHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let partialRotaryFactor: Double
    public let ropeTheta: Double
    public let maxPositionEmbeddings: Int

    // Gated DeltaNet linear-attention layers.
    public let linearVHeads: Int
    public let linearKHeads: Int
    public let linearKHeadDim: Int
    public let linearVHeadDim: Int
    public let convKernelSize: Int

    // Mixture of experts (every layer; decoder_sparse_step == 1 for all targets).
    public let expertCount: Int
    public let expertTopK: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let normTopKProb: Bool

    // MARK: - Derived layout facts

    public var fullAttentionLayerCount: Int { layerCount / fullAttentionInterval }
    public var linearLayerCount: Int { layerCount - fullAttentionLayerCount }
    /// 0-based check matching mlx-lm: layer i is linear unless (i+1) % interval == 0.
    public func isLinearLayer(_ index: Int) -> Bool {
        (index + 1) % fullAttentionInterval != 0
    }

    /// Parameters in one routed expert: gate/up/down projections.
    public var expertParamCount: Int { 3 * hiddenSize * moeIntermediateSize }
    public var routedExpertTotal: Int { layerCount * expertCount }
    public var routedFetchesPerToken: Int { layerCount * expertTopK }

    /// Bytes for one expert at MLX affine int4, group 64 (4 bits + bf16 scale
    /// + bf16 bias per 64 weights = 4.5 bits/weight effective).
    public var expertBlobBytesInt4G64: Int {
        let bits = expertParamCount * 4 + (expertParamCount / 64) * 32
        return bits / 8
    }

    /// KV bytes per token across the full-attention layers only (fp16 K and V).
    public var kvBytesPerToken: Int {
        fullAttentionLayerCount * kvHeads * headDim * 2 * 2
    }

    /// Fixed DeltaNet recurrent state across all linear layers (f32), constant
    /// with respect to context length.
    public var deltaNetStateBytes: Int {
        linearLayerCount * linearVHeads * linearKHeadDim * linearVHeadDim * 4
    }

    // MARK: - Known models

    public static let qwen3Next80B = ArchConfig(
        family: .qwen3Next,
        name: "Qwen3-Next-80B-A3B-Instruct",
        repackSource: "lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit",
        hiddenSize: 2048,
        layerCount: 48,
        fullAttentionInterval: 4,
        vocabSize: 151_936,
        tieWordEmbeddings: false,
        rmsNormEps: 1e-6,
        attnHeads: 16,
        kvHeads: 2,
        headDim: 256,
        partialRotaryFactor: 0.25,
        ropeTheta: 10_000_000,
        maxPositionEmbeddings: 262_144,
        linearVHeads: 32,
        linearKHeads: 16,
        linearKHeadDim: 128,
        linearVHeadDim: 128,
        convKernelSize: 4,
        expertCount: 512,
        expertTopK: 10,
        moeIntermediateSize: 512,
        sharedExpertIntermediateSize: 512,
        normTopKProb: true
    )

    public static let qwen3_5_397B = ArchConfig(
        family: .qwen3_5Moe,
        name: "Qwen3.5-397B-A17B",
        repackSource: "mlx-community/Qwen3.5-397B-A17B-4bit",
        hiddenSize: 4096,
        layerCount: 60,
        fullAttentionInterval: 4,
        vocabSize: 248_320,
        tieWordEmbeddings: false,
        rmsNormEps: 1e-6,
        attnHeads: 32,
        kvHeads: 2,
        headDim: 256,
        partialRotaryFactor: 0.25,
        ropeTheta: 10_000_000,
        maxPositionEmbeddings: 262_144,
        linearVHeads: 64,
        linearKHeads: 16,
        linearKHeadDim: 128,
        linearVHeadDim: 128,
        convKernelSize: 4,
        expertCount: 512,
        expertTopK: 10,
        moeIntermediateSize: 1024,
        sharedExpertIntermediateSize: 1024,
        normTopKProb: true
    )

    public static let qwen3_6_35B = ArchConfig(
        family: .qwen3_5Moe,
        name: "Qwen3.6-35B-A3B",
        repackSource: "mlx-community/Qwen3.6-35B-A3B-4bit",
        hiddenSize: 2048,
        layerCount: 40,
        fullAttentionInterval: 4,
        vocabSize: 248_320,
        tieWordEmbeddings: false,
        rmsNormEps: 1e-6,
        attnHeads: 16,
        kvHeads: 2,
        headDim: 256,
        partialRotaryFactor: 0.25,
        ropeTheta: 10_000_000,
        maxPositionEmbeddings: 262_144,
        linearVHeads: 32,
        linearKHeads: 16,
        linearKHeadDim: 128,
        linearVHeadDim: 128,
        convKernelSize: 4,
        expertCount: 256,
        expertTopK: 8,
        moeIntermediateSize: 512,
        sharedExpertIntermediateSize: 512,
        normTopKProb: true
    )

    public static let known: [String: ArchConfig] = [
        "qwen3-next-80b": .qwen3Next80B,
        "qwen3.5-397b": .qwen3_5_397B,
        "qwen3.6-35b": .qwen3_6_35B,
    ]
}
