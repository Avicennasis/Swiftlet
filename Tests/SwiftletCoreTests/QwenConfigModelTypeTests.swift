import Foundation
import Testing
@testable import SwiftletCore

/// `model_type` is how mlx-lm picks the model module, and it is the only
/// field that says which family a checkpoint is. Before this check the field
/// defaulted to `qwen3_next`, so a checkpoint of any other family (a
/// `glm5_next` or `deepseek_v4` export whose config happens to carry the same
/// field names) opened as Qwen3-Next and failed later on whichever tensor
/// differed first, with an error that blamed the file. The family is now
/// refused by name at every entry point, before a tensor or a weight byte is
/// read, the way `quantization.mode` already is.
@Suite struct QwenConfigModelTypeTests {
    static let fixturesDir = MetalModelTests.fixturesDir

    /// A flat qwen3_next-shaped config with `model_type` replaced.
    static func flatConfig(modelType: Any?) -> [String: Any] {
        var cfg: [String: Any] = [
            "hidden_size": 64, "num_hidden_layers": 8, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25, "vocab_size": 128, "max_position_embeddings": 512,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2, "linear_key_head_dim": 8,
            "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4, "num_experts": 8,
            "num_experts_per_tok": 2, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
        ]
        if let modelType { cfg["model_type"] = modelType }
        return cfg
    }

    static func expectRefused(_ expected: String?, _ open: () throws -> Void) {
        do {
            try open()
            Issue.record("a \"\(expected ?? "(absent)")\" model_type was accepted")
        } catch QwenConfig.Error.unsupportedModelType(let type) {
            #expect(type == expected, "the refusal must name the model_type it read")
            #expect(QwenConfig.Error.unsupportedModelType(type).description.contains("\"\(type)\""))
        } catch QwenConfig.Error.missingModelType {
            #expect(expected == nil, "refused as absent, but the config names \(expected ?? "")")
        } catch {
            Issue.record("refused with \(error), not by model_type")
        }
    }

    // MARK: the deciding function

    @Test func foreignFamilyIsRefusedByName() throws {
        for type in ["glm5_next", "deepseek_v4", "kimi_k3", "llama", "qwen2_moe", "qwen3_moe"] {
            Self.expectRefused(type) {
                _ = try QwenConfig.modelType(fromConfig: Self.flatConfig(modelType: type))
            }
        }
    }

    /// A multimodal checkpoint names the family at the top level and the text
    /// model under `text_config`; a foreign one is refused whichever carries it.
    @Test func nestedForeignFamilyIsRefused() throws {
        Self.expectRefused("glm5_next_text") {
            _ = try QwenConfig.modelType(fromConfig: [
                "model_type": "glm5_next", "text_config": Self.flatConfig(modelType: "glm5_next_text"),
            ])
        }
        Self.expectRefused("glm5_next") {
            var text = Self.flatConfig(modelType: nil)
            text.removeValue(forKey: "model_type")
            _ = try QwenConfig.modelType(fromConfig: ["model_type": "glm5_next", "text_config": text])
        }
    }

    @Test func absentModelTypeIsRefusedNotGuessed() throws {
        Self.expectRefused(nil) {
            _ = try QwenConfig.modelType(fromConfig: Self.flatConfig(modelType: nil))
        }
        // A non-string model_type is "absent" too: nothing is guessed from it.
        Self.expectRefused(nil) {
            _ = try QwenConfig.modelType(fromConfig: Self.flatConfig(modelType: 7))
        }
    }

    @Test func everySpellingOfTheTwoFamiliesIsAccepted() throws {
        for type in ["qwen3_next", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_5", "qwen3_6_moe", "qwen3_6_moe_text", "qwen3_6"] {
            #expect(try QwenConfig.modelType(fromConfig: Self.flatConfig(modelType: type)) == type)
        }
        // Nested: the text model's spelling is the one returned.
        #expect(try QwenConfig.modelType(fromConfig: [
            "model_type": "qwen3_6_moe", "text_config": ["model_type": "qwen3_6"],
        ]) == "qwen3_6")
    }

    // MARK: entry points

    /// `QwenConfig(url:)` is the opener every model and the repacker go
    /// through; the refusal there is the one a user sees.
    @Test func configOpenRefusesAForeignFamily() throws {
        let url = try QwenConfigTests.write(Self.flatConfig(modelType: "glm5_next"))
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        Self.expectRefused("glm5_next") { _ = try QwenConfig(url: url) }
    }

    @Test func configOpenStillDerivesTheLayoutFromTheAcceptedFamily() throws {
        let next = try QwenConfigTests.write(Self.flatConfig(modelType: "qwen3_next"))
        defer { try? FileManager.default.removeItem(at: next.deletingLastPathComponent()) }
        let n = try QwenConfig(url: next)
        #expect(n.modelType == "qwen3_next")
        #expect(n.deltaLayout == .fusedInterleaved)

        let q36 = try QwenConfigTests.write([
            "model_type": "qwen3_6_moe", "text_config": QwenConfigTests.textConfigCommon().merging(
                ["model_type": "qwen3_6"], uniquingKeysWith: { _, new in new }),
        ])
        defer { try? FileManager.default.removeItem(at: q36.deletingLastPathComponent()) }
        let q = try QwenConfig(url: q36)
        #expect(q.modelType == "qwen3_6")
        #expect(q.deltaLayout == .split)
        #expect(q.weightPrefix == "language_model.")
    }

    /// The shipped fixtures name their families and must keep opening.
    @Test func shippedFixturesStillOpen() throws {
        for (dir, type) in [("tiny-model", "qwen3_next"), ("tiny-model-q4", "qwen3_next"), ("tiny-model-q35", "qwen3_5")] {
            let cfg = try QwenConfig(url: Self.fixturesDir.appendingPathComponent(dir).appendingPathComponent("config.json"))
            #expect(cfg.modelType == type, "\(dir)")
        }
    }

    /// A copy of the 4-bit fixture whose config names another family.
    static func foreignCheckpoint() throws -> URL {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let dst = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-foreign-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: src, to: dst)
        let cfgURL = dst.appendingPathComponent("config.json")
        var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as! [String: Any]
        cfg["model_type"] = "glm5_next"
        try JSONSerialization.data(withJSONObject: cfg).write(to: cfgURL)
        return dst
    }

    /// The repacker refuses before it writes anything.
    @Test func repackerRefusesAForeignFamilyBeforeWriting() throws {
        let src = try Self.foreignCheckpoint()
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-foreign-repack-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        Self.expectRefused("glm5_next") { try repacker.repack() }
        #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent("manifest.json").path))
        #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent("packed_experts/layer_00.bin").path))
    }

    /// The streaming installer refuses before its shard plan and before it
    /// writes `config.json` into the output directory, so a foreign
    /// checkpoint never costs a weight byte.
    @Test func streamingInstallerRefusesAForeignFamilyBeforeTheFirstByte() throws {
        let src = try Self.foreignCheckpoint()
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-foreign-stream-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        let installer = StreamingInstaller(source: .localDirectory(src), outputDir: out)
        installer.log = { _ in }
        Self.expectRefused("glm5_next") { try installer.install() }
        for file in ["config.json", "manifest.json", "packed_experts/layer_00.bin", "model.safetensors"] {
            #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent(file).path),
                    "\(file) was written for a refused checkpoint")
        }
    }
}
