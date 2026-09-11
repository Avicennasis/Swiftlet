import Foundation
import Testing
@testable import SwiftletCore

/// mlx-lm records a quantization *mode* next to `group_size`/`bits`:
/// `affine` (the default), `mxfp4`, `nvfp4` or `mxfp8`. Only `affine` stores
/// the packed uint32 + scales + biases layout `Checkpoint` dequantizes; the
/// others carry e8m0/e4m3 scales and no biases. Before the mode was read, an
/// `mxfp4` checkpoint (`mlx-community/Qwen3.6-35B-A3B-mxfp4` is one) was
/// treated as affine and failed on the missing `.biases` tensor, an error that
/// blamed the file rather than naming the unsupported mode.
@Suite struct CheckpointQuantModeTests {
    static let fixturesDir = MetalModelTests.fixturesDir

    /// Copies the tiny 4-bit fixture and replaces its `quantization` block.
    /// The tensors stay affine; only the declared mode changes, which is
    /// enough because the refusal is decided from `config.json` alone.
    static func checkpoint(quantization: [String: Any]) throws -> URL {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let dst = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-mode-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: src, to: dst)
        let cfgURL = dst.appendingPathComponent("config.json")
        var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as! [String: Any]
        cfg["quantization"] = quantization
        try JSONSerialization.data(withJSONObject: cfg).write(to: cfgURL)
        return dst
    }

    static func expectRefused(mode: String, module: String, _ open: () throws -> Void) {
        do {
            try open()
            Issue.record("a \"\(mode)\" checkpoint was opened as affine")
        } catch Checkpoint.Error.unsupportedQuantMode(let m, let mod) {
            #expect(m == mode)
            #expect(mod == module, "the refusal must name the offending module")
        } catch {
            Issue.record("refused with \(error), not by quantization mode")
        }
    }

    /// The shipped fixture predates the field; it must read as affine.
    @Test func absentModeIsAffine() throws {
        let ckpt = try Checkpoint(dir: Self.fixturesDir.appendingPathComponent("tiny-model-q4"))
        #expect(ckpt.defaultQuant?.mode == Checkpoint.affineMode)
        #expect(ckpt.quantSpec(for: "model.embed_tokens")?.mode == Checkpoint.affineMode)
    }

    /// An explicit `affine` mode, at the top level and on a per-module
    /// override, dequantizes exactly as the mode-less fixture does.
    @Test func explicitAffineModeLoadsIdentically() throws {
        let dir = try Self.checkpoint(quantization: [
            "group_size": 32, "bits": 4, "mode": Checkpoint.affineMode,
            "model.layers.0.mlp.gate": ["group_size": 32, "bits": 4, "mode": Checkpoint.affineMode],
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        let explicit = try Checkpoint(dir: dir)
        let original = try Checkpoint(dir: Self.fixturesDir.appendingPathComponent("tiny-model-q4"))
        #expect(explicit.quantSpec(for: "model.layers.0.mlp.gate")?.mode == Checkpoint.affineMode)
        for module in ["model.embed_tokens", "model.layers.0.mlp.gate"] {
            #expect(try explicit.moduleWeight(module).map(\.bitPattern)
                    == original.moduleWeight(module).map(\.bitPattern),
                    "\(module): explicit affine mode changed the dequantized bytes")
        }
    }

    /// A per-module override without `mode` is affine: MLX passes the
    /// override dict to `to_quantized`, whose default is affine, and does not
    /// inherit the checkpoint-level mode.
    @Test func overrideWithoutModeIsAffine() throws {
        let dir = try Self.checkpoint(quantization: [
            "group_size": 32, "bits": 4, "mode": Checkpoint.affineMode,
            "model.layers.0.mlp.gate": ["group_size": 32, "bits": 4],
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        let ckpt = try Checkpoint(dir: dir)
        #expect(ckpt.quantSpec(for: "model.layers.0.mlp.gate")?.mode == Checkpoint.affineMode)
        _ = try ckpt.moduleWeight("model.layers.0.mlp.gate")
    }

    /// A non-affine checkpoint default is refused at open, by name, before
    /// any tensor is read.
    @Test(arguments: ["mxfp4", "nvfp4", "mxfp8"])
    func nonAffineDefaultIsRefusedAtOpen(_ mode: String) throws {
        let dir = try Self.checkpoint(quantization: ["group_size": 32, "bits": 4, "mode": mode])
        defer { try? FileManager.default.removeItem(at: dir) }
        Self.expectRefused(mode: mode, module: "quantization") {
            _ = try Checkpoint(dir: dir)
        }
    }

    /// An affine checkpoint with one non-affine module is refused at open and
    /// the refusal names that module.
    @Test func nonAffineOverrideIsRefusedAtOpen() throws {
        let dir = try Self.checkpoint(quantization: [
            "group_size": 32, "bits": 4, "mode": Checkpoint.affineMode,
            "model.layers.0.mlp.gate": ["group_size": 32, "bits": 4, "mode": "nvfp4"],
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        Self.expectRefused(mode: "nvfp4", module: "model.layers.0.mlp.gate") {
            _ = try Checkpoint(dir: dir)
        }
    }

    /// The repacker is the first thing a user points at a downloaded
    /// checkpoint; it must surface the mode refusal, not a missing tensor.
    @Test func repackRefusesNonAffineCheckpointByMode() throws {
        let src = try Self.checkpoint(quantization: ["group_size": 32, "bits": 4, "mode": "mxfp4"])
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("mode-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        Self.expectRefused(mode: "mxfp4", module: "quantization") {
            try repacker.repack()
        }
        #expect(!FileManager.default.fileExists(atPath: out.path), "refusal must not leave a partial container")
    }

    /// The streaming installer reads `config.json` before planning the shard
    /// download; it must refuse a non-affine checkpoint there, before the
    /// first weight byte and before it writes anything into the container.
    /// The local-directory source runs the same code as the HTTP sources.
    @Test func streamingInstallRefusesNonAffineBeforeDownloading() throws {
        let src = try Self.checkpoint(quantization: ["group_size": 32, "bits": 4, "mode": "mxfp4"])
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("mode-stream-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        let installer = StreamingInstaller(source: .localDirectory(src), outputDir: out)
        installer.log = { _ in }
        Self.expectRefused(mode: "mxfp4", module: "quantization") {
            try installer.install()
        }
        for file in ["config.json", "manifest.json", "packed_experts/layer_00.bin", "model.safetensors"] {
            #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent(file).path),
                    "\(file) must not be written for a refused checkpoint")
        }
    }
}
