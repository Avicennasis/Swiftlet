import Foundation
import Testing
@testable import SwiftletCore

/// The streaming installer builds the manifest for a container it has never
/// opened as a `Checkpoint`, so it resolved the expert quantization from the
/// checkpoint's top-level `bits`/`group_size`. A mixed-precision checkpoint
/// (8-bit dense default, 4-bit `switch_mlp` experts, as the published DWQ
/// variants) therefore streamed into a container whose manifest said 8-bit,
/// which the load-time cross-check from issue #30 then refused. The installer
/// must resolve per module exactly as the repacker does.
@Suite struct StreamingInstallerExpertQuantTests {
    static let fixturesDir = MetalModelTests.fixturesDir

    static func loadGuardAccepts(_ container: URL) throws -> Int {
        let manifest = try JSONDecoder().decode(
            Qpack.Manifest.self, from: Data(contentsOf: container.appendingPathComponent("manifest.json")))
        let layout = try JSONDecoder().decode(
            Qpack.Layout.self, from: Data(contentsOf: container.appendingPathComponent("packed_experts/layout.json")))
        let weight = try #require(layout.sections.first { $0.name == "gate_proj.weight" })
        let scales = try #require(layout.sections.first { $0.name == "gate_proj.scales" })
        // The same check QwenMetalModel runs on the manifest at load.
        return try Qpack.expertLogicalInDim(
            weightLastDim: weight.shape.last!, scalesLastDim: scales.shape.last!,
            bits: try #require(manifest.quantBits), groupSize: try #require(manifest.quantGroupSize),
            section: "gate_proj.weight")
    }

    /// An 8-bit dense default over 4-bit experts must stream to a manifest that
    /// records 4-bit -- the same value the repacker records for the same source
    /// -- and the streamed container must pass the load-time cross-check.
    @Test func streamingInstallRecordsExpertQuantNotCheckpointDefault() throws {
        let src = try QpackTests.mixedPrecisionCheckpoint(defaultBits: 8, expertBits: [
            "model.layers.0.mlp.switch_mlp.gate_proj": 4,
            "model.layers.0.mlp.switch_mlp.up_proj": 4,
            "model.layers.0.mlp.switch_mlp.down_proj": 4,
        ])
        let viaStream = FileManager.default.temporaryDirectory
            .appendingPathComponent("mixed-stream-\(UUID().uuidString).qpack")
        let viaRepack = FileManager.default.temporaryDirectory
            .appendingPathComponent("mixed-repack-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: viaStream)
            try? FileManager.default.removeItem(at: viaRepack)
        }
        let installer = StreamingInstaller(source: .localDirectory(src), outputDir: viaStream)
        installer.log = { _ in }
        try installer.install()
        var repacker = QpackRepacker(checkpointDir: src, outputDir: viaRepack)
        repacker.log = { _ in }
        try repacker.repack()

        let streamed = try JSONDecoder().decode(
            Qpack.Manifest.self, from: Data(contentsOf: viaStream.appendingPathComponent("manifest.json")))
        let repacked = try JSONDecoder().decode(
            Qpack.Manifest.self, from: Data(contentsOf: viaRepack.appendingPathComponent("manifest.json")))
        #expect(streamed.quantBits == 4, "manifest must record the 4-bit expert precision, not the 8-bit default")
        #expect(streamed.quantGroupSize == 32)
        #expect(streamed.quantBits == repacked.quantBits && streamed.quantGroupSize == repacked.quantGroupSize,
                "the two producers must record the same expert quantization for the same source")
        let inDim = try Self.loadGuardAccepts(viaStream)
        #expect(inDim == 64, "load-time cross-check must accept the streamed container")
    }

    /// Expert projections that disagree cannot be represented by one manifest
    /// value; the installer must refuse before the shard plan writes anything.
    @Test func streamingInstallRejectsDisagreeingExpertQuant() throws {
        let src = try QpackTests.mixedPrecisionCheckpoint(defaultBits: 8, expertBits: [
            "model.layers.0.mlp.switch_mlp.gate_proj": 4,
            "model.layers.0.mlp.switch_mlp.up_proj": 8,
            "model.layers.0.mlp.switch_mlp.down_proj": 4,
        ])
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("mixed-bad-stream-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        let installer = StreamingInstaller(source: .localDirectory(src), outputDir: out)
        installer.log = { _ in }
        #expect(throws: QpackRepacker.Error.self) { try installer.install() }
        for file in ["manifest.json", "packed_experts/layer_00.bin", "model.safetensors"] {
            #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent(file).path),
                    "\(file) must not be written for a refused checkpoint")
        }
    }

    /// A uniformly quantized checkpoint (the shipped fixture) is unaffected.
    @Test func streamingInstallKeepsUniformQuant() throws {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("uniform-stream-\(UUID().uuidString).qpack")
        defer { try? FileManager.default.removeItem(at: out) }
        let installer = StreamingInstaller(source: .localDirectory(src), outputDir: out)
        installer.log = { _ in }
        try installer.install()
        let manifest = try JSONDecoder().decode(
            Qpack.Manifest.self, from: Data(contentsOf: out.appendingPathComponent("manifest.json")))
        #expect(manifest.quantBits == 4 && manifest.quantGroupSize == 32)
        #expect(try Self.loadGuardAccepts(out) == 64)
    }
}
