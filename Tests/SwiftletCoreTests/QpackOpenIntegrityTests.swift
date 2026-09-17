import Foundation
import Testing
@testable import SwiftletCore

/// A container is checked against its own manifest and layout when it is
/// opened, not when a blob happens to be read. Before this
/// `QpackExpertReader(containerDir:)` read `layout.json` and nothing else, so
/// a truncated `layer_NN.bin` (an interrupted copy, a shard from another
/// container) opened cleanly, the model reported itself ready, and the
/// failure surfaced mid-generation as `short read: layer L expert E`. The
/// manifest already records every file's size, so no receipt is needed: a
/// container copied by hand is verified exactly like an installed one.
@Suite struct QpackOpenIntegrityTests {
    static let fixturesDir = MetalModelTests.fixturesDir

    static func repackedTiny() throws -> URL {
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("open-integrity-\(UUID().uuidString).qpack")
        var repacker = QpackRepacker(checkpointDir: Self.fixturesDir.appendingPathComponent("tiny-model-q4"), outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()
        return out
    }

    static func resize(_ file: URL, to size: Int) throws {
        let handle = try FileHandle(forWritingTo: file)
        if UInt64(size) < (try handle.seekToEnd()) {
            try handle.truncate(atOffset: UInt64(size))
        } else {
            try handle.seekToEnd()
            let current = Int(try handle.offset())
            try handle.write(contentsOf: Data(repeating: 0, count: size - current))
        }
        try handle.close()
    }

    static func sizeOnDisk(_ file: URL) throws -> Int {
        try FileManager.default.attributesOfItem(atPath: file.path)[.size] as! Int
    }

    @Test func intactContainerOpens() throws {
        let out = try Self.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let reader = try QpackExpertReader(containerDir: out)
        let layout = try Qpack.verify(containerDir: out)
        #expect(reader.layout.expertStride == layout.expertStride)
        #expect(reader.layout.layerCount == layout.layerCount)
    }

    /// The case that used to pass open and fail mid-generation.
    @Test func truncatedLayerFileIsRefusedAtOpen() throws {
        let out = try Self.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let file = out.appendingPathComponent("packed_experts/layer_00.bin")
        let full = try Self.sizeOnDisk(file)
        try Self.resize(file, to: full - 1)

        do {
            _ = try QpackExpertReader(containerDir: out)
            Issue.record("a container with a truncated layer file was opened")
        } catch Qpack.Error.sizeMismatch(let path, let expected, let actual) {
            #expect(path == "packed_experts/layer_00.bin")
            #expect(expected == full)
            #expect(actual == full - 1)
            let text = Qpack.Error.sizeMismatch(path: path, expected: expected, actual: actual).description
            #expect(text.contains(path) && text.contains("\(expected)") && text.contains("\(actual)"),
                    "the refusal must name the file and both sizes")
        } catch {
            Issue.record("refused with \(error), not by file size")
        }
    }

    /// If the manifest agrees with a wrong size (a hand-edited manifest, or a
    /// container whose layout and blobs were produced by different runs), the
    /// layout still decides: every blob file is stride × experts.
    @Test func layoutDisagreementIsRefusedEvenWhenTheManifestAgrees() throws {
        let out = try Self.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let file = out.appendingPathComponent("packed_experts/layer_01.bin")
        let full = try Self.sizeOnDisk(file)
        try Self.resize(file, to: full - 4096)
        let manifestURL = out.appendingPathComponent("manifest.json")
        var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
        var files = manifest["files"] as! [String: Int]
        files["packed_experts/layer_01.bin"] = full - 4096
        manifest["files"] = files
        try JSONSerialization.data(withJSONObject: manifest).write(to: manifestURL)

        do {
            _ = try QpackExpertReader(containerDir: out)
            Issue.record("a container whose blob disagrees with its layout was opened")
        } catch Qpack.Error.layoutMismatch(let why) {
            #expect(why.contains("packed_experts/layer_01.bin"))
            #expect(why.contains("\(full)"), "the message must state the size the layout promises")
        } catch {
            Issue.record("refused with \(error), not by layout")
        }
    }

    @Test func missingLayerFileIsRefusedAtOpen() throws {
        let out = try Self.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        try FileManager.default.removeItem(at: out.appendingPathComponent("packed_experts/layer_02.bin"))
        do {
            _ = try QpackExpertReader(containerDir: out)
            Issue.record("a container missing a layer file was opened")
        } catch Qpack.Error.missingFile(let path) {
            #expect(path == "packed_experts/layer_02.bin")
        } catch {
            Issue.record("refused with \(error), not as a missing file")
        }
    }

    /// An oversized dense file is as foreign as a short one: it is not the
    /// file the manifest describes.
    @Test func oversizedDenseFileIsRefusedAtOpen() throws {
        let out = try Self.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let file = out.appendingPathComponent("model.safetensors")
        let full = try Self.sizeOnDisk(file)
        try Self.resize(file, to: full + 1)
        do {
            _ = try QpackExpertReader(containerDir: out)
            Issue.record("a container with an oversized dense file was opened")
        } catch Qpack.Error.sizeMismatch(let path, let expected, let actual) {
            #expect(path == "model.safetensors")
            #expect(expected == full && actual == full + 1)
        } catch {
            Issue.record("refused with \(error), not by file size")
        }
    }

    /// The installer writes `manifest.json` last, so a directory without one
    /// is either not a container or an interrupted install; the model opener
    /// says so by name instead of failing on a file read.
    @Test func containerWithoutManifestIsRefusedByName() throws {
        let out = try Self.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        try FileManager.default.removeItem(at: out.appendingPathComponent("manifest.json"))
        do {
            _ = try Qpack.verify(containerDir: out)
            Issue.record("a directory without manifest.json verified as a container")
        } catch Qpack.Error.notAContainer(let dir) {
            #expect(dir == out.path)
        } catch {
            Issue.record("refused with \(error), not as a non-container")
        }
    }

    /// The whole reader path stays available to a scaffold that carries only
    /// `packed_experts/` (the Metal cache tests build one): the manifest
    /// check applies to containers, which always have one.
    @Test func layoutOnlyScaffoldStillOpensForTheCacheTests() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("open-integrity-scaffold-\(UUID().uuidString)")
        let dir = root.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let layout = Qpack.Layout(
            expertCount: 2, layerCount: 1, expertStride: 16_384,
            sections: [Qpack.Section(name: "gate_proj.weight", dtype: "uint8", shape: [16_384], offset: 0, size: 16_384)],
            linearLayers: [true])
        try JSONEncoder().encode(layout).write(to: dir.appendingPathComponent("layout.json"))
        let reader = try QpackExpertReader(containerDir: root)
        #expect(reader.layout.expertCount == 2)
    }
}
