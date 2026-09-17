import Foundation
import Testing
@testable import SwiftletCore

/// A directory without `config.json` is not a checkpoint, and must be refused
/// by name at open. Before this `Checkpoint(dir:)` read the absent file as an
/// empty config, so a shard directory with no config opened as an
/// unquantized checkpoint and failed later as `missing tensor ...scales`,
/// the wrong check with a misleading message; `QwenConfig(url:)` surfaced a
/// Foundation file error, or, for a config that was valid JSON but not an
/// object, "missing field hidden_size". Both openers now decide the same way.
@Suite struct CheckpointConfigPresenceTests {
    static let fixturesDir = MetalModelTests.fixturesDir

    static func copyOfFixture(_ name: String) throws -> URL {
        let dst = FileManager.default.temporaryDirectory
            .appendingPathComponent("config-presence-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: Self.fixturesDir.appendingPathComponent(name), to: dst)
        return dst
    }

    static func expectMissingConfig(in dir: URL, _ open: () throws -> Void) {
        do {
            try open()
            Issue.record("a directory without config.json was opened")
        } catch Checkpoint.Error.missingConfig(let named) {
            #expect(named == dir.path, "the refusal must name the directory")
            #expect(Checkpoint.Error.missingConfig(named).description.contains("not a checkpoint directory"))
        } catch {
            Issue.record("refused with \(error), not as a missing config.json")
        }
    }

    static func expectMalformedConfig(_ open: () throws -> Void) {
        do {
            try open()
            Issue.record("a non-object config.json was accepted")
        } catch Checkpoint.Error.malformedConfig(let path, let reason) {
            #expect(path.hasSuffix("config.json"))
            #expect(!reason.isEmpty)
        } catch {
            Issue.record("refused with \(error), not as a malformed config.json")
        }
    }

    /// The shard directory case: everything but config.json is present.
    /// Before, `Checkpoint(dir:)` opened it and the first quantized module
    /// read failed on `.scales`; now the open itself is refused.
    @Test func shardDirectoryWithoutConfigIsRefusedAtOpen() throws {
        let dir = try Self.copyOfFixture("tiny-model-q4")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.removeItem(at: dir.appendingPathComponent("config.json"))

        Self.expectMissingConfig(in: dir) { _ = try Checkpoint(dir: dir) }
        Self.expectMissingConfig(in: dir) { _ = try QwenConfig(url: dir.appendingPathComponent("config.json")) }
    }

    /// An empty directory is refused for the same reason, not for a missing
    /// `model.safetensors`.
    @Test func emptyDirectoryIsNotACheckpoint() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("config-presence-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        Self.expectMissingConfig(in: dir) { _ = try Checkpoint(dir: dir) }
    }

    @Test func nonObjectConfigIsRefusedByName() throws {
        for body in ["[]", "42", "\"qwen\"", "not json at all", ""] {
            let dir = try Self.copyOfFixture("tiny-model-q4")
            defer { try? FileManager.default.removeItem(at: dir) }
            try Data(body.utf8).write(to: dir.appendingPathComponent("config.json"))
            Self.expectMalformedConfig { _ = try Checkpoint(dir: dir) }
            Self.expectMalformedConfig { _ = try QwenConfig(url: dir.appendingPathComponent("config.json")) }
        }
    }

    /// The refusal names the directory and says what a checkpoint carries.
    @Test func refusalSaysWhatIsMissing() {
        let text = Checkpoint.Error.missingConfig("/models/x").description
        #expect(text.contains("/models/x"))
        #expect(text.contains("config.json"))
        #expect(text.contains("model.safetensors"))
    }

    /// The shipped fixtures and a repacked container (which carries its own
    /// copy of config.json) keep opening through both openers.
    @Test func fixturesAndContainersStillOpen() throws {
        for name in ["tiny-model", "tiny-model-q4", "tiny-model-q35"] {
            let dir = Self.fixturesDir.appendingPathComponent(name)
            _ = try Checkpoint(dir: dir)
            _ = try QwenConfig(url: dir.appendingPathComponent("config.json"))
        }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("config-presence-\(UUID().uuidString).qpack")
        defer { try? FileManager.default.removeItem(at: out) }
        var repacker = QpackRepacker(checkpointDir: Self.fixturesDir.appendingPathComponent("tiny-model-q4"), outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()
        _ = try Checkpoint(dir: out)
        _ = try QwenConfig(url: out.appendingPathComponent("config.json"))
    }
}
