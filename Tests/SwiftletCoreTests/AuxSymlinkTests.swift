import Testing
import Foundation
@testable import SwiftletCore

/// Regression tests for issue #11: `swiftlet-repack --source <hf-snapshot>` wrote
/// the tokenizer/config aux files as dangling symlinks, because HF snapshot dirs
/// store them as relative links into ../../blobs and `copyItem` preserves the
/// link. The repacker now resolves the link to the real file before copying.
@Suite struct AuxSymlinkTests {

    @Test func resolvesRelativeHFStyleSymlinkToRealFile() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("qpack-aux-\(UUID().uuidString)")
        let blobs = root.appendingPathComponent("blobs")
        let snapshot = root.appendingPathComponent("snapshot")
        try fm.createDirectory(at: blobs, withIntermediateDirectories: true)
        try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        // Real bytes live in blobs/; the snapshot exposes them as a *relative*
        // symlink, exactly like a Hugging Face snapshot directory.
        let payload = Data(#"{"tokenizer": true}"#.utf8)
        let blob = blobs.appendingPathComponent("deadbeef")
        try payload.write(to: blob)
        let link = snapshot.appendingPathComponent("tokenizer.json")
        try fm.createSymbolicLink(atPath: link.path, withDestinationPath: "../blobs/deadbeef")

        // The resolver must point at the real blob, not the link.
        let resolved = QpackRepacker.resolveSymlink(link, fileManager: fm)
        #expect(fm.fileExists(atPath: resolved.path))

        // Copying the resolved URL yields a real regular file with the real
        // bytes (the pre-fix `copyItem(at: link)` would have produced a link
        // that dangles once the container moves off the snapshot dir).
        let dst = root.appendingPathComponent("tokenizer.json")
        try fm.copyItem(at: resolved, to: dst)
        let type = try fm.attributesOfItem(atPath: dst.path)[.type] as? FileAttributeType
        #expect(type == .typeRegular)
        #expect(try Data(contentsOf: dst) == payload)
    }

    @Test func passesRealFilesThroughUnchanged() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("qpack-real-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        let file = dir.appendingPathComponent("config.json")
        try Data("{}".utf8).write(to: file)
        let resolved = QpackRepacker.resolveSymlink(file, fileManager: fm)
        // A non-symlink resolves to a path with the same bytes.
        #expect(try Data(contentsOf: resolved) == Data("{}".utf8))
    }
}
