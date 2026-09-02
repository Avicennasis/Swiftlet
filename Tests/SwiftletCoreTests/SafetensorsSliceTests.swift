import Foundation
import Testing
@testable import SwiftletCore

/// The sub-range readers (`uint32s(_:elementRange:)`, `floats(_:elementRange:)`)
/// feed every per-token embedding lookup. They must return exactly the
/// bytes of the requested range -- and only touch that range: the whole
/// tensor is hundreds of MB on a real model and a per-token copy of it was
/// the largest single item in the M1 decode CPU gap.
@Suite struct SafetensorsSliceTests {
    static let fixturesDir = MetalModelTests.fixturesDir

    /// Writes a minimal safetensors file. `headerPadding` lets the data
    /// section start at an arbitrary byte alignment, which real files do
    /// (the tiny fixtures start their data at offset 34345).
    static func writeSafetensors(
        tensors: [(name: String, dtype: String, shape: [Int], bytes: Data)],
        headerPadding: Int
    ) throws -> URL {
        var header: [String: Any] = [:]
        var offset = 0
        var payload = Data()
        for t in tensors {
            header[t.name] = ["dtype": t.dtype, "shape": t.shape,
                              "data_offsets": [offset, offset + t.bytes.count]]
            offset += t.bytes.count
            payload.append(t.bytes)
        }
        var json = try JSONSerialization.data(withJSONObject: header)
        json.append(Data(repeating: 0x20, count: headerPadding))
        var file = Data()
        var len = UInt64(json.count).littleEndian
        file.append(Data(bytes: &len, count: 8))
        file.append(json)
        file.append(payload)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("slice-\(UUID().uuidString).safetensors")
        try file.write(to: url)
        return url
    }

    /// One embedding row must equal the same row of the whole-module read,
    /// bitwise, on every fixture: packed U32 + scales/biases (q4, the
    /// dequantizing path the 35B takes) and plain F32 under a
    /// `language_model.` prefix (q35) -- including the last row.
    @Test(arguments: ["tiny-model-q4", "tiny-model-q35"])
    func embeddingRowSliceMatchesWholeModule(_ fixture: String) throws {
        let dir = Self.fixturesDir.appendingPathComponent(fixture)
        let ckpt = try Checkpoint(dir: dir)
        #expect(ckpt.isQuantized("model.embed_tokens") == (fixture == "tiny-model-q4"),
                "\(fixture): fixture quantization changed; the test no longer covers both paths")
        let whole = try ckpt.moduleWeight("model.embed_tokens")
        let rows = try ckpt.shape("model.embed_tokens.weight")[0]
        let cols = whole.count / rows
        for row in [0, 1, rows / 2, rows - 1] {
            let slice = try ckpt.moduleWeightSlice("model.embed_tokens", rowRange: row..<(row + 1))
            #expect(slice.count == cols, "\(fixture) row \(row): width")
            #expect(slice.map(\.bitPattern) == whole[(row * cols)..<((row + 1) * cols)].map(\.bitPattern),
                    "\(fixture) row \(row): slice diverged from the whole-module dequant")
        }
        let pair = try ckpt.moduleWeightSlice("model.embed_tokens", rowRange: 2..<4)
        #expect(pair.map(\.bitPattern) == whole[(2 * cols)..<(4 * cols)].map(\.bitPattern),
                "\(fixture): two-row slice diverged")
    }

    /// Plain (F32) module rows through the same slice path.
    @Test func plainRowSliceMatchesWholeModule() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model")
        let ckpt = try Checkpoint(dir: dir)
        let whole = try ckpt.moduleWeight("model.embed_tokens")
        let cols = try ckpt.shape("model.embed_tokens.weight").last!
        let rows = whole.count / cols
        for row in [0, rows - 1] {
            let slice = try ckpt.moduleWeightSlice("model.embed_tokens", rowRange: row..<(row + 1))
            #expect(slice.map(\.bitPattern) == whole[(row * cols)..<((row + 1) * cols)].map(\.bitPattern),
                    "row \(row): plain slice diverged")
        }
    }

    /// Every dtype the range readers accept, read from a file whose data
    /// section starts at an odd byte offset: values must be exact, and 2000
    /// single-row reads of a 32 MB tensor must finish in milliseconds (the
    /// whole-tensor copy per read they replaced took 2.2 s on the M1 mini).
    @Test func rangeReadersAreExactAndTouchOnlyTheRange() throws {
        let rows = 4096, u32Cols = 2048, fCols = 64
        var u32 = [UInt32](repeating: 0, count: rows * u32Cols)
        for i in u32.indices { u32[i] = UInt32(truncatingIfNeeded: i &* 2654435761) }
        var f32 = [Float](repeating: 0, count: rows * fCols)
        for i in f32.indices { f32[i] = Float(i) * 0.25 - 3 }
        var f16 = [Float16](repeating: 0, count: rows * fCols)
        for i in f16.indices { f16[i] = Float16(Float(i % 2048) / 16 - 7) }
        var bf16 = [UInt16](repeating: 0, count: rows * fCols)
        for i in bf16.indices { bf16[i] = UInt16(truncatingIfNeeded: i &* 40503) }

        let url = try Self.writeSafetensors(tensors: [
            ("w", "U32", [rows, u32Cols], u32.withUnsafeBytes { Data($0) }),
            ("s32", "F32", [rows, fCols], f32.withUnsafeBytes { Data($0) }),
            ("s16", "F16", [rows, fCols], f16.withUnsafeBytes { Data($0) }),
            ("b16", "BF16", [rows, fCols], bf16.withUnsafeBytes { Data($0) }),
        ], headerPadding: 3)
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try SafetensorsFile(url: url)

        for row in [0, 1, 777, rows - 1] {
            let wr = row * u32Cols, fr = row * fCols
            #expect(try file.uint32s("w", elementRange: wr..<(wr + u32Cols)) == Array(u32[wr..<(wr + u32Cols)]),
                    "U32 row \(row)")
            #expect(try file.floats("s32", elementRange: fr..<(fr + fCols)).map(\.bitPattern)
                    == f32[fr..<(fr + fCols)].map(\.bitPattern), "F32 row \(row)")
            #expect(try file.floats("s16", elementRange: fr..<(fr + fCols)).map(\.bitPattern)
                    == f16[fr..<(fr + fCols)].map { Float($0).bitPattern }, "F16 row \(row)")
            #expect(try file.floats("b16", elementRange: fr..<(fr + fCols)).map(\.bitPattern)
                    == bf16[fr..<(fr + fCols)].map { UInt32($0) << 16 }, "BF16 row \(row)")
        }
        // Odd-length ranges and the whole tensor through the same reader.
        #expect(try file.uint32s("w", elementRange: 5..<8) == Array(u32[5..<8]))
        #expect(try file.uint32s("w", elementRange: 0..<0).isEmpty)
        #expect(try file.uint32s("w") == u32, "whole-tensor U32 read")
        #expect(try file.floats("s16", elementRange: 1..<2).map(\.bitPattern) == [Float(f16[1]).bitPattern])

        let start = ProcessInfo.processInfo.systemUptime
        var checksum: UInt32 = 0
        for i in 0..<2000 {
            let row = (i * 97) % rows
            let r = try file.uint32s("w", elementRange: (row * u32Cols)..<((row + 1) * u32Cols))
            checksum &+= r[i % u32Cols]
        }
        let elapsed = ProcessInfo.processInfo.systemUptime - start
        #expect(checksum != 0)
        #expect(elapsed < 0.25, "2000 row reads took \(elapsed)s: the reader copies the whole tensor")
    }
}
