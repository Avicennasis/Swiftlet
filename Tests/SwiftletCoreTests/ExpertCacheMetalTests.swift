import Foundation
import Metal
import Testing
@testable import SwiftletCore

/// Live-Metal accounting on a synthetic container whose expert stride is one
/// byte over a 16 KiB page. The repacker page-aligns real strides, so this is
/// the only way to make `MTLBuffer.allocatedSize` exceed the stride and drive
/// the cache into its physical-cap path on current Apple devices.
@Suite struct ExpertCacheMetalTests {
    static let stride = 16_385
    static let expertCount = 16
    static let layerCount = 4

    /// Writes `packed_experts/layout.json` plus one blob file per layer whose
    /// every expert is filled with its own index, so a read-back can be
    /// checked byte for byte.
    static func makeContainer() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-cache-metal-\(UUID().uuidString)")
        let dir = root.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let layout = Qpack.Layout(
            expertCount: expertCount,
            layerCount: layerCount,
            expertStride: stride,
            sections: [Qpack.Section(
                name: "gate_proj.weight", dtype: "uint8", shape: [stride], offset: 0, size: stride
            )],
            linearLayers: Array(repeating: true, count: layerCount)
        )
        try JSONEncoder().encode(layout).write(to: dir.appendingPathComponent("layout.json"))
        for layer in 0..<layerCount {
            var blob = Data(count: stride * expertCount)
            for expert in 0..<expertCount {
                blob.replaceSubrange(
                    expert * stride..<(expert + 1) * stride,
                    with: Data(repeating: UInt8(expert), count: stride)
                )
            }
            try blob.write(to: dir.appendingPathComponent(String(format: "layer_%02d.bin", layer)))
        }
        return root
    }

    /// A stride the device cannot allocate at all: `makeBuffer` returns nil
    /// on the very first miss, before any blob is read. The cache must turn
    /// that into an error naming the zero-slot pool, remember the cap so no
    /// further allocation is attempted, and leave the accounting at zero.
    @Test func allocatorRefusalWithNoSlotsIsAnError() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let stride = Qpack.align(device.maxBufferLength + 1, to: Qpack.pageAlignment)
        #expect(device.makeBuffer(length: stride, options: .storageModeShared) == nil)

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-cache-refusal-\(UUID().uuidString)")
        let dir = root.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let layout = Qpack.Layout(
            expertCount: 1, layerCount: 1, expertStride: stride,
            sections: [Qpack.Section(
                name: "gate_proj.weight", dtype: "uint8", shape: [stride], offset: 0, size: stride
            )],
            linearLayers: [true]
        )
        try JSONEncoder().encode(layout).write(to: dir.appendingPathComponent("layout.json"))

        let cache = try ExpertCache(containerDir: root, device: device, budgetBytes: 16 * stride)
        #expect(cache.slotCount == 1)
        for _ in 0..<2 {
            #expect(throws: Checkpoint.Error.self) {
                _ = try cache.buffers(layer: 0, experts: [0])
            }
        }
        #expect(cache.allocatedSlots == 0)
        #expect(cache.slotCount == 0)
        #expect(cache.allocatedBytes == 0)
        #expect(cache.logicalBytes == 0)
        #expect(cache.misses == 2)
    }

    @Test func physicalCapFollowsAllocatorRounding() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let container = try Self.makeContainer()
        defer { try? FileManager.default.removeItem(at: container) }

        // What this device actually allocates for one slot decides the
        // expected shape: a rounding device caps below the logical ceiling, a
        // non-rounding device reaches it. Both are asserted exactly.
        let sample = try #require(device.makeBuffer(length: Self.stride, options: .storageModeShared))
        let slotBytes = max(Self.stride, sample.allocatedSize)
        let logicalSlots = 20
        let budget = logicalSlots * Self.stride
        let expectedSlots = min(logicalSlots, budget / slotBytes)

        let cache = try ExpertCache(containerDir: container, device: device, budgetBytes: budget)
        #expect(cache.slotCount == logicalSlots)
        #expect(cache.allocatedSlots == 0)

        // Fill past the physical ceiling one miss at a time. Every request is
        // a fresh (layer, expert) pair, so each one is a miss.
        var served = 0
        for layer in 0..<Self.layerCount {
            for expert in 0..<Self.expertCount where served < logicalSlots + 3 {
                let bufs = try cache.buffers(layer: layer, experts: [expert])
                #expect(bufs.count == 1)
                #expect(bufs[0].length == Self.stride)
                #expect(bufs[0].allocatedSize == sample.allocatedSize)
                #expect(bufs[0].contents().load(as: UInt8.self) == UInt8(expert))
                served += 1
            }
        }
        #expect(cache.misses == served)
        #expect(cache.hits == 0)
        #expect(cache.allocatedSlots == expectedSlots)
        #expect(cache.slotCount == expectedSlots)
        #expect(cache.logicalBytes == expectedSlots * Self.stride)
        #expect(cache.allocatedBytes == expectedSlots * slotBytes)
        #expect(cache.allocatedBytes <= cache.budgetBytes)
        if slotBytes > Self.stride {
            #expect(cache.allocatedBytes > cache.logicalBytes)
            #expect(expectedSlots < logicalSlots)
        } else {
            #expect(cache.allocatedBytes == cache.logicalBytes)
            #expect(expectedSlots == logicalSlots)
        }

        // Once capped, a whole batch that fits stays resident together and
        // does not grow the pool; one that does not fit is refused, not
        // silently over-allocated.
        let fits = Array(0..<min(expectedSlots, Self.expertCount))
        let batch = try cache.buffers(layer: Self.layerCount - 1, experts: fits)
        for (expert, buf) in zip(fits, batch) {
            #expect(buf.contents().load(as: UInt8.self) == UInt8(expert))
        }
        #expect(cache.allocatedSlots == expectedSlots)
        #expect(cache.allocatedBytes == expectedSlots * slotBytes)
        if expectedSlots < Self.expertCount {
            #expect(throws: Checkpoint.Error.self) {
                _ = try cache.buffers(layer: 0, experts: Array(0..<(expectedSlots + 1)))
            }
            #expect(cache.allocatedSlots == expectedSlots)
        }
    }
}
