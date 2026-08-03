import Foundation
import Metal

/// Bounded expert-blob cache over a .qpack container: fixed slot pool of
/// GPU-shared buffers, filled by single preads, LFU eviction with recency
/// tie-break (TurboFieldfare's benchmarked policy). This replaces OS paging so
/// the working set can never thrash the machine: memory use is exactly
/// `slots * expertStride`, no more.
public final class ExpertCache {
    let reader: QpackExpertReader
    public let stride: Int
    private let device: MTLDevice

    private var slots: [MTLBuffer] = []
    private var slotKey: [Int64] = []          // key occupying each slot, -1 free
    private var keyToSlot: [Int64: Int] = [:]
    private var freq: [Int64: Int] = [:]
    private var lastUse: [Int: Int] = [:]      // slot -> tick
    private var tick = 0

    public private(set) var hits = 0
    public private(set) var misses = 0

    private let maxSlots: Int

    init(containerDir: URL, device: MTLDevice, budgetBytes: Int) throws {
        self.device = device
        reader = try QpackExpertReader(containerDir: containerDir)
        stride = reader.layout.expertStride
        guard stride > 0, !reader.layout.sections.isEmpty else {
            throw Checkpoint.Error.badShape("corrupt container: empty expert layout (re-download the model)")
        }
        let total = reader.layout.expertCount * reader.layout.layerCount
        maxSlots = min(max(budgetBytes / stride, 16), total)
        // Slots allocate lazily on demand (memory grows with use, never past
        // the budget) — important on iOS where an up-front multi-GB allocation
        // invites jetsam before the model even runs.
    }

    public var slotCount: Int { maxSlots }
    public var allocatedSlots: Int { slots.count }

    private func key(_ layer: Int, _ expert: Int) -> Int64 {
        Int64(layer) << 32 | Int64(expert)
    }

    /// Buffers for a batch of experts in one layer. The whole batch is
    /// resident simultaneously (a member of the batch is never evicted to make
    /// room for another member).
    func buffers(layer: Int, experts: [Int]) throws -> [MTLBuffer] {
        tick += 1
        var protectedSlots = Set<Int>()
        var result: [MTLBuffer] = []
        result.reserveCapacity(experts.count)

        for e in experts {
            let k = key(layer, e)
            freq[k, default: 0] += 1
            if let s = keyToSlot[k] {
                hits += 1
                lastUse[s] = tick
                protectedSlots.insert(s)
                result.append(slots[s])
                continue
            }
            misses += 1
            let s = try slotForFill(excluding: protectedSlots)
            if slotKey[s] >= 0 { keyToSlot.removeValue(forKey: slotKey[s]) }
            try reader.readExpert(layer: layer, expert: e, into: slots[s].contents())
            slotKey[s] = k
            keyToSlot[k] = s
            lastUse[s] = tick
            protectedSlots.insert(s)
            result.append(slots[s])
        }
        return result
    }

    /// LFU victim with recency tie-break; grow-on-demand, then free slots,
    /// then eviction.
    private func slotForFill(excluding: Set<Int>) throws -> Int {
        if slots.count < maxSlots,
           let b = device.makeBuffer(length: stride, options: .storageModeShared) {
            slots.append(b)
            slotKey.append(-1)
            return slots.count - 1
        }
        if let free = slotKey.firstIndex(of: -1) { return free }
        var victim = -1
        var victimScore = (Int.max, Int.max)
        for s in 0..<slots.count where !excluding.contains(s) {
            let f = freq[slotKey[s]] ?? 0
            let score = (f, lastUse[s] ?? 0)
            if score < victimScore {
                victimScore = score
                victim = s
            }
        }
        guard victim >= 0 else { throw Checkpoint.Error.badShape("expert cache too small for batch") }
        return victim
    }
}
