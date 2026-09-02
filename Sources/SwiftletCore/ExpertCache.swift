import Foundation
import Metal

/// Bounded expert-blob cache over a .qpack container: fixed slot pool of
/// GPU-shared buffers, filled by single preads, LFU eviction with recency
/// tie-break (TurboFieldfare's benchmarked policy). This replaces OS paging so
/// the working set can never thrash the machine: memory use is exactly
/// `slots * expertStride`, no more.
///
/// S3c: a batch resolves every hit and every victim first (the policy is
/// sequential by definition -- each victim choice excludes the batch's
/// earlier members), then issues the batch's preads together. Victim
/// selection scans two flat per-slot arrays instead of two dictionaries;
/// the choice is identical (minimum (frequency, last use), lowest slot on
/// ties), only cheaper.
public final class ExpertCache {
    let reader: QpackExpertReader
    public let stride: Int
    private let device: MTLDevice

    private var slots: [MTLBuffer] = []
    private var slotKey: [Int64] = []          // key occupying each slot, -1 free
    private var keyToSlot: [Int64: Int] = [:]
    /// Access frequency per key, resident or not (LFU with history).
    private var freq: [Int64: Int] = [:]
    /// Per-slot mirror of `freq[slotKey[s]]` for resident keys, and the
    /// tick of each slot's last use -- the two numbers the victim scan reads.
    private var slotFreq: [Int] = []
    private var slotLastUse: [Int] = []
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

    /// Test hook: the slot holding (layer, expert), or nil when not resident.
    func residentSlot(layer: Int, expert: Int) -> Int? {
        keyToSlot[key(layer, expert)]
    }

    /// Buffers for a batch of experts in one layer. The whole batch is
    /// resident simultaneously (a member of the batch is never evicted to make
    /// room for another member). Misses are read into their slots together
    /// (concurrent preads when there is more than one); a failed read leaves
    /// its slot free rather than claiming a key whose bytes never landed.
    func buffers(layer: Int, experts: [Int]) throws -> [MTLBuffer] {
        tick += 1
        var protectedSlots = Set<Int>()
        var result: [MTLBuffer] = []
        result.reserveCapacity(experts.count)
        var fills: [(slot: Int, expert: Int)] = []

        for e in experts {
            let k = key(layer, e)
            let f = freq[k, default: 0] + 1
            freq[k] = f
            if let s = keyToSlot[k] {
                hits += 1
                slotLastUse[s] = tick
                slotFreq[s] = f
                protectedSlots.insert(s)
                result.append(slots[s])
                continue
            }
            misses += 1
            let s = try slotForFill(excluding: protectedSlots)
            if slotKey[s] >= 0 { keyToSlot.removeValue(forKey: slotKey[s]) }
            slotKey[s] = k
            keyToSlot[k] = s
            slotLastUse[s] = tick
            slotFreq[s] = f
            protectedSlots.insert(s)
            result.append(slots[s])
            fills.append((slot: s, expert: e))
        }

        if !fills.isEmpty {
            do {
                try reader.readExperts(
                    layer: layer,
                    fills.map { (expert: $0.expert, into: slots[$0.slot].contents()) })
            } catch {
                // Nothing in the batch is trustworthy after a short read:
                // release every slot this batch filled so no key points at
                // bytes that never arrived.
                for fill in fills {
                    keyToSlot.removeValue(forKey: slotKey[fill.slot])
                    slotKey[fill.slot] = -1
                    slotFreq[fill.slot] = 0
                }
                throw error
            }
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
            slotFreq.append(0)
            slotLastUse.append(0)
            return slots.count - 1
        }
        if let free = slotKey.firstIndex(of: -1) { return free }
        var victim = -1
        var victimFreq = Int.max
        var victimUse = Int.max
        for s in 0..<slots.count where !excluding.contains(s) {
            let f = slotFreq[s]
            let u = slotLastUse[s]
            if f < victimFreq || (f == victimFreq && u < victimUse) {
                victimFreq = f
                victimUse = u
                victim = s
            }
        }
        guard victim >= 0 else { throw Checkpoint.Error.badShape("expert cache too small for batch") }
        return victim
    }
}
