import Foundation
import Metal
import Testing
@testable import SwiftletCore

/// Metal kernels vs exact CPU references on deterministic pseudo-random data.
@Suite struct MetalKernelTests {
    /// Deterministic LCG so failures reproduce.
    struct Rand {
        var s: UInt64
        init(_ seed: UInt64) { s = seed }
        mutating func next32() -> UInt32 {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return UInt32(truncatingIfNeeded: s >> 32)
        }
        mutating func float() -> Float { Float(next32()) / Float(UInt32.max) * 2 - 1 }
    }

    static func bf16(_ f: Float) -> UInt16 { UInt16(truncatingIfNeeded: f.bitPattern >> 16) }
    static func fromBF16(_ u: UInt16) -> Float { Float(bitPattern: UInt32(u) << 16) }

    @Test(arguments: [4, 8], [false, true])
    func gemvAffineMatchesCPU(bits: Int, useFast: Bool) throws {
        let engine = try MetalEngine()
        let O = 48, I = 128, group = 32
        let perWord = 32 / bits
        var rng = Rand(42)

        let packed = (0..<O * I / perWord).map { _ in rng.next32() }
        let scalesF = (0..<O * I / group).map { _ in rng.float() * 0.1 }
        let biasesF = (0..<O * I / group).map { _ in rng.float() * 0.05 }
        let x = (0..<I).map { _ in rng.float() }

        func cpuReference(scaleAt: (Int) -> Float, biasAt: (Int) -> Float) -> [Float] {
            var y = [Float](repeating: 0, count: O)
            let mask = UInt32((1 << bits) - 1)
            let groups = I / group
            for o in 0..<O {
                var acc: Float = 0
                for i in 0..<I {
                    let word = packed[o * (I / perWord) + i / perWord]
                    let q = Float((word >> (UInt32(bits) * UInt32(i % perWord))) & mask)
                    let g = o * groups + i / group
                    acc += (scaleAt(g) * q + biasAt(g)) * x[i]
                }
                y[o] = acc
            }
            return y
        }

        // f32 scales.
        let yF32 = try engine.gemvQuantized(
            x: x,
            packed: packed.withUnsafeBytes { Data($0) },
            scales: scalesF.withUnsafeBytes { Data($0) },
            biases: biasesF.withUnsafeBytes { Data($0) },
            outDim: O, inDim: I, groupSize: group, bits: bits, scalesType: .f32,
            useFast: useFast
        )
        let refF32 = cpuReference(scaleAt: { scalesF[$0] }, biasAt: { biasesF[$0] })
        for i in 0..<O { #expect(abs(yF32[i] - refF32[i]) < 1e-3, "f32 row \(i): \(yF32[i]) vs \(refF32[i])") }

        // bf16 scales: reference uses the same truncated values.
        let scalesB = scalesF.map(Self.bf16)
        let biasesB = biasesF.map(Self.bf16)
        let yBF16 = try engine.gemvQuantized(
            x: x,
            packed: packed.withUnsafeBytes { Data($0) },
            scales: scalesB.withUnsafeBytes { Data($0) },
            biases: biasesB.withUnsafeBytes { Data($0) },
            outDim: O, inDim: I, groupSize: group, bits: bits, scalesType: .bf16,
            useFast: useFast
        )
        let refBF16 = cpuReference(
            scaleAt: { Self.fromBF16(scalesB[$0]) }, biasAt: { Self.fromBF16(biasesB[$0]) }
        )
        for i in 0..<O { #expect(abs(yBF16[i] - refBF16[i]) < 1e-3, "bf16 row \(i)") }
    }

    @Test func gatedDeltaStepMatchesCPU() throws {
        let engine = try MetalEngine()
        let T = 5, Hk = 2, Hv = 4, Dk = 32, Dv = 8
        var rng = Rand(7)

        let q = (0..<T * Hk * Dk).map { _ in rng.float() }
        let k = (0..<T * Hk * Dk).map { _ in rng.float() }
        let v = (0..<T * Hv * Dv).map { _ in rng.float() }
        let g = (0..<T * Hv).map { _ in abs(rng.float()) * 0.9 }
        let beta = (0..<T * Hv).map { _ in abs(rng.float()) }
        let state0 = (0..<Hv * Dv * Dk).map { _ in rng.float() * 0.1 }

        // CPU reference: references/gated_delta.py ops path.
        var refState = state0
        var refY = [Float](repeating: 0, count: T * Hv * Dv)
        let rep = Hv / Hk
        for t in 0..<T {
            for hv in 0..<Hv {
                let hk = hv / rep
                let qB = t * Hk * Dk + hk * Dk
                let kB = t * Hk * Dk + hk * Dk
                let vB = t * Hv * Dv + hv * Dv
                for dvi in 0..<Dv {
                    let sB = (hv * Dv + dvi) * Dk
                    var kvMem: Float = 0
                    for i in 0..<Dk {
                        refState[sB + i] *= g[t * Hv + hv]
                        kvMem += refState[sB + i] * k[kB + i]
                    }
                    let delta = (v[vB + dvi] - kvMem) * beta[t * Hv + hv]
                    var out: Float = 0
                    for i in 0..<Dk {
                        refState[sB + i] += k[kB + i] * delta
                        out += refState[sB + i] * q[qB + i]
                    }
                    refY[t * Hv * Dv + hv * Dv + dvi] = out
                }
            }
        }

        let (y, state) = try engine.gatedDeltaStep(
            q: q, k: k, v: v, g: g, beta: beta, state: state0,
            T: T, Hk: Hk, Hv: Hv, Dk: Dk, Dv: Dv
        )
        var maxY: Float = 0, maxS: Float = 0
        for i in 0..<refY.count { maxY = max(maxY, abs(y[i] - refY[i])) }
        for i in 0..<refState.count { maxS = max(maxS, abs(state[i] - refState[i])) }
        #expect(maxY < 1e-4, "delta y maxAbsDiff \(maxY)")
        #expect(maxS < 1e-4, "delta state maxAbsDiff \(maxS)")
    }

    // MARK: - S1b token-batched GEMV

    static func quantizedLinear(
        _ engine: MetalEngine, bits: Int, pad: Int, outDim O: Int = 48, inDim I: Int = 128
    ) -> MetalShardStore.GPULinear {
        let group = 32
        let perWord = 32 / bits
        var rng = Rand(UInt64(bits * 100 + pad + 7))
        let packed = (0..<O * I / perWord).map { _ in rng.next32() }
        let scales = (0..<O * I / group).map { _ in rng.float() * 0.1 }
        let biases = (0..<O * I / group).map { _ in rng.float() * 0.05 }
        // One blob: [pad] weights | scales | biases. pad = 2 yields weight
        // rows that are only 2-byte aligned, forcing the scalar kernel.
        var blob = Data(repeating: 0, count: pad)
        packed.withUnsafeBytes { blob.append(contentsOf: $0) }
        let sOff = blob.count
        scales.withUnsafeBytes { blob.append(contentsOf: $0) }
        let bOff = blob.count
        biases.withUnsafeBytes { blob.append(contentsOf: $0) }
        let buf = engine.makeBuffer(blob)
        return MetalShardStore.GPULinear(
            wBuffer: buf, sBuffer: buf, bBuffer: buf, outDim: O, inDim: I,
            isQuantized: true, groupSize: group, bits: bits, scalesType: 0,
            wOff: pad, sOff: sOff, bOff: bOff
        )
    }

    static func plainLinear(
        _ engine: MetalEngine, dtype: UInt32, outDim O: Int = 48, inDim I: Int = 128
    ) -> MetalShardStore.GPULinear {
        var rng = Rand(UInt64(900 + dtype))
        let rows = (0..<O * I).map { _ in rng.float() }
        var blob = Data()
        switch dtype {
        case 0: rows.withUnsafeBytes { blob.append(contentsOf: $0) }
        case 1:
            let h = rows.map { Float16($0) }
            h.withUnsafeBytes { blob.append(contentsOf: $0) }
        default:
            let b = rows.map { Self.bf16($0) }
            b.withUnsafeBytes { blob.append(contentsOf: $0) }
        }
        let buf = engine.makeBuffer(blob)
        return MetalShardStore.GPULinear(
            wBuffer: buf, sBuffer: buf, bBuffer: buf, outDim: O, inDim: I,
            isQuantized: false, plainDtype: dtype
        )
    }

    /// Applies `lin` to `tokens` slots (distinct x/y offsets in shared
    /// buffers) once through per-token encodeGemv dispatches and once through
    /// encodeGemvBatch, requiring bitwise-identical output buffers plus the
    /// mechanical dispatch reduction.
    static func expectBatchMatches(
        _ engine: MetalEngine, _ lin: MetalShardStore.GPULinear, label: String,
        tokens: Int = 3, expectedBatchDispatches: Int = 1
    ) throws {
        let O = lin.outDim, I = lin.inDim
        let xStride = I + 7, yStride = O + 5
        var rng = Rand(4242)
        let x = (0..<tokens * xStride).map { _ in rng.float() }
        let xBuf = engine.makeBuffer(x)
        let slots = (0..<tokens).map { (xOff: $0 * xStride, yOff: $0 * yStride + 3) }
        let yFloats = tokens * yStride + 3

        func run(_ body: (MTLComputeCommandEncoder, MTLBuffer) throws -> Void)
            throws -> (y: [Float], dispatches: Int)
        {
            let yBuf = engine.device.makeBuffer(length: yFloats * 4)!
            memset(yBuf.contents(), 0, yFloats * 4)
            var dispatches = 0
            engine.computeDispatchObserver = { dispatches += 1 }
            defer { engine.computeDispatchObserver = nil }
            let cb = engine.queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            try body(enc, yBuf)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            let y = Array(UnsafeBufferPointer(
                start: yBuf.contents().bindMemory(to: Float.self, capacity: yFloats),
                count: yFloats))
            return (y, dispatches)
        }

        let perToken = try run { enc, yBuf in
            for s in slots {
                try engine.encodeGemv(enc, lin, x: xBuf, xOff: s.xOff, y: yBuf, yOff: s.yOff)
            }
        }
        let batched = try run { enc, yBuf in
            try engine.encodeGemvBatch(enc, lin, x: xBuf, y: yBuf, slots: slots)
        }
        #expect(perToken.dispatches == tokens, "\(label): per-token dispatch count")
        #expect(batched.dispatches == expectedBatchDispatches, "\(label): batched dispatch count")
        #expect(perToken.y.map(\.bitPattern) == batched.y.map(\.bitPattern),
                "\(label): batched GEMV diverges bitwise from per-token dispatches")
    }

    /// Every kernel variant the runtime selects — fast 4/8-bit, the scalar
    /// fallback (2-byte-aligned weights), and each plain dtype — must batch
    /// bitwise-identically, with N token dispatches collapsing to one.
    @Test func gemvBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        try Self.expectBatchMatches(engine, Self.quantizedLinear(engine, bits: 4, pad: 0), label: "q4 fast")
        try Self.expectBatchMatches(engine, Self.quantizedLinear(engine, bits: 8, pad: 0), label: "q8 fast")
        try Self.expectBatchMatches(engine, Self.quantizedLinear(engine, bits: 4, pad: 2), label: "q4 scalar")
        try Self.expectBatchMatches(engine, Self.plainLinear(engine, dtype: 0), label: "plain f32")
        try Self.expectBatchMatches(engine, Self.plainLinear(engine, dtype: 1), label: "plain f16")
        try Self.expectBatchMatches(engine, Self.plainLinear(engine, dtype: 2), label: "plain bf16")
    }

    /// A single slot delegates to the unbatched encode (one dispatch, same
    /// bytes as the decode schedule); a table past the setBytes ceiling
    /// splits into ceil(n / 512) dispatches, still bitwise.
    @Test func gemvBatchRespectsTableLimits() throws {
        let engine = try MetalEngine()
        let lin = Self.quantizedLinear(engine, bits: 4, pad: 0, outDim: 8, inDim: 32)
        try Self.expectBatchMatches(engine, lin, label: "single slot", tokens: 1)
        try Self.expectBatchMatches(
            engine, lin, label: "split table", tokens: 513, expectedBatchDispatches: 2)
    }
}
