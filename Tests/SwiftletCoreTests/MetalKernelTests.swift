import Foundation
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

    @Test(arguments: [4, 8])
    func gemvAffineMatchesCPU(bits: Int) throws {
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
            outDim: O, inDim: I, groupSize: group, bits: bits, scalesType: .f32
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
            outDim: O, inDim: I, groupSize: group, bits: bits, scalesType: .bf16
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
}
