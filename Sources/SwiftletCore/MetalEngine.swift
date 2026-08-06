import Foundation
import Metal

/// Metal device/queue/pipeline owner for the GPU runtime. The convenience
/// methods here are blocking (dispatch + wait) and copy in/out; the streaming
/// runtime will manage persistent buffers itself.
public final class MetalEngine {
    public enum Error: Swift.Error {
        case noDevice
        case kernelMissing(String)
    }

    public let device: MTLDevice
    public let queue: MTLCommandQueue
    let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    /// A/B switch: SWIFTLET_NO_FAST_GEMV=1 forces the scalar gemv kernel.
    static let fastGemvEnabled =
        ProcessInfo.processInfo.environment["SWIFTLET_NO_FAST_GEMV"] != "1"

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { throw Error.noDevice }
        self.device = device
        self.queue = queue
        // Runtime-compiled from bundled source (TurboFieldfare's pattern):
        // plain `swift build` doesn't produce a metallib, and this keeps
        // shader edits a rebuild away with no Xcode step. Shipped as .txt so
        // xcodebuild (iOS) doesn't require the Metal build toolchain.
        guard let url = Bundle.module.url(forResource: "Kernels.metal", withExtension: "txt") else {
            throw Error.kernelMissing("Kernels.metal.txt resource")
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        library = try device.makeLibrary(source: source, options: nil)
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        if let p = pipelines[name] { return p }
        guard let fn = library.makeFunction(name: name) else { throw Error.kernelMissing(name) }
        let p = try device.makeComputePipelineState(function: fn)
        pipelines[name] = p
        return p
    }

    struct GemvParams {
        var wOff: UInt64 = 0   // byte offsets, 64-bit (shards exceed 4 GB)
        var sOff: UInt64 = 0
        var bOff: UInt64 = 0
        var outDim: UInt32
        var inDim: UInt32
        var groupSize: UInt32
        var bits: UInt32
        var scalesType: UInt32
        var yOff: UInt32 = 0
        var xOff: UInt32 = 0
    }

    struct GemvPlainParams {
        var wOff: UInt64 = 0
        var outDim: UInt32
        var inDim: UInt32
        var dtype: UInt32
        var yOff: UInt32 = 0
        var xOff: UInt32 = 0
    }

    public enum ScalesType: UInt32 {
        case f32 = 0, f16 = 1, bf16 = 2

        public init?(dtype: String) {
            switch dtype {
            case "F32": self = .f32
            case "F16": self = .f16
            case "BF16": self = .bf16
            default: return nil
            }
        }
    }

    func makeBuffer<T>(_ array: [T]) -> MTLBuffer {
        array.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
    }

    func makeBuffer(_ data: Data) -> MTLBuffer {
        data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 1))! }
    }

    // MARK: - Encoder-level API (persistent-buffer runtime path)

    /// Encode one GEMV over a shard-resident linear. `wExtra`/`sExtra`/`bExtra`
    /// advance into stacked tensors (expert slicing), in the same units as the
    /// descriptor offsets. Output rows land at `y[yOff...]`.
    func encodeGemv(
        _ enc: MTLComputeCommandEncoder,
        _ lin: MetalShardStore.GPULinear,
        rows: Int? = nil,
        wExtra: Int = 0, sExtra: Int = 0, bExtra: Int = 0,
        x: MTLBuffer, xOff: Int = 0, y: MTLBuffer, yOff: Int
    ) throws {
        let outDim = rows ?? lin.outDim
        if lin.isQuantized {
            // Cooperative fast path: one simdgroup per row, vectorized loads.
            // Needs 4-byte-aligned rows (qpack blobs and resident buffers are).
            // SWIFTLET_NO_FAST_GEMV=1 forces the scalar kernel (A/B debugging).
            let aligned = (lin.wOff + wExtra) % 4 == 0
            let fastName: String? = !Self.fastGemvEnabled || !aligned ? nil
                : lin.bits == 4 && lin.groupSize % 8 == 0 ? "gemv_affine_fast"
                : lin.bits == 8 && lin.groupSize % 4 == 0 ? "gemv_affine_fast8"
                : nil
            if let fastName {
                let pipe = try pipeline(fastName)
                enc.setComputePipelineState(pipe)
                var p = GemvParams(
                    wOff: UInt64(lin.wOff + wExtra), sOff: UInt64(lin.sOff + sExtra),
                    bOff: UInt64(lin.bOff + bExtra),
                    outDim: UInt32(outDim), inDim: UInt32(lin.inDim),
                    groupSize: UInt32(lin.groupSize), bits: UInt32(lin.bits),
                    scalesType: lin.scalesType, yOff: UInt32(yOff), xOff: UInt32(xOff)
                )
                enc.setBuffer(x, offset: 0, index: 0)
                enc.setBuffer(lin.wBuffer, offset: 0, index: 1)
                enc.setBuffer(lin.sBuffer, offset: 0, index: 2)
                enc.setBuffer(lin.bBuffer, offset: 0, index: 3)
                enc.setBuffer(y, offset: 0, index: 4)
                enc.setBytes(&p, length: MemoryLayout<GemvParams>.stride, index: 5)
                enc.dispatchThreads(
                    MTLSize(width: 32, height: outDim, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
                return
            }
            let pipe = try pipeline("gemv_affine")
            enc.setComputePipelineState(pipe)
            var p = GemvParams(
                wOff: UInt64(lin.wOff + wExtra), sOff: UInt64(lin.sOff + sExtra),
                bOff: UInt64(lin.bOff + bExtra),
                outDim: UInt32(outDim), inDim: UInt32(lin.inDim),
                groupSize: UInt32(lin.groupSize), bits: UInt32(lin.bits),
                scalesType: lin.scalesType, yOff: UInt32(yOff), xOff: UInt32(xOff)
            )
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(lin.wBuffer, offset: 0, index: 1)
            enc.setBuffer(lin.sBuffer, offset: 0, index: 2)
            enc.setBuffer(lin.bBuffer, offset: 0, index: 3)
            enc.setBuffer(y, offset: 0, index: 4)
            enc.setBytes(&p, length: MemoryLayout<GemvParams>.stride, index: 5)
            enc.dispatchThreads(
                MTLSize(width: outDim, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(64, outDim), height: 1, depth: 1)
            )
        } else {
            let pipe = try pipeline("gemv_plain")
            enc.setComputePipelineState(pipe)
            var p = GemvPlainParams(
                wOff: UInt64(lin.wOff + wExtra),
                outDim: UInt32(outDim), inDim: UInt32(lin.inDim),
                dtype: lin.plainDtype, yOff: UInt32(yOff), xOff: UInt32(xOff)
            )
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(lin.wBuffer, offset: 0, index: 1)
            enc.setBuffer(y, offset: 0, index: 2)
            enc.setBytes(&p, length: MemoryLayout<GemvPlainParams>.stride, index: 3)
            enc.dispatchThreads(
                MTLSize(width: outDim, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(64, outDim), height: 1, depth: 1)
            )
        }
    }

    func encodeSiluMul(
        _ enc: MTLComputeCommandEncoder,
        buf: MTLBuffer, count: Int, gOff: Int, uOff: Int, dstOff: Int
    ) throws {
        let pipe = try pipeline("silu_mul")
        enc.setComputePipelineState(pipe)
        var p = SIMD4<UInt32>(UInt32(count), UInt32(gOff), UInt32(uOff), UInt32(dstOff))
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBuffer(buf, offset: 0, index: 1)
        enc.setBuffer(buf, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 3)
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, count), height: 1, depth: 1)
        )
    }

    /// Blocking quantized GEMV: y = W x with MLX affine layout.
    public func gemvQuantized(
        x: [Float], packed: Data, scales: Data, biases: Data,
        outDim: Int, inDim: Int, groupSize: Int, bits: Int, scalesType: ScalesType,
        useFast: Bool = false
    ) throws -> [Float] {
        let pipe = try pipeline(
            useFast ? (bits == 4 ? "gemv_affine_fast" : "gemv_affine_fast8") : "gemv_affine")
        var params = GemvParams(
            outDim: UInt32(outDim), inDim: UInt32(inDim),
            groupSize: UInt32(groupSize), bits: UInt32(bits),
            scalesType: scalesType.rawValue, yOff: 0, xOff: 0
        )
        let yBuf = device.makeBuffer(length: outDim * 4)!

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        enc.setBuffer(makeBuffer(x), offset: 0, index: 0)
        enc.setBuffer(makeBuffer(packed), offset: 0, index: 1)
        enc.setBuffer(makeBuffer(scales), offset: 0, index: 2)
        enc.setBuffer(makeBuffer(biases), offset: 0, index: 3)
        enc.setBuffer(yBuf, offset: 0, index: 4)
        enc.setBytes(&params, length: MemoryLayout<GemvParams>.stride, index: 5)
        if useFast {
            enc.dispatchThreads(
                MTLSize(width: 32, height: outDim, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        } else {
            let tg = min(pipe.maxTotalThreadsPerThreadgroup, 64)
            enc.dispatchThreads(
                MTLSize(width: outDim, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
            )
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        return Array(UnsafeBufferPointer(
            start: yBuf.contents().bindMemory(to: Float.self, capacity: outDim), count: outDim
        ))
    }

    struct DeltaParams {
        var T: UInt32
        var Hk: UInt32
        var Hv: UInt32
        var Dk: UInt32
        var Dv: UInt32
    }

    /// Blocking gated-delta recurrence over T steps (B = 1). Returns y
    /// [T, Hv, Dv] and the updated state [Hv, Dv, Dk].
    public func gatedDeltaStep(
        q: [Float], k: [Float], v: [Float], g: [Float], beta: [Float],
        state: [Float], T: Int, Hk: Int, Hv: Int, Dk: Int, Dv: Int
    ) throws -> (y: [Float], state: [Float]) {
        precondition(Dk % 32 == 0 && Dk <= 256 && Dv % 4 == 0)
        let pipe = try pipeline("gated_delta_step")
        var params = DeltaParams(T: UInt32(T), Hk: UInt32(Hk), Hv: UInt32(Hv), Dk: UInt32(Dk), Dv: UInt32(Dv))
        let yBuf = device.makeBuffer(length: T * Hv * Dv * 4)!
        let stateOut = device.makeBuffer(length: state.count * 4)!

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        for (i, arr) in [q, k, v, g, beta, state].enumerated() {
            enc.setBuffer(makeBuffer(arr), offset: 0, index: i)
        }
        enc.setBuffer(yBuf, offset: 0, index: 6)
        enc.setBuffer(stateOut, offset: 0, index: 7)
        enc.setBytes(&params, length: MemoryLayout<DeltaParams>.stride, index: 8)
        enc.dispatchThreads(
            MTLSize(width: 32, height: Dv, depth: Hv),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let y = Array(UnsafeBufferPointer(
            start: yBuf.contents().bindMemory(to: Float.self, capacity: T * Hv * Dv), count: T * Hv * Dv
        ))
        let s = Array(UnsafeBufferPointer(
            start: stateOut.contents().bindMemory(to: Float.self, capacity: state.count), count: state.count
        ))
        return (y, s)
    }
}
