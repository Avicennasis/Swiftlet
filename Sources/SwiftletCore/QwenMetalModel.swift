import Foundation
import Metal

/// GPU runtime: every linear (dense projections, router, experts, lm_head)
/// runs as a quantized GEMV directly on mmapped checkpoint bytes — weights are
/// never decompressed. The CPU keeps only glue math (norms, RoPE, conv step,
/// delta recurrence, attention softmax over cached KV, top-k routing), all of
/// it microseconds per token. Numerics mirror QwenCPUModel exactly.
public final class QwenMetalModel {
    /// S3a whole-step aggregate baseline. These totals do not identify phases,
    /// layer costs, overlap, or command-buffer timeline gaps.
    public struct StepMetrics: Equatable, Sendable {
        public let tokensProcessed: Int
        public let logitProjections: Int
        public let commandBuffersCommitted: Int
        public let blockingWaits: Int
        /// Wall time spent inside the existing waitUntilCompleted calls.
        public let blockingWaitSeconds: Double
        /// Compute dispatch calls encoded, including partial work if step throws.
        public let computeDispatchesEncoded: Int
        public let stepWallSeconds: Double
        /// Buffers whose status was not completed after waitUntilCompleted.
        public let commandBufferErrors: Int
        /// Sum of valid per-command-buffer GPU durations, not an elapsed span.
        public let gpuExecutionSeconds: Double
        public let gpuTimedCommandBuffers: Int
        /// Completed buffers with unavailable or invalid GPU timestamps.
        public let gpuUntimedCommandBuffers: Int
        /// False when step exited by throwing after publishing partial counters.
        public let completedWithoutThrow: Bool

        public var avoidedLogitProjections: Int {
            max(0, tokensProcessed - logitProjections)
        }
    }

    private final class StepCounters {
        var tokensProcessed = 0
        var logitProjections = 0
        var commandBuffersCommitted = 0
        var blockingWaits = 0
        var blockingWaitSeconds = 0.0
        var computeDispatchesEncoded = 0
        var commandBufferErrors = 0
        var gpuExecutionSeconds = 0.0
        var gpuTimedCommandBuffers = 0
        var gpuUntimedCommandBuffers = 0
        var completedWithoutThrow = false
    }

    public let config: QwenConfig
    let ckpt: Checkpoint
    let engine: MetalEngine
    let store: MetalShardStore
    typealias GPULinear = MetalShardStore.GPULinear

    struct ExpertStack {
        var lin: GPULinear
        var wStride: Int   // per-expert advance, bytes
        var sStride: Int   // per-expert advance, bytes
        var rows: Int      // rows per expert
    }

    struct AttnGPU {
        var qProj, kProj, vProj, oProj: GPULinear
        var qNorm, kNorm: [Float]
    }

    struct DeltaGPU {
        var qkvz, ba, qkv, zProj, bProj, aProj: GPULinear?
        var conv, dtBias, aLog, norm: [Float]
        var outProj: GPULinear
    }

    struct MoEGPU {
        var gate: GPULinear
        var stacks: (gate: ExpertStack, up: ExpertStack, down: ExpertStack)?
        var sharedGate, sharedUp, sharedDown: GPULinear
        var sharedExpertGate: [Float]
    }

    /// Per-expert projection geometry inside a .qpack blob (offsets in bytes).
    struct ExpertProj {
        var wOff, sOff, bOff: Int
        var outDim, inDim: Int
        var groupSize, bits: Int
        var scalesType: UInt32
        var isQuantized: Bool
        var plainDtype: UInt32

        func linear(on buf: MTLBuffer) -> GPULinear {
            GPULinear(
                wBuffer: buf, sBuffer: buf, bBuffer: buf,
                outDim: outDim, inDim: inDim, isQuantized: isQuantized,
                groupSize: groupSize, bits: bits, scalesType: scalesType,
                wOff: wOff, sOff: sOff, bOff: bOff, plainDtype: plainDtype
            )
        }
    }

    struct LayerGPU {
        var inputNorm, postAttnNorm: [Float]
        var attn: AttnGPU?
        var delta: DeltaGPU?
        var moe: MoEGPU
    }

    var layers: [LayerGPU] = []
    var finalNorm: [Float]
    var lmHead: GPULinear
    public internal(set) var expertCache: ExpertCache?
    var expertProjs: (gate: ExpertProj, up: ExpertProj, down: ExpertProj)?
    let xBuf: MTLBuffer
    let yBuf: MTLBuffer
    let logitsBuf: MTLBuffer
    /// Latest step snapshot. A throwing step publishes its partial counters.
    public private(set) var lastStepMetrics = StepMetrics(
        tokensProcessed: 0, logitProjections: 0,
        commandBuffersCommitted: 0, blockingWaits: 0, blockingWaitSeconds: 0,
        computeDispatchesEncoded: 0, stepWallSeconds: 0, commandBufferErrors: 0,
        gpuExecutionSeconds: 0, gpuTimedCommandBuffers: 0, gpuUntimedCommandBuffers: 0,
        completedWithoutThrow: false
    )
    private var activeStepCounters: StepCounters?

    // MARK: Fast path (split DeltaNet layout): one command buffer per layer.
    struct Regions {
        var x0 = 0, qkv = 0, z = 0, b = 0, a = 0, conv = 0, gb = 0
        var dy = 0, dn = 0, r = 0, xmoe = 0, rout = 0
        var qout = 0, knew = 0, vnew = 0, att = 0
        var qkvzStage = 0, baStage = 0
        var exp = 0, dexp = 0, sh = 0, shg = 0
        var total = 0
    }
    struct FastLayer {
        var inputNorm, postNorm: MTLBuffer
        var convW, aLog, dtBias, deltaNormW: MTLBuffer?
        var hist, state: MTLBuffer?
        var sharedGateLin: GPULinear
    }
    var fastLayers: [FastLayer] = []
    var finalNormBuf: MTLBuffer?
    var sBuf: MTLBuffer?
    var hBuf: MTLBuffer?
    var reg = Regions()
    var boundStateID: ObjectIdentifier?
    struct PendingMoE {
        var bufs: [MTLBuffer]
        var weights: [Float]
        var stacksLayer: Int
        var picks: [(Int, Float)]
    }

    public init(modelDir: URL, cacheBudgetGB: Double = 8) throws {
        config = try QwenConfig(url: modelDir.appendingPathComponent("config.json"))
        ckpt = try Checkpoint(dir: modelDir)
        engine = try MetalEngine()
        store = MetalShardStore(device: engine.device)

        let cfg = config
        let ckpt = self.ckpt
        let store = self.store

        // .qpack container: dense weights copied into resident GPU buffers,
        // experts streamed through the bounded cache. Raw checkpoint dir:
        // everything mmapped (fine when the model fits in RAM).
        let qpackMode = FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("packed_experts/layout.json").path
        )
        func denseLinear(_ path: String) throws -> GPULinear {
            qpackMode
                ? try store.residentLinear(ckpt: ckpt, path: path)
                : try store.gpuLinear(ckpt: ckpt, path: path)
        }

        finalNorm = try ckpt.tensor("model.norm.weight")
        lmHead = try denseLinear(cfg.tieWordEmbeddings ? "model.embed_tokens" : "lm_head")

        if qpackMode {
            let cache = try ExpertCache(
                containerDir: modelDir, device: engine.device,
                budgetBytes: Int(cacheBudgetGB * 1_073_741_824)
            )
            expertCache = cache

            let manifestData = try Data(contentsOf: modelDir.appendingPathComponent("manifest.json"))
            let manifest = try JSONDecoder().decode(Qpack.Manifest.self, from: manifestData)
            func proj(_ name: String, outDim: Int) throws -> ExpertProj {
                guard let w = cache.reader.section(name + ".weight") else {
                    throw Checkpoint.Error.missingTensor("qpack section \(name)")
                }
                if w.dtype == "U32", let bits = manifest.quantBits, let group = manifest.quantGroupSize,
                   let sSec = cache.reader.section(name + ".scales"),
                   let bSec = cache.reader.section(name + ".biases") {
                    return ExpertProj(
                        wOff: w.offset, sOff: sSec.offset, bOff: bSec.offset,
                        outDim: outDim, inDim: w.shape.last! * (32 / bits),
                        groupSize: group, bits: bits,
                        scalesType: MetalEngine.ScalesType(dtype: sSec.dtype)?.rawValue ?? 2,
                        isQuantized: true, plainDtype: 0
                    )
                }
                let dtype: UInt32 = w.dtype == "F32" ? 0 : (w.dtype == "F16" ? 1 : 2)
                return ExpertProj(
                    wOff: w.offset, sOff: 0, bOff: 0,
                    outDim: outDim, inDim: w.shape.last!,
                    groupSize: 0, bits: 0, scalesType: 0,
                    isQuantized: false, plainDtype: dtype
                )
            }
            expertProjs = (
                gate: try proj("gate_proj", outDim: cfg.moeIntermediateSize),
                up: try proj("up_proj", outDim: cfg.moeIntermediateSize),
                down: try proj("down_proj", outDim: cfg.hiddenSize)
            )
        }
        func expertStack(_ path: String, rowsPerExpert: Int) throws -> ExpertStack {
            let lin = try store.gpuLinear(ckpt: ckpt, path: path)
            if lin.isQuantized {
                let perWord = 32 / lin.bits
                let packedCols = lin.inDim / perWord
                let groups = lin.inDim / lin.groupSize
                let scaleBytes = lin.scalesType == 0 ? 4 : 2
                return ExpertStack(
                    lin: lin,
                    wStride: rowsPerExpert * packedCols * 4,      // bytes
                    sStride: rowsPerExpert * groups * scaleBytes, // bytes
                    rows: rowsPerExpert
                )
            }
            let elemBytes = lin.plainDtype == 0 ? 4 : 2
            return ExpertStack(
                lin: lin,
                wStride: rowsPerExpert * lin.inDim * elemBytes,
                sStride: 0,
                rows: rowsPerExpert
            )
        }

        for i in 0..<cfg.numHiddenLayers {
            let p = "model.layers.\(i)."
            let stacks: (ExpertStack, ExpertStack, ExpertStack)? = qpackMode ? nil : (
                try expertStack(p + "mlp.switch_mlp.gate_proj", rowsPerExpert: cfg.moeIntermediateSize),
                try expertStack(p + "mlp.switch_mlp.up_proj", rowsPerExpert: cfg.moeIntermediateSize),
                try expertStack(p + "mlp.switch_mlp.down_proj", rowsPerExpert: cfg.hiddenSize)
            )
            let moe = MoEGPU(
                gate: try denseLinear(p + "mlp.gate"),
                stacks: stacks,
                sharedGate: try denseLinear(p + "mlp.shared_expert.gate_proj"),
                sharedUp: try denseLinear(p + "mlp.shared_expert.up_proj"),
                sharedDown: try denseLinear(p + "mlp.shared_expert.down_proj"),
                sharedExpertGate: try ckpt.moduleWeight(p + "mlp.shared_expert_gate")
            )
            var layer = LayerGPU(
                inputNorm: try ckpt.tensor(p + "input_layernorm.weight"),
                postAttnNorm: try ckpt.tensor(p + "post_attention_layernorm.weight"),
                moe: moe
            )
            if cfg.isLinearLayer(i) {
                var d = DeltaGPU(
                    conv: try ckpt.tensor(p + "linear_attn.conv1d.weight"),
                    dtBias: try ckpt.tensor(p + "linear_attn.dt_bias"),
                    aLog: try ckpt.tensor(p + "linear_attn.A_log"),
                    norm: try ckpt.tensor(p + "linear_attn.norm.weight"),
                    outProj: try denseLinear(p + "linear_attn.out_proj")
                )
                switch cfg.deltaLayout {
                case .fusedInterleaved:
                    d.qkvz = try denseLinear(p + "linear_attn.in_proj_qkvz")
                    d.ba = try denseLinear(p + "linear_attn.in_proj_ba")
                case .split:
                    d.qkv = try denseLinear(p + "linear_attn.in_proj_qkv")
                    d.zProj = try denseLinear(p + "linear_attn.in_proj_z")
                    d.bProj = try denseLinear(p + "linear_attn.in_proj_b")
                    d.aProj = try denseLinear(p + "linear_attn.in_proj_a")
                }
                layer.delta = d
            } else {
                layer.attn = AttnGPU(
                    qProj: try denseLinear(p + "self_attn.q_proj"),
                    kProj: try denseLinear(p + "self_attn.k_proj"),
                    vProj: try denseLinear(p + "self_attn.v_proj"),
                    oProj: try denseLinear(p + "self_attn.o_proj"),
                    qNorm: try ckpt.tensor(p + "self_attn.q_norm.weight"),
                    kNorm: try ckpt.tensor(p + "self_attn.k_norm.weight")
                )
            }
            layers.append(layer)
        }

        let maxIn = max(cfg.hiddenSize, cfg.valueDim, cfg.numAttentionHeads * cfg.headDim, cfg.convDim)
        xBuf = engine.device.makeBuffer(length: maxIn * 4, options: .storageModeShared)!
        let moeScratch = 3 * cfg.numExpertsPerTok * cfg.moeIntermediateSize
            + cfg.numExpertsPerTok * cfg.hiddenSize
            + 3 * cfg.sharedExpertIntermediateSize + cfg.hiddenSize
            + cfg.numExperts + 64
        let yLen = max(
            2 * cfg.keyDim + 2 * cfg.valueDim + 2 * cfg.linearNumValueHeads + 64,
            moeScratch,
            2 * cfg.numAttentionHeads * cfg.headDim + 2 * cfg.numKeyValueHeads * cfg.headDim + 64
        )
        yBuf = engine.device.makeBuffer(length: yLen * 4, options: .storageModeShared)!
        logitsBuf = engine.device.makeBuffer(length: cfg.vocabSize * 4, options: .storageModeShared)!

        print("[QwenMetalModel] descriptors ready; building fast path...")
        try setupFastPath(denseLinear: denseLinear)
        print("[QwenMetalModel] fast path ready")
    }

    private func setupFastPath(denseLinear: (String) throws -> GPULinear) throws {
        let cfg = config
        let D = cfg.hiddenSize, E = cfg.numExperts, nv = cfg.linearNumValueHeads
        let H = cfg.numAttentionHeads, hd = cfg.headDim, KVH = cfg.numKeyValueHeads
        let inter = cfg.moeIntermediateSize, shInter = cfg.sharedExpertIntermediateSize
        let K = cfg.numExpertsPerTok

        var o = Regions()
        var c = 0
        func take(_ n: Int) -> Int { let v = c; c += n; return v }
        o.x0 = take(D)
        o.qkv = take(cfg.convDim)
        o.z = take(cfg.valueDim)
        o.b = take(nv)
        o.a = take(nv)
        o.conv = take(cfg.convDim)
        o.gb = take(2 * nv)
        o.dy = take(cfg.valueDim)
        o.dn = take(cfg.valueDim)
        o.r = take(D)
        o.xmoe = take(D)
        o.rout = take(E)
        o.qout = take(2 * H * hd)
        o.knew = take(KVH * hd)
        o.vnew = take(KVH * hd)
        o.att = take(H * hd)
        o.qkvzStage = take(2 * cfg.keyDim + 2 * cfg.valueDim)
        o.baStage = take(2 * nv)
        o.exp = take(3 * K * inter)
        o.dexp = take(K * D)
        o.sh = take(3 * shInter + D)
        o.shg = take(4)
        o.total = c
        reg = o

        let opts: MTLResourceOptions = [.storageModeShared, .hazardTrackingModeUntracked]
        sBuf = engine.device.makeBuffer(length: o.total * 4, options: opts)
        hBuf = engine.device.makeBuffer(length: D * 4, options: opts)
        finalNormBuf = engine.makeBuffer(finalNorm)

        for i in 0..<cfg.numHiddenLayers {
            let L = layers[i]
            let p = "model.layers.\(i)."
            var fl = FastLayer(
                inputNorm: engine.makeBuffer(L.inputNorm),
                postNorm: engine.makeBuffer(L.postAttnNorm),
                sharedGateLin: try denseLinear(p + "mlp.shared_expert_gate")
            )
            if let d = L.delta {
                fl.convW = engine.makeBuffer(d.conv)
                fl.aLog = engine.makeBuffer(d.aLog)
                fl.dtBias = engine.makeBuffer(d.dtBias)
                fl.deltaNormW = engine.makeBuffer(d.norm)
                fl.hist = engine.device.makeBuffer(length: (cfg.linearConvKernelDim - 1) * cfg.convDim * 4, options: opts)
                fl.state = engine.device.makeBuffer(
                    length: nv * cfg.linearValueHeadDim * cfg.linearKeyHeadDim * 4, options: opts)
            }
            fastLayers.append(fl)
        }
    }

    /// Replaces the expert cache with a smaller one (old slots free
    /// immediately; the new cache refills lazily). Memory-pressure valve.
    public func shrinkCache(toGB gb: Double) {
        guard expertCache != nil else { return }
        // Int(Double) traps on NaN, infinity, and out-of-range values; a
        // pressure valve must refuse such a request, not crash on it.
        let bytes = gb * 1_073_741_824
        guard bytes >= 0, let budget = Int(exactly: bytes.rounded(.down)) else { return }
        guard let replacement = try? ExpertCache(
            containerDir: ckpt.dir, device: engine.device, budgetBytes: budget
        ) else { return }
        expertCache = replacement
    }

    // MARK: - GPU phase helper

    private func loadX(_ v: [Float]) {
        v.withUnsafeBufferPointer {
            xBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: v.count * 4)
        }
    }

    private func commitAndWait(_ cb: MTLCommandBuffer) {
        cb.commit()
        activeStepCounters?.commandBuffersCommitted += 1
        let waitStart = ProcessInfo.processInfo.systemUptime
        cb.waitUntilCompleted()
        activeStepCounters?.blockingWaits += 1
        activeStepCounters?.blockingWaitSeconds += max(
            0, ProcessInfo.processInfo.systemUptime - waitStart
        )

        guard cb.status == .completed else {
            activeStepCounters?.commandBufferErrors += 1
            return
        }
        let start = cb.gpuStartTime
        let end = cb.gpuEndTime
        guard start.isFinite, end.isFinite, start > 0, end > 0, end >= start else {
            activeStepCounters?.gpuUntimedCommandBuffers += 1
            return
        }
        activeStepCounters?.gpuExecutionSeconds += end - start
        activeStepCounters?.gpuTimedCommandBuffers += 1
    }

    private func runPhase(_ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        let cb = engine.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try body(enc)
        enc.endEncoding()
        commitAndWait(cb)
    }

    private func readY(_ offset: Int, _ count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: yBuf.contents().advanced(by: offset * 4).bindMemory(to: Float.self, capacity: count),
            count: count
        ))
    }

    // MARK: - Step

    /// Incremental step matching QwenCPUModel.step semantics.
    public func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float] {
        let counters = StepCounters()
        let wallStart = ProcessInfo.processInfo.systemUptime
        activeStepCounters = counters
        engine.computeDispatchObserver = { counters.computeDispatchesEncoded += 1 }
        defer {
            engine.computeDispatchObserver = nil
            activeStepCounters = nil
            lastStepMetrics = StepMetrics(
                tokensProcessed: counters.tokensProcessed,
                logitProjections: counters.logitProjections,
                commandBuffersCommitted: counters.commandBuffersCommitted,
                blockingWaits: counters.blockingWaits,
                blockingWaitSeconds: counters.blockingWaitSeconds,
                computeDispatchesEncoded: counters.computeDispatchesEncoded,
                stepWallSeconds: max(0, ProcessInfo.processInfo.systemUptime - wallStart),
                commandBufferErrors: counters.commandBufferErrors,
                gpuExecutionSeconds: counters.gpuExecutionSeconds,
                gpuTimedCommandBuffers: counters.gpuTimedCommandBuffers,
                gpuUntimedCommandBuffers: counters.gpuUntimedCommandBuffers,
                completedWithoutThrow: counters.completedWithoutThrow
            )
        }

        var logits: [Float] = []
        for (index, t) in tokens.enumerated() {
            // S1a LM-head elision: a multi-token call consumes only the final
            // position's logits, so intermediate vocabulary projections are
            // unnecessary.
            let projectLogits = index == tokens.count - 1
            logits = try stepOne(t, state: state, projectLogits: projectLogits)
            state.position += 1
            counters.tokensProcessed += 1
        }
        counters.completedWithoutThrow = true
        return logits
    }

    private func stepOne(
        _ token: Int, state: QwenCPUModel.DecodeState, projectLogits: Bool
    ) throws -> [Float] {
        if sBuf != nil {
            return try stepOneFast(token, state: state, projectLogits: projectLogits)
        }
        let cfg = config
        let D = cfg.hiddenSize
        var h = try ckpt.moduleWeightSlice("model.embed_tokens", rowRange: token..<(token + 1))
        precondition(h.count == D)

        for li in 0..<cfg.numHiddenLayers {
            let layer = layers[li]
            var x = h
            QwenCPUModel.rmsNorm(&x, rows: 1, dim: D, weight: layer.inputNorm, eps: Float(cfg.rmsNormEps))

            let r: [Float]
            if let delta = layer.delta {
                r = try deltaForward(x, w: delta, layerIndex: li, state: state)
            } else {
                r = try attnForward(x, w: layer.attn!, layerIndex: li, state: state)
            }
            for i in 0..<D { h[i] += r[i] }

            var x2 = h
            QwenCPUModel.rmsNorm(&x2, rows: 1, dim: D, weight: layer.postAttnNorm, eps: Float(cfg.rmsNormEps))
            let m = try moeForward(x2, w: layer.moe, layerIndex: li)
            for i in 0..<D { h[i] += m[i] }
        }

        guard projectLogits else { return [] }
        QwenCPUModel.rmsNorm(&h, rows: 1, dim: D, weight: finalNorm, eps: Float(cfg.rmsNormEps))
        loadX(h)
        try runPhase { enc in
            try engine.encodeGemv(enc, lmHead, x: xBuf, y: logitsBuf, yOff: 0)
        }
        activeStepCounters?.logitProjections += 1
        return Array(UnsafeBufferPointer(
            start: logitsBuf.contents().bindMemory(to: Float.self, capacity: cfg.vocabSize),
            count: cfg.vocabSize
        ))
    }

    // MARK: - Attention (GQA decode, KV on CPU)

    private func attnForward(_ x: [Float], w: AttnGPU, layerIndex: Int, state: QwenCPUModel.DecodeState) throws -> [Float] {
        let cfg = config
        let H = cfg.numAttentionHeads, KVH = cfg.numKeyValueHeads, hd = cfg.headDim
        let eps = Float(cfg.rmsNormEps)
        let past = state.position
        let qOutDim = H * hd * 2
        let kvDim = KVH * hd

        loadX(x)
        try runPhase { enc in
            try engine.encodeGemv(enc, w.qProj, x: xBuf, y: yBuf, yOff: 0)
            try engine.encodeGemv(enc, w.kProj, x: xBuf, y: yBuf, yOff: qOutDim)
            try engine.encodeGemv(enc, w.vProj, x: xBuf, y: yBuf, yOff: qOutDim + kvDim)
        }
        let qOut = readY(0, qOutDim)
        var k = readY(qOutDim, kvDim)
        var v = readY(qOutDim + kvDim, kvDim)

        var q = [Float](repeating: 0, count: H * hd)
        var gate = [Float](repeating: 0, count: H * hd)
        for head in 0..<H {
            for i in 0..<hd {
                q[head * hd + i] = qOut[head * 2 * hd + i]
                gate[head * hd + i] = qOut[head * 2 * hd + hd + i]
            }
        }
        QwenCPUModel.rmsNorm(&q, rows: H, dim: hd, weight: w.qNorm, eps: eps)
        QwenCPUModel.rmsNorm(&k, rows: KVH, dim: hd, weight: w.kNorm, eps: eps)
        applyRope(&q, heads: H, position: past)
        applyRope(&k, heads: KVH, position: past)

        var cache = state.kv[layerIndex] ?? (k: [], v: [])
        cache.k.append(contentsOf: k)
        cache.v.append(contentsOf: v)
        state.kv[layerIndex] = cache
        let kAll = cache.k, vAll = cache.v
        let kvLen = past + 1

        let scale = 1 / Float(hd).squareRoot()
        var attnOut = [Float](repeating: 0, count: H * hd)
        let group = H / KVH
        var scores = [Float](repeating: 0, count: kvLen)
        for head in 0..<H {
            let kvHead = head / group
            for sj in 0..<kvLen {
                var dot: Float = 0
                for i in 0..<hd { dot += q[head * hd + i] * kAll[(sj * KVH + kvHead) * hd + i] }
                scores[sj] = dot * scale
            }
            QwenCPUModel.softmaxRow(&scores, base: 0, count: kvLen)
            for sj in 0..<kvLen {
                let p = scores[sj]
                let vBase = (sj * KVH + kvHead) * hd
                for i in 0..<hd { attnOut[head * hd + i] += p * vAll[vBase + i] }
            }
        }
        for i in 0..<attnOut.count { attnOut[i] *= QwenCPUModel.sigmoid(gate[i]) }

        loadX(attnOut)
        try runPhase { enc in
            try engine.encodeGemv(enc, w.oProj, x: xBuf, y: yBuf, yOff: 0)
        }
        return readY(0, cfg.hiddenSize)
    }

    private func applyRope(_ x: inout [Float], heads: Int, position: Int) {
        let hd = config.headDim
        let rot = config.rotaryDims
        let half = rot / 2
        for head in 0..<heads {
            let base = head * hd
            for j in 0..<half {
                let invFreq = powf(Float(config.ropeTheta), -Float(2 * j) / Float(rot))
                let angle = Float(position) * invFreq
                let c = cosf(angle), sn = sinf(angle)
                let a = x[base + j]
                let b = x[base + half + j]
                x[base + j] = a * c - b * sn
                x[base + half + j] = b * c + a * sn
            }
        }
    }

    // MARK: - Gated DeltaNet (recurrence on CPU, projections on GPU)

    private func deltaForward(_ x: [Float], w: DeltaGPU, layerIndex: Int, state: QwenCPUModel.DecodeState) throws -> [Float] {
        let cfg = config
        let nk = cfg.linearNumKeyHeads, nv = cfg.linearNumValueHeads
        let dk = cfg.linearKeyHeadDim, dv = cfg.linearValueHeadDim
        let rep = nv / nk
        let keyDim = cfg.keyDim, valueDim = cfg.valueDim, convDim = cfg.convDim
        let K = cfg.linearConvKernelDim

        var mixedQKV: [Float]
        var z: [Float]
        var bArr: [Float]
        var aArr: [Float]

        loadX(x)
        switch cfg.deltaLayout {
        case .fusedInterleaved:
            let qkvzDim = 2 * keyDim + 2 * valueDim
            try runPhase { enc in
                try engine.encodeGemv(enc, w.qkvz!, x: xBuf, y: yBuf, yOff: 0)
                try engine.encodeGemv(enc, w.ba!, x: xBuf, y: yBuf, yOff: qkvzDim)
            }
            let qkvz = readY(0, qkvzDim)
            let ba = readY(qkvzDim, 2 * nv)
            let chunk = 2 * dk + 2 * rep * dv
            mixedQKV = [Float](repeating: 0, count: convDim)
            z = [Float](repeating: 0, count: valueDim)
            bArr = [Float](repeating: 0, count: nv)
            aArr = [Float](repeating: 0, count: nv)
            for hk in 0..<nk {
                let src = hk * chunk
                for i in 0..<dk {
                    mixedQKV[hk * dk + i] = qkvz[src + i]
                    mixedQKV[keyDim + hk * dk + i] = qkvz[src + dk + i]
                }
                for ri in 0..<rep {
                    let hv = hk * rep + ri
                    for i in 0..<dv {
                        mixedQKV[2 * keyDim + hv * dv + i] = qkvz[src + 2 * dk + ri * dv + i]
                        z[hv * dv + i] = qkvz[src + 2 * dk + rep * dv + ri * dv + i]
                    }
                }
                for ri in 0..<rep {
                    bArr[hk * rep + ri] = ba[hk * 2 * rep + ri]
                    aArr[hk * rep + ri] = ba[hk * 2 * rep + rep + ri]
                }
            }
        case .split:
            try runPhase { enc in
                try engine.encodeGemv(enc, w.qkv!, x: xBuf, y: yBuf, yOff: 0)
                try engine.encodeGemv(enc, w.zProj!, x: xBuf, y: yBuf, yOff: convDim)
                try engine.encodeGemv(enc, w.bProj!, x: xBuf, y: yBuf, yOff: convDim + valueDim)
                try engine.encodeGemv(enc, w.aProj!, x: xBuf, y: yBuf, yOff: convDim + valueDim + nv)
            }
            mixedQKV = readY(0, convDim)
            z = readY(convDim, valueDim)
            bArr = readY(convDim + valueDim, nv)
            aArr = readY(convDim + valueDim + nv, nv)
        }

        // Conv step + recurrence: identical math to QwenCPUModel, S = 1.
        let tailRows = K - 1
        var padded = state.convTail[layerIndex] ?? [Float](repeating: 0, count: tailRows * convDim)
        padded.append(contentsOf: mixedQKV)
        var convOut = [Float](repeating: 0, count: convDim)
        for c in 0..<convDim {
            var acc: Float = 0
            for j in 0..<K { acc += w.conv[c * K + j] * padded[j * convDim + c] }
            convOut[c] = QwenCPUModel.silu(acc)
        }
        state.convTail[layerIndex] = Array(padded.suffix(tailRows * convDim))

        var qh = Array(convOut[0..<keyDim])
        var kh = Array(convOut[keyDim..<2 * keyDim])
        let vh = Array(convOut[2 * keyDim..<convDim])
        QwenCPUModel.rmsNorm(&qh, rows: nk, dim: dk, weight: nil, eps: 1e-6)
        QwenCPUModel.rmsNorm(&kh, rows: nk, dim: dk, weight: nil, eps: 1e-6)
        let invScale = 1 / Float(dk).squareRoot()
        for i in 0..<qh.count { qh[i] *= invScale * invScale }
        for i in 0..<kh.count { kh[i] *= invScale }

        var delta0 = state.deltaState[layerIndex] ?? [Float](repeating: 0, count: nv * dv * dk)
        var out = [Float](repeating: 0, count: valueDim)
        for hv in 0..<nv {
            let hk = hv / rep
            let g = expf(-expf(w.aLog[hv]) * QwenCPUModel.softplus(aArr[hv] + w.dtBias[hv]))
            let beta = QwenCPUModel.sigmoid(bArr[hv])
            for dvi in 0..<dv {
                let stBase = (hv * dv + dvi) * dk
                var kvMem: Float = 0
                for dki in 0..<dk {
                    delta0[stBase + dki] *= g
                    kvMem += delta0[stBase + dki] * kh[hk * dk + dki]
                }
                let d = (vh[hv * dv + dvi] - kvMem) * beta
                var yv: Float = 0
                for dki in 0..<dk {
                    delta0[stBase + dki] += kh[hk * dk + dki] * d
                    yv += delta0[stBase + dki] * qh[hk * dk + dki]
                }
                out[hv * dv + dvi] = yv
            }
        }
        state.deltaState[layerIndex] = delta0

        var normed = out
        QwenCPUModel.rmsNorm(&normed, rows: nv, dim: dv, weight: w.norm, eps: Float(cfg.rmsNormEps))
        for i in 0..<normed.count { normed[i] *= QwenCPUModel.silu(z[i]) }

        loadX(normed)
        try runPhase { enc in
            try engine.encodeGemv(enc, w.outProj, x: xBuf, y: yBuf, yOff: 0)
        }
        return readY(0, cfg.hiddenSize)
    }

    // MARK: - MoE (router + experts on GPU)

    private func moeForward(_ x: [Float], w: MoEGPU, layerIndex: Int) throws -> [Float] {
        let cfg = config
        let D = cfg.hiddenSize, E = cfg.numExperts, inter = cfg.moeIntermediateSize
        let sharedInter = cfg.sharedExpertIntermediateSize
        let topK = cfg.numExpertsPerTok

        loadX(x)
        try runPhase { enc in
            try engine.encodeGemv(enc, w.gate, x: xBuf, y: yBuf, yOff: 0)
        }
        var router = readY(0, E)
        QwenCPUModel.softmaxRow(&router, base: 0, count: E)
        var picks: [(Int, Float)] = []
        for e in 0..<E {
            let p = router[e]
            if picks.count < topK {
                picks.append((e, p))
                picks.sort { $0.1 > $1.1 }
            } else if p > picks[topK - 1].1 {
                picks[topK - 1] = (e, p)
                picks.sort { $0.1 > $1.1 }
            }
        }

        // Scratch layout in yBuf (floats):
        //   [0, K*inter)                        expert gate outputs
        //   [gBase2, +K*inter)                  expert up outputs
        //   [hBase, +K*inter)                   silu(g)*u
        //   [dBase, +K*D)                       expert down outputs
        //   [shBase, +2*sharedInter+sharedInter+D) shared expert chain
        let K = picks.count
        let gBase2 = K * inter
        let hBase = 2 * K * inter
        let dBase = 3 * K * inter
        let shBase = dBase + K * D

        // Fetch expert buffers up front (qpack path: preads on cache misses).
        var expertBufs: [MTLBuffer] = []
        if let cache = expertCache {
            expertBufs = try cache.buffers(layer: layerIndex, experts: picks.map { $0.0 })
        }

        try runPhase { enc in
            for (ki, pick) in picks.enumerated() {
                let e = pick.0
                if let projs = expertProjs {
                    let buf = expertBufs[ki]
                    try engine.encodeGemv(enc, projs.gate.linear(on: buf), x: xBuf, y: yBuf, yOff: ki * inter)
                    try engine.encodeGemv(enc, projs.up.linear(on: buf), x: xBuf, y: yBuf, yOff: gBase2 + ki * inter)
                    try engine.encodeSiluMul(
                        enc, buf: yBuf, count: inter,
                        gOff: ki * inter, uOff: gBase2 + ki * inter, dstOff: hBase + ki * inter
                    )
                    try engine.encodeGemv(
                        enc, projs.down.linear(on: buf),
                        x: yBuf, xOff: hBase + ki * inter, y: yBuf, yOff: dBase + ki * D
                    )
                    continue
                }
                let st = w.stacks!
                try engine.encodeGemv(
                    enc, st.gate.lin, rows: inter,
                    wExtra: e * st.gate.wStride, sExtra: e * st.gate.sStride, bExtra: e * st.gate.sStride,
                    x: xBuf, y: yBuf, yOff: ki * inter
                )
                try engine.encodeGemv(
                    enc, st.up.lin, rows: inter,
                    wExtra: e * st.up.wStride, sExtra: e * st.up.sStride, bExtra: e * st.up.sStride,
                    x: xBuf, y: yBuf, yOff: gBase2 + ki * inter
                )
                try engine.encodeSiluMul(
                    enc, buf: yBuf, count: inter,
                    gOff: ki * inter, uOff: gBase2 + ki * inter, dstOff: hBase + ki * inter
                )
                try engine.encodeGemv(
                    enc, st.down.lin, rows: D,
                    wExtra: e * st.down.wStride, sExtra: e * st.down.sStride, bExtra: e * st.down.sStride,
                    x: yBuf, xOff: hBase + ki * inter, y: yBuf, yOff: dBase + ki * D
                )
            }
            // Shared expert chain.
            try engine.encodeGemv(enc, w.sharedGate, x: xBuf, y: yBuf, yOff: shBase)
            try engine.encodeGemv(enc, w.sharedUp, x: xBuf, y: yBuf, yOff: shBase + sharedInter)
            try engine.encodeSiluMul(
                enc, buf: yBuf, count: sharedInter,
                gOff: shBase, uOff: shBase + sharedInter, dstOff: shBase + 2 * sharedInter
            )
            try engine.encodeGemv(
                enc, w.sharedDown, x: yBuf, xOff: shBase + 2 * sharedInter,
                y: yBuf, yOff: shBase + 3 * sharedInter
            )
        }

        var scoreSum: Float = 0
        for (_, p) in picks { scoreSum += p }
        var out = [Float](repeating: 0, count: D)
        for (ki, pick) in picks.enumerated() {
            let weight = cfg.normTopkProb ? pick.1 / scoreSum : pick.1
            let dOut = readY(dBase + ki * D, D)
            for i in 0..<D { out[i] += weight * dOut[i] }
        }
        var sg: Float = 0
        for i in 0..<D { sg += w.sharedExpertGate[i] * x[i] }
        let sharedScale = QwenCPUModel.sigmoid(sg)
        let shOut = readY(shBase + 3 * sharedInter, D)
        for i in 0..<D { out[i] += sharedScale * shOut[i] }
        return out
    }
}


// MARK: - Fast path: one command buffer per DeltaNet layer, two per attention
// layer, explicit barriers on an untracked scratch buffer so independent
// dispatches (e.g. all expert GEMVs) actually run in parallel.
extension QwenMetalModel {

    private func setBytesParams<T>(_ enc: MTLComputeCommandEncoder, _ v: inout T, index: Int) {
        enc.setBytes(&v, length: MemoryLayout<T>.stride, index: index)
    }

    private func dispatchRows(_ enc: MTLComputeCommandEncoder, rows: Int) {
        engine.dispatchThreads(
            enc, threads: MTLSize(width: 32, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    private func dispatchN(_ enc: MTLComputeCommandEncoder, _ n: Int) {
        engine.dispatchThreads(
            enc, threads: MTLSize(width: n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, n), height: 1, depth: 1)
        )
    }

    struct NormParams {
        var rows: UInt32
        var dim: UInt32
        var eps: Float
        var hasWeight: UInt32
        var scale: Float
        var off: UInt32
    }

    private func encNormCopy(_ enc: MTLComputeCommandEncoder, w: MTLBuffer, dstOff: Int) throws {
        enc.setComputePipelineState(try engine.pipeline("norm_copy"))
        var p = NormParams(rows: 1, dim: UInt32(config.hiddenSize), eps: Float(config.rmsNormEps),
                           hasWeight: 1, scale: 1, off: UInt32(dstOff))
        enc.setBuffer(hBuf!, offset: 0, index: 0)
        enc.setBuffer(sBuf!, offset: 0, index: 1)
        enc.setBuffer(w, offset: 0, index: 2)
        setBytesParams(enc, &p, index: 3)
        dispatchRows(enc, rows: 1)
    }

    private func encRMSNorm(_ enc: MTLComputeCommandEncoder, off: Int, rows: Int, dim: Int,
                            w: MTLBuffer?, scale: Float, eps: Float) throws {
        enc.setComputePipelineState(try engine.pipeline("rmsnorm_rows"))
        var p = NormParams(rows: UInt32(rows), dim: UInt32(dim), eps: eps,
                           hasWeight: w != nil ? 1 : 0, scale: scale, off: UInt32(off))
        enc.setBuffer(sBuf!, offset: 0, index: 0)
        enc.setBuffer(w ?? sBuf!, offset: 0, index: 1)
        setBytesParams(enc, &p, index: 2)
        dispatchRows(enc, rows: rows)
    }

    private func barrier(_ enc: MTLComputeCommandEncoder) {
        enc.memoryBarrier(scope: .buffers)
    }

    private func readS(_ off: Int, _ n: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: sBuf!.contents().advanced(by: off * 4).bindMemory(to: Float.self, capacity: n),
            count: n))
    }

    private func writeS(_ off: Int, _ v: [Float]) {
        v.withUnsafeBufferPointer {
            sBuf!.contents().advanced(by: off * 4)
                .copyMemory(from: $0.baseAddress!, byteCount: v.count * 4)
        }
    }

    private func resetGPUState() {
        for fl in fastLayers {
            if let h = fl.hist { memset(h.contents(), 0, h.length) }
            if let st = fl.state { memset(st.contents(), 0, st.length) }
        }
    }

    private func encodePendingMoE(_ enc: MTLComputeCommandEncoder, _ pend: PendingMoE) throws {
        let cfg = config
        let D = cfg.hiddenSize, inter = cfg.moeIntermediateSize
        let shInter = cfg.sharedExpertIntermediateSize
        let moe = layers[pend.stacksLayer].moe
        let fl = fastLayers[pend.stacksLayer]
        let projs = expertProjs

        for (ki, pick) in pend.picks.enumerated() {
            let gLin: GPULinear
            let uLin: GPULinear
            let dLin: GPULinear
            var wE = 0, sE = 0
            if let projs, ki < pend.bufs.count {
                let buf = pend.bufs[ki]
                gLin = projs.gate.linear(on: buf)
                uLin = projs.up.linear(on: buf)
                dLin = projs.down.linear(on: buf)
            } else {
                let st = moe.stacks!
                gLin = st.gate.lin; uLin = st.up.lin; dLin = st.down.lin
                wE = pick.0; sE = pick.0
            }
            try engine.encodeGemv(enc, gLin, rows: inter,
                wExtra: wE * (moe.stacks?.gate.wStride ?? 0), sExtra: sE * (moe.stacks?.gate.sStride ?? 0),
                bExtra: sE * (moe.stacks?.gate.sStride ?? 0),
                x: sBuf!, xOff: reg.xmoe, y: sBuf!, yOff: reg.exp + ki * inter)
            try engine.encodeGemv(enc, uLin, rows: inter,
                wExtra: wE * (moe.stacks?.up.wStride ?? 0), sExtra: sE * (moe.stacks?.up.sStride ?? 0),
                bExtra: sE * (moe.stacks?.up.sStride ?? 0),
                x: sBuf!, xOff: reg.xmoe, y: sBuf!, yOff: reg.exp + pend.picks.count * inter + ki * inter)
        }
        try engine.encodeGemv(enc, moe.sharedGate, x: sBuf!, xOff: reg.xmoe, y: sBuf!, yOff: reg.sh)
        try engine.encodeGemv(enc, moe.sharedUp, x: sBuf!, xOff: reg.xmoe, y: sBuf!, yOff: reg.sh + shInter)
        try engine.encodeGemv(enc, fl.sharedGateLin, x: sBuf!, xOff: reg.xmoe, y: sBuf!, yOff: reg.shg)
        barrier(enc)
        let K = pend.picks.count
        for ki in 0..<K {
            try engine.encodeSiluMul(enc, buf: sBuf!, count: inter,
                gOff: reg.exp + ki * inter, uOff: reg.exp + K * inter + ki * inter,
                dstOff: reg.exp + 2 * K * inter + ki * inter)
        }
        try engine.encodeSiluMul(enc, buf: sBuf!, count: shInter,
            gOff: reg.sh, uOff: reg.sh + shInter, dstOff: reg.sh + 2 * shInter)
        barrier(enc)
        for (ki, pick) in pend.picks.enumerated() {
            let dLin: GPULinear
            var wE = 0, sE = 0
            if let projs = expertProjs, ki < pend.bufs.count {
                dLin = projs.down.linear(on: pend.bufs[ki])
            } else {
                dLin = moe.stacks!.down.lin
                wE = pick.0; sE = pick.0
            }
            try engine.encodeGemv(enc, dLin, rows: D,
                wExtra: wE * (moe.stacks?.down.wStride ?? 0), sExtra: sE * (moe.stacks?.down.sStride ?? 0),
                bExtra: sE * (moe.stacks?.down.sStride ?? 0),
                x: sBuf!, xOff: reg.exp + 2 * K * inter + ki * inter,
                y: sBuf!, yOff: reg.dexp + ki * D)
        }
        try engine.encodeGemv(enc, moe.sharedDown, x: sBuf!, xOff: reg.sh + 2 * shInter,
            y: sBuf!, yOff: reg.sh + 3 * shInter)
        barrier(enc)

        enc.setComputePipelineState(try engine.pipeline("weighted_accum"))
        var ap = SIMD8<UInt32>(UInt32(D), UInt32(K), UInt32(reg.dexp),
                               UInt32(reg.sh + 3 * shInter), UInt32(reg.shg), 0, 0, 0)
        enc.setBuffer(hBuf!, offset: 0, index: 0)
        enc.setBuffer(sBuf!, offset: 0, index: 1)
        setBytesParams(enc, &ap, index: 2)
        pend.weights.withUnsafeBufferPointer {
            enc.setBytes($0.baseAddress!, length: max(1, $0.count) * 4, index: 3)
        }
        dispatchN(enc, D)
        barrier(enc)
    }

    private func routerPicks() -> ([(Int, Float)], [Float]) {
        let cfg = config
        var router = readS(reg.rout, cfg.numExperts)
        QwenCPUModel.softmaxRow(&router, base: 0, count: cfg.numExperts)
        var picks: [(Int, Float)] = []
        for e in 0..<cfg.numExperts {
            let p = router[e]
            if picks.count < cfg.numExpertsPerTok {
                picks.append((e, p)); picks.sort { $0.1 > $1.1 }
            } else if p > picks[cfg.numExpertsPerTok - 1].1 {
                picks[cfg.numExpertsPerTok - 1] = (e, p); picks.sort { $0.1 > $1.1 }
            }
        }
        var sum: Float = 0
        for (_, p) in picks { sum += p }
        let weights = picks.map { cfg.normTopkProb ? $0.1 / sum : $0.1 }
        return (picks, weights)
    }

    func stepOneFast(
        _ token: Int, state: QwenCPUModel.DecodeState, projectLogits: Bool
    ) throws -> [Float] {
        let cfg = config
        let D = cfg.hiddenSize
        if boundStateID != ObjectIdentifier(state) || state.position == 0 {
            boundStateID = ObjectIdentifier(state)
            if state.position == 0 { resetGPUState() }
        }

        let h0 = try ckpt.moduleWeightSlice("model.embed_tokens", rowRange: token..<(token + 1))
        h0.withUnsafeBufferPointer {
            hBuf!.contents().copyMemory(from: $0.baseAddress!, byteCount: D * 4)
        }

        var pending: PendingMoE? = nil

        for li in 0..<cfg.numHiddenLayers {
            let L = layers[li]
            let fl = fastLayers[li]

            if let delta = L.delta {
                let cb = engine.queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                if let p = pending { try encodePendingMoE(enc, p); pending = nil }
                try encDeltaLayer(enc, delta: delta, fl: fl, moeGate: L.moe.gate)
                enc.endEncoding()
                commitAndWait(cb)
            } else {
                // Attention: projections, CPU core, then out-proj + router.
                let cb1 = engine.queue.makeCommandBuffer()!
                let e1 = cb1.makeComputeCommandEncoder()!
                if let p = pending { try encodePendingMoE(e1, p); pending = nil }
                try encNormCopy(e1, w: fl.inputNorm, dstOff: reg.x0)
                barrier(e1)
                let attn = L.attn!
                try engine.encodeGemv(e1, attn.qProj, x: sBuf!, xOff: reg.x0, y: sBuf!, yOff: reg.qout)
                try engine.encodeGemv(e1, attn.kProj, x: sBuf!, xOff: reg.x0, y: sBuf!, yOff: reg.knew)
                try engine.encodeGemv(e1, attn.vProj, x: sBuf!, xOff: reg.x0, y: sBuf!, yOff: reg.vnew)
                e1.endEncoding()
                commitAndWait(cb1)

                let attnOut = attnCoreCPU(layerIndex: li, state: state, attn: attn)
                writeS(reg.att, attnOut)

                let cb2 = engine.queue.makeCommandBuffer()!
                let e2 = cb2.makeComputeCommandEncoder()!
                try engine.encodeGemv(e2, attn.oProj, x: sBuf!, xOff: reg.att, y: sBuf!, yOff: reg.r)
                barrier(e2)
                var np = SIMD2<UInt32>(UInt32(D), UInt32(reg.r))
                e2.setComputePipelineState(try engine.pipeline("add_inplace"))
                e2.setBuffer(hBuf!, offset: 0, index: 0)
                e2.setBuffer(sBuf!, offset: 0, index: 1)
                setBytesParams(e2, &np, index: 2)
                dispatchN(e2, D)
                barrier(e2)
                try encNormCopy(e2, w: fl.postNorm, dstOff: reg.xmoe)
                barrier(e2)
                try engine.encodeGemv(e2, L.moe.gate, x: sBuf!, xOff: reg.xmoe, y: sBuf!, yOff: reg.rout)
                e2.endEncoding()
                commitAndWait(cb2)
            }

            let (picks, weights) = routerPicks()
            var bufs: [MTLBuffer] = []
            if let cache = expertCache {
                bufs = try cache.buffers(layer: li, experts: picks.map { $0.0 })
            }
            pending = PendingMoE(bufs: bufs, weights: weights, stacksLayer: li, picks: picks)
        }

        // Always apply the last layer's experts. Only the final token in a
        // multi-token call also needs final norm + vocabulary projection.
        let cb = engine.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        if let p = pending { try encodePendingMoE(enc, p) }
        if projectLogits {
            enc.setComputePipelineState(try engine.pipeline("norm_copy"))
            var np = NormParams(rows: 1, dim: UInt32(D), eps: Float(cfg.rmsNormEps), hasWeight: 1, scale: 1, off: UInt32(reg.x0))
            enc.setBuffer(hBuf!, offset: 0, index: 0)
            enc.setBuffer(sBuf!, offset: 0, index: 1)
            enc.setBuffer(finalNormBuf!, offset: 0, index: 2)
            setBytesParams(enc, &np, index: 3)
            dispatchRows(enc, rows: 1)
            barrier(enc)
            try engine.encodeGemv(enc, lmHead, x: sBuf!, xOff: reg.x0, y: logitsBuf, yOff: 0)
            activeStepCounters?.logitProjections += 1
        }
        enc.endEncoding()
        commitAndWait(cb)

        guard projectLogits else { return [] }
        return Array(UnsafeBufferPointer(
            start: logitsBuf.contents().bindMemory(to: Float.self, capacity: cfg.vocabSize),
            count: cfg.vocabSize))
    }

    private func encDeltaLayer(_ enc: MTLComputeCommandEncoder, delta: DeltaGPU, fl: FastLayer, moeGate: GPULinear) throws {
        let cfg = config
        let D = cfg.hiddenSize
        let nk = cfg.linearNumKeyHeads, nv = cfg.linearNumValueHeads
        let dk = cfg.linearKeyHeadDim, dv = cfg.linearValueHeadDim
        let keyDim = cfg.keyDim, convDim = cfg.convDim

        try encNormCopy(enc, w: fl.inputNorm, dstOff: reg.x0)
        barrier(enc)
        switch config.deltaLayout {
        case .split:
            try engine.encodeGemv(enc, delta.qkv!, x: sBuf!, xOff: reg.x0, y: sBuf!, yOff: reg.qkv)
            try engine.encodeGemv(enc, delta.zProj!, x: sBuf!, xOff: reg.x0, y: sBuf!, yOff: reg.z)
            try engine.encodeGemv(enc, delta.bProj!, x: sBuf!, xOff: reg.x0, y: sBuf!, yOff: reg.b)
            try engine.encodeGemv(enc, delta.aProj!, x: sBuf!, xOff: reg.x0, y: sBuf!, yOff: reg.a)
            barrier(enc)
        case .fusedInterleaved:
            try engine.encodeGemv(enc, delta.qkvz!, x: sBuf!, xOff: reg.x0, y: sBuf!, yOff: reg.qkvzStage)
            try engine.encodeGemv(enc, delta.ba!, x: sBuf!, xOff: reg.x0, y: sBuf!, yOff: reg.baStage)
            barrier(enc)
            enc.setComputePipelineState(try engine.pipeline("deinterleave_qkvz"))
            struct DeintParams {
                var nk: UInt32; var dk: UInt32; var rep: UInt32; var dv: UInt32
                var keyDim: UInt32; var srcOff: UInt32; var baOff: UInt32
                var qkvOff: UInt32; var zOff: UInt32; var bOff: UInt32; var aOff: UInt32
            }
            var dip = DeintParams(
                nk: UInt32(nk), dk: UInt32(dk), rep: UInt32(nv / nk), dv: UInt32(dv),
                keyDim: UInt32(keyDim), srcOff: UInt32(reg.qkvzStage), baOff: UInt32(reg.baStage),
                qkvOff: UInt32(reg.qkv), zOff: UInt32(reg.z), bOff: UInt32(reg.b), aOff: UInt32(reg.a)
            )
            enc.setBuffer(sBuf!, offset: 0, index: 0)
            setBytesParams(enc, &dip, index: 1)
            dispatchN(enc, nk)
            barrier(enc)
        }

        enc.setComputePipelineState(try engine.pipeline("conv_step"))
        var cp = SIMD4<UInt32>(UInt32(convDim), UInt32(cfg.linearConvKernelDim), UInt32(reg.qkv), UInt32(reg.conv))
        enc.setBuffer(sBuf!, offset: 0, index: 0)
        enc.setBuffer(fl.hist!, offset: 0, index: 1)
        enc.setBuffer(fl.convW!, offset: 0, index: 2)
        setBytesParams(enc, &cp, index: 3)
        dispatchN(enc, convDim)

        enc.setComputePipelineState(try engine.pipeline("delta_pre"))
        var dp = SIMD4<UInt32>(UInt32(nv), UInt32(reg.a), UInt32(reg.b), UInt32(reg.gb))
        enc.setBuffer(sBuf!, offset: 0, index: 0)
        enc.setBuffer(fl.aLog!, offset: 0, index: 1)
        enc.setBuffer(fl.dtBias!, offset: 0, index: 2)
        setBytesParams(enc, &dp, index: 3)
        dispatchN(enc, nv)
        barrier(enc)

        let invScale = 1 / Float(dk).squareRoot()
        try encRMSNorm(enc, off: reg.conv, rows: nk, dim: dk, w: nil, scale: invScale * invScale, eps: 1e-6)
        try encRMSNorm(enc, off: reg.conv + keyDim, rows: nk, dim: dk, w: nil, scale: invScale, eps: 1e-6)
        barrier(enc)

        enc.setComputePipelineState(try engine.pipeline("gated_delta_step"))
        var delp = MetalEngine.DeltaParams(T: 1, Hk: UInt32(nk), Hv: UInt32(nv), Dk: UInt32(dk), Dv: UInt32(dv))
        enc.setBuffer(sBuf!, offset: reg.conv * 4, index: 0)                    // q
        enc.setBuffer(sBuf!, offset: (reg.conv + keyDim) * 4, index: 1)         // k
        enc.setBuffer(sBuf!, offset: (reg.conv + 2 * keyDim) * 4, index: 2)     // v
        enc.setBuffer(sBuf!, offset: reg.gb * 4, index: 3)                      // g
        enc.setBuffer(sBuf!, offset: (reg.gb + nv) * 4, index: 4)               // beta
        enc.setBuffer(fl.state!, offset: 0, index: 5)
        enc.setBuffer(sBuf!, offset: reg.dy * 4, index: 6)
        enc.setBuffer(fl.state!, offset: 0, index: 7)
        setBytesParams(enc, &delp, index: 8)
        engine.dispatchThreads(
            enc, threads: MTLSize(width: 32, height: dv, depth: nv),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
        )
        barrier(enc)

        enc.setComputePipelineState(try engine.pipeline("gated_norm_mul"))
        struct GNP { var nv: UInt32; var dv: UInt32; var eps: Float; var yOff: UInt32; var zOff: UInt32; var outOff: UInt32 }
        var gp = GNP(nv: UInt32(nv), dv: UInt32(dv), eps: Float(cfg.rmsNormEps),
                     yOff: UInt32(reg.dy), zOff: UInt32(reg.z), outOff: UInt32(reg.dn))
        enc.setBuffer(sBuf!, offset: 0, index: 0)
        enc.setBuffer(fl.deltaNormW!, offset: 0, index: 1)
        setBytesParams(enc, &gp, index: 2)
        dispatchRows(enc, rows: nv)
        barrier(enc)

        try engine.encodeGemv(enc, delta.outProj, x: sBuf!, xOff: reg.dn, y: sBuf!, yOff: reg.r)
        barrier(enc)
        var ap = SIMD2<UInt32>(UInt32(D), UInt32(reg.r))
        enc.setComputePipelineState(try engine.pipeline("add_inplace"))
        enc.setBuffer(hBuf!, offset: 0, index: 0)
        enc.setBuffer(sBuf!, offset: 0, index: 1)
        setBytesParams(enc, &ap, index: 2)
        dispatchN(enc, D)
        barrier(enc)
        try encNormCopy(enc, w: fl.postNorm, dstOff: reg.xmoe)
        barrier(enc)
        try engine.encodeGemv(enc, moeGate, x: sBuf!, xOff: reg.xmoe, y: sBuf!, yOff: reg.rout)
    }

    private func attnCoreCPU(layerIndex: Int, state: QwenCPUModel.DecodeState, attn: AttnGPU) -> [Float] {
        let cfg = config
        let H = cfg.numAttentionHeads, KVH = cfg.numKeyValueHeads, hd = cfg.headDim
        let eps = Float(cfg.rmsNormEps)
        let past = state.position
        let qOut = readS(reg.qout, H * hd * 2)
        var k = readS(reg.knew, KVH * hd)
        let v = readS(reg.vnew, KVH * hd)

        var q = [Float](repeating: 0, count: H * hd)
        var gate = [Float](repeating: 0, count: H * hd)
        for head in 0..<H {
            for i in 0..<hd {
                q[head * hd + i] = qOut[head * 2 * hd + i]
                gate[head * hd + i] = qOut[head * 2 * hd + hd + i]
            }
        }
        QwenCPUModel.rmsNorm(&q, rows: H, dim: hd, weight: attn.qNorm, eps: eps)
        QwenCPUModel.rmsNorm(&k, rows: KVH, dim: hd, weight: attn.kNorm, eps: eps)
        applyRopeFast(&q, heads: H, position: past)
        applyRopeFast(&k, heads: KVH, position: past)

        var cache = state.kv[layerIndex] ?? (k: [], v: [])
        cache.k.append(contentsOf: k)
        cache.v.append(contentsOf: v)
        state.kv[layerIndex] = cache
        let kAll = cache.k, vAll = cache.v
        let kvLen = past + 1
        let scale = 1 / Float(hd).squareRoot()
        var attnOut = [Float](repeating: 0, count: H * hd)
        let group = H / KVH
        var scores = [Float](repeating: 0, count: kvLen)
        for head in 0..<H {
            let kvHead = head / group
            for sj in 0..<kvLen {
                var dot: Float = 0
                for i in 0..<hd { dot += q[head * hd + i] * kAll[(sj * KVH + kvHead) * hd + i] }
                scores[sj] = dot * scale
            }
            QwenCPUModel.softmaxRow(&scores, base: 0, count: kvLen)
            for sj in 0..<kvLen {
                let p = scores[sj]
                let vBase = (sj * KVH + kvHead) * hd
                for i in 0..<hd { attnOut[head * hd + i] += p * vAll[vBase + i] }
            }
        }
        for i in 0..<attnOut.count { attnOut[i] *= QwenCPUModel.sigmoid(gate[i]) }
        return attnOut
    }

    private func applyRopeFast(_ x: inout [Float], heads: Int, position: Int) {
        let hd = config.headDim
        let rot = config.rotaryDims
        let half = rot / 2
        for head in 0..<heads {
            let base = head * hd
            for j in 0..<half {
                let invFreq = powf(Float(config.ropeTheta), -Float(2 * j) / Float(rot))
                let angle = Float(position) * invFreq
                let c = cosf(angle), sn = sinf(angle)
                let a = x[base + j]
                let b = x[base + half + j]
                x[base + j] = a * c - b * sn
                x[base + half + j] = b * c + a * sn
            }
        }
    }
}
