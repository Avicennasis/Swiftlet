import Foundation

extension QwenMetalModel {
    /// S3c: where the CPU thread spends the time between command buffers.
    ///
    /// The S3a/S3b decomposition leaves "wall − blocking wait − encode" as one
    /// undifferentiated CPU gap; the M1 bench (`docs/BENCH_QWEN36_35B_M1_MINI.md`)
    /// measured it at 38–56% of a decode step. This struct sub-attributes that
    /// gap to the CPU work the step performs by construction — every field
    /// is the wall time inside one kind of scope, timed with the same clock
    /// as the encode and wait figures. Fields never overlap (the scopes are
    /// flat and disjoint), so their sum is a lower bound on the gap; whatever
    /// remains is `StepMetrics.cpuGapOtherSeconds` (Swift loop overhead,
    /// buffer growth, Metal object teardown). Throw-safe like every other
    /// counter: a throwing step publishes the partial values it accumulated.
    public struct CPUGapBreakdown: Equatable, Sendable {
        /// Embedding row lookups: checkpoint read + dequantization + copy
        /// into the hidden buffer. One per token processed.
        public let embeddingSeconds: Double
        public let embeddingLookups: Int
        /// Router logits readback, softmax, top-k selection and weight
        /// normalisation, per (token, layer).
        public let routerSeconds: Double
        /// Expert-cache fetches (`ExpertCache.buffers`): a hit is a
        /// dictionary lookup, a miss is a victim scan plus one pread into
        /// the slot. Zero on raw checkpoint dirs (no cache).
        public let expertFetchSeconds: Double
        public let expertFetchHits: Int
        public let expertFetchMisses: Int
        /// Appending the step's GPU KV rows to the CPU-side `state.kv`
        /// mirror (attention layers only).
        public let kvMirrorSeconds: Double
        /// Command-buffer + encoder creation, before the buffer's encode
        /// clock starts.
        public let commandBufferSetupSeconds: Double
        /// `MTLCommandBuffer.commit()` itself, between encode end and the
        /// blocking wait.
        public let commitSeconds: Double
        /// Copying the vocabulary logits out of the shared buffer into the
        /// returned array.
        public let logitsReadbackSeconds: Double

        public static let zero = CPUGapBreakdown(
            embeddingSeconds: 0, embeddingLookups: 0, routerSeconds: 0,
            expertFetchSeconds: 0, expertFetchHits: 0, expertFetchMisses: 0,
            kvMirrorSeconds: 0, commandBufferSetupSeconds: 0, commitSeconds: 0,
            logitsReadbackSeconds: 0
        )

        /// Sum of every attributed scope.
        public var attributedSeconds: Double {
            embeddingSeconds + routerSeconds + expertFetchSeconds + kvMirrorSeconds
                + commandBufferSetupSeconds + commitSeconds + logitsReadbackSeconds
        }
    }
}

extension QwenMetalModel.StepMetrics {
    /// CPU encode wall time across the committed buffers (creation to
    /// commit), as the S3b timeline reports it.
    public var encodeSeconds: Double {
        commandBufferTimeline.reduce(0) { $0 + $1.encodeSeconds }
    }

    /// The S3a/S3b residual: wall − blocking wait − encode. Meaningful for a
    /// step that completed without throwing (a thrown step's uncommitted
    /// encode is not in the timeline).
    public var cpuGapSeconds: Double {
        max(0, stepWallSeconds - blockingWaitSeconds - encodeSeconds)
    }

    /// Gap time no S3c scope claims. Slightly negative values are clock
    /// jitter, never a real overlap (scopes are disjoint).
    public var cpuGapOtherSeconds: Double {
        cpuGapSeconds - cpuGap.attributedSeconds
    }
}
