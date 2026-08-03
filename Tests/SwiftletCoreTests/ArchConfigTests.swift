import Testing
@testable import SwiftletCore

@Suite struct ArchConfigTests {
    @Test func qwen3Next80BLayout() {
        let c = ArchConfig.qwen3Next80B
        #expect(c.linearLayerCount == 36)
        #expect(c.fullAttentionLayerCount == 12)
        // mlx-lm pattern: layers 3, 7, 11, ... are full attention (0-based).
        #expect(!c.isLinearLayer(3) && !c.isLinearLayer(47))
        #expect(c.isLinearLayer(0) && c.isLinearLayer(1) && c.isLinearLayer(2) && c.isLinearLayer(4))
        #expect(c.expertParamCount == 3 * 2048 * 512)
        // ~1.77 MB per expert blob at int4 g64, ~44 GB pool.
        #expect(c.expertBlobBytesInt4G64 == 1_769_472)
        let poolGiB = Double(c.expertBlobBytesInt4G64 * c.routedExpertTotal) / Double(1 << 30)
        #expect(poolGiB > 40 && poolGiB < 45)
        // Per-token cold IO ~810 MiB.
        let coldMiB = Double(c.expertBlobBytesInt4G64 * c.routedFetchesPerToken) / Double(1 << 20)
        #expect(coldMiB > 700 && coldMiB < 900)
    }

    @Test func deltaNetStateIsContextIndependent() {
        let c = ArchConfig.qwen3Next80B
        // 36 layers x 32 heads x 128x128 f32 = 72 MiB exactly.
        #expect(c.deltaNetStateBytes == 36 * 32 * 128 * 128 * 4)
        // KV only exists on the 12 GQA layers: 12 x 2 x 256 x 2 x 2 = 24 KiB/token.
        #expect(c.kvBytesPerToken == 24_576)
    }

    @Test func familyVariants() {
        #expect(ArchConfig.qwen3_5_397B.linearLayerCount == 45)
        #expect(ArchConfig.qwen3_6_35B.routedExpertTotal == 40 * 256)
    }
}
