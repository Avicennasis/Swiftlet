# Swiftlet on iPhone

Goal: run Qwen3.6-35B-A3B (or Qwen3-30B-A3B class) locally on an iPhone by
streaming experts from NAND. An iPhone can never hold an 18 GB model in RAM,
so expert streaming is not an optimization here, it is the only way. That
makes the phone the strongest demo of the whole thesis.

Status: SwiftletCore (checkpoint reader, .qpack, CPU reference, MetalEngine)
compiles for iOS unchanged (`xcodebuild -scheme SwiftletCore -destination
'generic/platform=iOS'`). The shader is runtime-compiled from a bundled
source file, so no Metal build toolchain is needed.

## Feasibility budget (Qwen3.6-35B-A3B, iPhone 17 Pro class)

| Item | Size |
|---|---|
| Model on NAND (.qpack, int4 g64) | ~18 GB |
| Resident dense weights (stay int4, dequant in-kernel) | ~1.3 GB |
| KV cache (10 GQA layers only), 8K ctx fp16 | ~160 MB |
| DeltaNet fixed state (30 linear layers) | ~60 MB |
| Expert cache budget | 2-3 GB (8 GB devices) / 4-6 GB (12 GB devices) |
| Cold IO per token (8 x 40 experts x ~1.8 MB) | ~570 MB |

iPhone NAND sequential reads: ~1.5-3 GB/s -> cold floor ~3-5 tok/s; a 2-3 GB
cache holds 15-30% of the expert pool, so warm workloads should do
meaningfully better. Active compute is only ~3B params/token, well within the
A-series GPU.

## iOS-specific constraints and answers

- **Jetsam (per-app memory limit)**: ~5-6 GB usable on 8 GB devices. Our
  resident set is ~1.6 GB; the expert cache must be elastic: register for
  memory-pressure notifications, shrink the cache instead of dying. The
  `com.apple.developer.kernel.increased-memory-limit` entitlement raises the
  ceiling on Pro devices.
- **Storage**: model lives in Application Support (excluded from iCloud
  backup via `isExcludedFromBackup`). Download needs a 256 GB+ device
  realistically; installer must be resumable (same HTTP-range design as Mac).
- **mmap**: works on iOS; dense file maps read-only exactly like macOS.
  `F_NOCACHE` also exists on iOS for the streaming path.
- **Background**: decode only runs foregrounded; that's fine for chat.
- **Thermals**: sustained decode will throttle; expect tok/s to sag after
  minutes. Acceptable for a demo; production wants a low-power mode (smaller
  cache, batched IO).
- **App Store**: local-LLM apps with post-install model downloads are
  established practice. Apache-2.0 weights are clean.

## Build plan

1. **Mac Metal runtime first** (M3): kernels are identical on iOS; every hour
   spent on the Mac runtime is iPhone progress.
2. **iOS smoke target**: a minimal SwiftUI app (new Xcode project embedding
   this package) that runs the TINY fixture model on-device to prove the
   Metal path on A-series silicon. No big download needed.
3. **35B on device**: installer UI (resumable download + repack on device or
   download pre-repacked .qpack), elastic expert cache, memory-pressure
   handling, thermal-aware decode.
4. **Ship**: TestFlight beta.

Device targets: iPhone 16/17 Pro (8-12 GB RAM, 256 GB+). The 12 GB Pro
models are the comfortable target; 8 GB is the stretch goal, mirroring
TurboFieldfare's 8 GB Mac story.
