import Foundation
import SwiftletCore

// Installer/repacker for .qpack containers.
//   Local repack:      swiftlet-repack --source <mlx-checkpoint-dir> --output <dir>.qpack
//   Streaming install: swiftlet-repack --from-hf <org/repo> --output <dir>.qpack
// The streaming path never stores the raw checkpoint: bytes route from the
// network straight into the container. Re-run to resume after interruption.

let args = CommandLine.arguments
func value(_ flag: String) -> String? {
    guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
    return args[i + 1]
}

guard let output = value("--output") else {
    print("usage:")
    print("  swiftlet-repack --source <mlx-checkpoint-dir> --output <dir>.qpack")
    print("  swiftlet-repack --from-hf <org/repo> --output <dir>.qpack   (streaming, resumable)")
    exit(2)
}

let start = Date()
do {
    if let repo = value("--from-hf") {
        let installer = StreamingInstaller(
            source: .huggingFace(repo: repo),
            outputDir: URL(fileURLWithPath: output)
        )
        try installer.install()
    } else if let source = value("--source") {
        let repacker = QpackRepacker(
            checkpointDir: URL(fileURLWithPath: source),
            outputDir: URL(fileURLWithPath: output)
        )
        try repacker.repack()
    } else {
        print("need --source or --from-hf")
        exit(2)
    }
    print(String(format: "done in %.1fs", -start.timeIntervalSinceNow))
} catch {
    print("install failed: \(error)")
    exit(1)
}
