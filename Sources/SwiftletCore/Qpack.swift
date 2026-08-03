import Foundation

/// The `.qpack` container (v1), modeled on TurboFieldfare's `.gturbo`:
///
///   <name>.qpack/
///     manifest.json          arch + quant snapshot, file sizes
///     config.json            copied from the source checkpoint
///     model.safetensors      all dense/resident tensors, original quantized
///                            bytes (readable by `Checkpoint` as-is)
///     tokenizer files        copied from the source checkpoint
///     packed_experts/
///       layout.json          expert stride + intra-blob section offsets
///       layer_00.bin ...     one fixed-stride blob per expert, page aligned
///
/// One expert blob packs the six quantized sub-tensors of its three matrices
/// contiguously (gate/up/down x weight/scales/biases), so fetching an expert
/// is exactly one pread of `expertStride` bytes.
public enum Qpack {
    public static let pageAlignment = 16_384
    public static let manifestVersion = 1

    public struct Section: Codable, Sendable {
        public let name: String     // e.g. "gate_proj.weight"
        public let dtype: String
        public let shape: [Int]     // per-expert logical shape
        public let offset: Int      // byte offset inside the blob
        public let size: Int
    }

    public struct Layout: Codable, Sendable {
        public let expertCount: Int
        public let layerCount: Int
        public let expertStride: Int
        public let sections: [Section]
        public let linearLayers: [Bool]
    }

    public struct Manifest: Codable, Sendable {
        public let magic: String
        public let version: Int
        public let modelName: String
        public let sourceCheckpoint: String
        public let quantBits: Int?
        public let quantGroupSize: Int?
        public let files: [String: Int]   // relative path -> byte size
    }

    static func align(_ n: Int, to a: Int) -> Int { (n + a - 1) / a * a }
}

/// Repacks a local mlx-lm checkpoint directory into a `.qpack` container.
public struct QpackRepacker {
    public let checkpointDir: URL
    public let outputDir: URL
    public var log: (String) -> Void = { print($0) }

    public init(checkpointDir: URL, outputDir: URL) {
        self.checkpointDir = checkpointDir
        self.outputDir = outputDir
    }

    static let expertTensorSuffixes = ["gate_proj", "up_proj", "down_proj"]

    public func repack() throws {
        let fm = FileManager.default
        let config = try QwenConfig(url: checkpointDir.appendingPathComponent("config.json"))
        let ckpt = try Checkpoint(dir: checkpointDir)

        let expertsDir = outputDir.appendingPathComponent("packed_experts")
        try fm.createDirectory(at: expertsDir, withIntermediateDirectories: true)

        // --- Expert layout: derive per-expert section table from layer 0. ---
        let l0 = "model.layers.0.mlp.switch_mlp."
        var sections: [Qpack.Section] = []
        var offset = 0
        for proj in Self.expertTensorSuffixes {
            for part in ["weight", "scales", "biases"] {
                let name = l0 + proj + "." + part
                guard ckpt.contains(name) else { continue }
                let (info, _) = try ckpt.rawTensor(name)
                guard info.shape.first == config.numExperts,
                      let perElem = SafetensorsFile.bytesPerElement(info.dtype)
                else { throw Checkpoint.Error.badShape(name) }
                let perExpertShape = Array(info.shape.dropFirst())
                let perExpertBytes = perExpertShape.reduce(1, *) * perElem
                sections.append(Qpack.Section(
                    name: proj + "." + part, dtype: info.dtype, shape: perExpertShape,
                    offset: offset, size: perExpertBytes
                ))
                offset += perExpertBytes
            }
        }
        let stride = Qpack.align(offset, to: Qpack.pageAlignment)
        let layout = Qpack.Layout(
            expertCount: config.numExperts,
            layerCount: config.numHiddenLayers,
            expertStride: stride,
            sections: sections,
            linearLayers: (0..<config.numHiddenLayers).map(config.isLinearLayer)
        )
        log("expert blob: \(offset) B payload, stride \(stride) B, \(sections.count) sections")

        // --- Pack layer files. ---
        var files: [String: Int] = [:]
        for layer in 0..<config.numHiddenLayers {
            let prefix = "model.layers.\(layer).mlp.switch_mlp."
            var blob = Data(count: stride * config.numExperts)
            for section in sections {
                let (info, bytes) = try ckpt.rawTensor(prefix + section.name)
                guard info.shape.first == config.numExperts else {
                    throw Checkpoint.Error.badShape(prefix + section.name)
                }
                let perExpert = section.size
                for e in 0..<config.numExperts {
                    let src = e * perExpert..<(e + 1) * perExpert
                    let dst = e * stride + section.offset
                    blob.replaceSubrange(dst..<dst + perExpert, with: bytes[src])
                }
            }
            let fileName = String(format: "layer_%02d.bin", layer)
            try blob.write(to: expertsDir.appendingPathComponent(fileName))
            files["packed_experts/" + fileName] = blob.count
            log("packed \(fileName) (\(blob.count / 1_048_576) MiB)")
        }

        let layoutData = try JSONEncoder.sorted.encode(layout)
        try layoutData.write(to: expertsDir.appendingPathComponent("layout.json"))
        files["packed_experts/layout.json"] = layoutData.count

        // --- Dense/resident file: everything except experts and MTP. ---
        var dense: [(name: String, dtype: String, shape: [Int], bytes: Data)] = []
        for name in ckpt.tensorNames.sorted() {
            if name.contains(".mlp.switch_mlp.") || name.hasPrefix("mtp.") || name.contains(".mtp.") { continue }
            if name.hasPrefix("vision_tower.") || name.contains("model.visual") { continue }
            let (info, bytes) = try ckpt.rawTensor(name)
            dense.append((name, info.dtype, info.shape, bytes))
        }
        let denseURL = outputDir.appendingPathComponent("model.safetensors")
        try SafetensorsFile.write(to: denseURL, tensors: dense)
        files["model.safetensors"] = try fm.attributesOfItem(atPath: denseURL.path)[.size] as? Int ?? 0
        log("dense file: \(dense.count) tensors, \(files["model.safetensors"]! / 1_048_576) MiB")

        // --- Copy config + tokenizer artifacts. ---
        for aux in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                    "merges.txt", "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"] {
            let src = checkpointDir.appendingPathComponent(aux)
            guard fm.fileExists(atPath: src.path) else { continue }
            let dst = outputDir.appendingPathComponent(aux)
            try? fm.removeItem(at: dst)
            try fm.copyItem(at: src, to: dst)
            files[aux] = try fm.attributesOfItem(atPath: dst.path)[.size] as? Int ?? 0
        }

        let manifest = Qpack.Manifest(
            magic: "QPACK",
            version: Qpack.manifestVersion,
            modelName: config.modelType,
            sourceCheckpoint: checkpointDir.lastPathComponent,
            quantBits: ckpt.defaultQuant?.bits,
            quantGroupSize: ckpt.defaultQuant?.groupSize,
            files: files
        )
        try JSONEncoder.sorted.encode(manifest).write(to: outputDir.appendingPathComponent("manifest.json"))
        log("wrote manifest; container complete at \(outputDir.path)")
    }
}

/// Reads expert blobs back out of a `.qpack` container, one pread per expert.
public final class QpackExpertReader {
    public let layout: Qpack.Layout
    private let dir: URL
    private var fds: [Int32]

    public init(containerDir: URL) throws {
        dir = containerDir.appendingPathComponent("packed_experts")
        let layoutData = try Data(contentsOf: dir.appendingPathComponent("layout.json"))
        layout = try JSONDecoder().decode(Qpack.Layout.self, from: layoutData)
        fds = Array(repeating: -1, count: layout.layerCount)
    }

    deinit { for fd in fds where fd >= 0 { close(fd) } }

    /// Single-pread fetch of one expert blob into a caller-provided buffer of
    /// at least `layout.expertStride` bytes.
    public func readExpert(layer: Int, expert: Int, into buffer: UnsafeMutableRawPointer) throws {
        if fds[layer] < 0 {
            let path = dir.appendingPathComponent(String(format: "layer_%02d.bin", layer)).path
            let fd = open(path, O_RDONLY)
            guard fd >= 0 else { throw Checkpoint.Error.missingTensor(path) }
            fds[layer] = fd
        }
        let offset = off_t(expert * layout.expertStride)
        let n = pread(fds[layer], buffer, layout.expertStride, offset)
        guard n == layout.expertStride else {
            throw Checkpoint.Error.badShape("short read: layer \(layer) expert \(expert)")
        }
    }

    public func section(_ name: String) -> Qpack.Section? {
        layout.sections.first { $0.name == name }
    }
}

extension JSONEncoder {
    static var sorted: JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys, .prettyPrinted]
        return e
    }
}
