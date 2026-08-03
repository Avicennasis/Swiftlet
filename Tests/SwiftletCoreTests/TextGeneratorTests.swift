import Foundation
import Testing
@testable import SwiftletCore

@Suite struct TextGeneratorTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    @Test func tokenStreamMatchesDirectGenerate() async throws {
        let model = try QwenCPUModel(modelDir: Self.fixturesDir.appendingPathComponent("tiny-model-q4"))
        model.retainAllLayers = true
        let generator = TextGenerator(model: model)

        var direct: [Int] = []
        try generator.generate(promptIds: [1, 5, 9], maxNew: 6) { direct.append($0); return true }

        var streamed: [Int] = []
        for try await token in generator.tokenStream(promptIds: [1, 5, 9], maxNew: 6) {
            streamed.append(token)
        }
        #expect(streamed == direct)
        #expect(streamed.count == 6)
    }
}
