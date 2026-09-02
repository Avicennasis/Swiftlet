import Testing
@testable import SwiftletCore

@Suite struct ExpertCacheBudgetTests {
    @Test func requiresMinimumLogicalWorkingSet() {
        #expect(ExpertCacheBudget.slotCapacity(
            limitBytes: 15_999, stride: 1_000, totalSlots: 100
        ) == nil)
        #expect(ExpertCacheBudget.slotCapacity(
            limitBytes: 16_000, stride: 1_000, totalSlots: 100
        ) == 16)

        // Small models need all of their slots, not an arbitrary minimum of 16.
        #expect(ExpertCacheBudget.slotCapacity(
            limitBytes: 8_000, stride: 1_000, totalSlots: 8
        ) == 8)
    }

    @Test func chargesPhysicalAllocationWithoutCrossingLimit() {
        var budget = ExpertCacheBudget(limitBytes: 100)

        let first = budget.reserve(64)
        #expect(first)
        #expect(budget.allocatedBytes == 64)
        let over = budget.reserve(37)
        #expect(!over)
        #expect(budget.allocatedBytes == 64)
        let exact = budget.reserve(36)
        #expect(exact)
        #expect(budget.allocatedBytes == 100)
        let full = budget.reserve(1)
        #expect(!full)
    }

    @Test func reservationIsOverflowSafe() {
        var budget = ExpertCacheBudget(limitBytes: .max)

        let zero = budget.reserve(0)
        let negative = budget.reserve(-1)
        let nearlyFull = budget.reserve(Int.max - 1)
        let overflow = budget.reserve(2)
        #expect(!zero)
        #expect(!negative)
        #expect(nearlyFull)
        #expect(!overflow)
        #expect(budget.allocatedBytes == Int.max - 1)
    }
}
