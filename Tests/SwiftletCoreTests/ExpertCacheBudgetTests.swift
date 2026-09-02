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

        #expect(budget.reserve(64))
        #expect(budget.allocatedBytes == 64)
        #expect(!budget.reserve(37))
        #expect(budget.allocatedBytes == 64)
        #expect(budget.reserve(36))
        #expect(budget.allocatedBytes == 100)
        #expect(!budget.reserve(1))
    }

    @Test func reservationIsOverflowSafe() {
        var budget = ExpertCacheBudget(limitBytes: .max)

        #expect(!budget.reserve(0))
        #expect(!budget.reserve(-1))
        #expect(budget.reserve(Int.max - 1))
        #expect(!budget.reserve(2))
        #expect(budget.allocatedBytes == Int.max - 1)
    }
}
