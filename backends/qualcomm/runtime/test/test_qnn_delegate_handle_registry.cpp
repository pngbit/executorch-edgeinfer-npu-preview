/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnDelegateHandleRegistry.h>
#include <executorch/backends/qualcomm/runtime/QnnMemoryRange.h>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace qnn {
namespace {

TEST(QnnDelegateHandleRegistryTest, SharedHandleLivesUntilFinalRelease) {
  QnnDelegateHandleRegistry<int> registry;
  int handle = 1;
  const std::string identity = "backend=htp;soc=sm8650;context=7";

  EXPECT_EQ(registry.cache_or_acquire(identity, &handle), &handle);
  EXPECT_EQ(registry.acquire(identity), &handle);
  EXPECT_EQ(registry.acquire(identity), &handle);
  EXPECT_TRUE(registry.contains(&handle));
  EXPECT_FALSE(registry.release(&handle));
  EXPECT_TRUE(registry.contains(&handle));
  EXPECT_FALSE(registry.release(&handle));
  EXPECT_TRUE(registry.contains(&handle));
  EXPECT_TRUE(registry.release(&handle));
  EXPECT_FALSE(registry.contains(&handle));
  EXPECT_EQ(registry.acquire(identity), nullptr);
}

TEST(QnnDelegateHandleRegistryTest, UniqueHandlesDoNotAliasSignatures) {
  QnnDelegateHandleRegistry<int> registry;
  int shared = 1;
  int unique = 2;

  EXPECT_EQ(registry.cache_or_acquire("shared-identity", &shared), &shared);
  EXPECT_TRUE(registry.track_unique(&unique));
  EXPECT_FALSE(registry.track_unique(&unique));
  EXPECT_TRUE(registry.release(&unique));
  EXPECT_TRUE(registry.contains(&shared));
  EXPECT_TRUE(registry.release(&shared));
}

TEST(QnnDelegateHandleRegistryTest, ConcurrentCacheChoosesOneHandle) {
  constexpr size_t kThreads = 8;
  QnnDelegateHandleRegistry<int> registry;
  std::array<int, kThreads> candidates{};
  std::array<int*, kThreads> selected{};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (size_t index = 0; index < kThreads; ++index) {
    candidates[index] = static_cast<int>(index);
    workers.emplace_back([&, index]() {
      selected[index] =
          registry.cache_or_acquire("concurrent-identity", &candidates[index]);
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }

  for (int* handle : selected) {
    EXPECT_EQ(handle, selected.front());
  }
  for (size_t index = 0; index + 1 < kThreads; ++index) {
    EXPECT_FALSE(registry.release(selected.front()));
  }
  EXPECT_TRUE(registry.release(selected.front()));
  EXPECT_FALSE(registry.contains(selected.front()));
}

TEST(QnnDelegateHandleRegistryTest, DifferentFullIdentitiesDoNotShare) {
  QnnDelegateHandleRegistry<int> registry;
  int sm8650_vtcm8 = 1;
  int sm8650_vtcm4 = 2;
  const std::string first_identity =
      "backend=htp;library=libQnnHtp.so;soc=sm8650;vtcm=8;context=19";
  const std::string second_identity =
      "backend=htp;library=libQnnHtp.so;soc=sm8650;vtcm=4;context=19";

  EXPECT_EQ(
      registry.cache_or_acquire(first_identity, &sm8650_vtcm8), &sm8650_vtcm8);
  EXPECT_EQ(
      registry.cache_or_acquire(second_identity, &sm8650_vtcm4), &sm8650_vtcm4);
  EXPECT_EQ(registry.acquire(first_identity), &sm8650_vtcm8);
  EXPECT_EQ(registry.acquire(second_identity), &sm8650_vtcm4);

  EXPECT_FALSE(registry.release(&sm8650_vtcm8));
  EXPECT_TRUE(registry.release(&sm8650_vtcm8));
  EXPECT_FALSE(registry.release(&sm8650_vtcm4));
  EXPECT_TRUE(registry.release(&sm8650_vtcm4));
}

TEST(
    QnnDelegateHandleRegistryTest,
    DestroyWaitsForExecutionAndRejectsRetiringHandle) {
  using namespace std::chrono_literals;

  QnnDelegateHandleRegistry<int> registry;
  int handle = 1;
  const std::string identity = "backend=htp;soc=sm8650;context=23";
  ASSERT_EQ(registry.cache_or_acquire(identity, &handle), &handle);
  auto execution = registry.acquire_execution(&handle);
  ASSERT_TRUE(execution);

  std::atomic<bool> destroy_started{false};
  std::atomic<bool> destroy_finished{false};
  std::atomic<bool> final_release{false};
  std::thread destroyer([&]() {
    destroy_started.store(true, std::memory_order_release);
    final_release.store(
        registry.release_and_destroy(
            &handle,
            [&](int* destroyed) {
              EXPECT_EQ(destroyed, &handle);
              destroy_finished.store(true, std::memory_order_release);
            }),
        std::memory_order_release);
  });

  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while ((!destroy_started.load(std::memory_order_acquire) ||
          registry.contains(&handle)) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  const bool retired = !registry.contains(&handle);
  EXPECT_TRUE(retired);
  EXPECT_FALSE(destroy_finished.load(std::memory_order_acquire));
  if (retired) {
    EXPECT_FALSE(registry.acquire_execution(&handle));
    EXPECT_EQ(registry.acquire(identity), nullptr);
  }

  execution = {};
  destroyer.join();
  EXPECT_TRUE(final_release.load(std::memory_order_acquire));
  EXPECT_TRUE(destroy_finished.load(std::memory_order_acquire));
  EXPECT_FALSE(registry.contains(&handle));
}

TEST(QnnMemoryRangeTest, ResolvesAlignedPointerToRawRpcAllocationBase) {
  auto* raw_base = reinterpret_cast<void*>(uintptr_t{0x1000});
  auto* aligned = reinterpret_cast<void*>(uintptr_t{0x1040});
  std::unordered_map<void*, void*> restore_map{{aligned, raw_base}};

  EXPECT_EQ(ResolveRpcAllocationBase(restore_map, aligned), raw_base);
  EXPECT_EQ(ResolveRpcAllocationBase(restore_map, raw_base), nullptr);
}

TEST(QnnMemoryRangeTest, ValidatesRpcMemAllocationBoundaries) {
  auto minimum = GetRpcMemAllocationLayout(1, 1);
  ASSERT_TRUE(minimum.has_value());
  EXPECT_EQ(minimum->allocation_bytes, 1);
  EXPECT_EQ(minimum->padding_bytes, size_t{0});

  auto aligned = GetRpcMemAllocationLayout(1024, 64);
  ASSERT_TRUE(aligned.has_value());
  EXPECT_EQ(aligned->allocation_bytes, 1087);
  EXPECT_EQ(aligned->padding_bytes, size_t{63});

  const size_t max_rpcmem =
      static_cast<size_t>(std::numeric_limits<int32_t>::max());
  auto maximum = GetRpcMemAllocationLayout(max_rpcmem - 63, 64);
  ASSERT_TRUE(maximum.has_value());
  EXPECT_EQ(maximum->allocation_bytes, std::numeric_limits<int32_t>::max());

  EXPECT_FALSE(GetRpcMemAllocationLayout(0, 64).has_value());
  EXPECT_FALSE(GetRpcMemAllocationLayout(1, 0).has_value());
  EXPECT_FALSE(GetRpcMemAllocationLayout(1, 3).has_value());
  EXPECT_FALSE(GetRpcMemAllocationLayout(max_rpcmem - 62, 64).has_value());
  EXPECT_FALSE(GetRpcMemAllocationLayout(max_rpcmem + 1, 1).has_value());
  EXPECT_FALSE(
      GetRpcMemAllocationLayout(std::numeric_limits<size_t>::max() - 31, 64)
          .has_value());
}

TEST(QnnMemoryRangeTest, RejectsRpcMemAddressAlignmentOverflow) {
  auto aligned = GetAlignedRpcMemAddress(uintptr_t{0x1003}, 64);
  ASSERT_TRUE(aligned.has_value());
  EXPECT_EQ(*aligned, uintptr_t{0x1040});
  EXPECT_FALSE(GetAlignedRpcMemAddress(uintptr_t{0x1000}, 0).has_value());
  EXPECT_FALSE(GetAlignedRpcMemAddress(uintptr_t{0x1000}, 3).has_value());
  EXPECT_FALSE(
      GetAlignedRpcMemAddress(std::numeric_limits<uintptr_t>::max(), 64)
          .has_value());
}

TEST(QnnMemoryRangeTest, DescriptorOffsetIsRelativeToRawFdBase) {
  auto* raw_base = reinterpret_cast<void*>(uintptr_t{0x1003});
  auto* aligned_base = reinterpret_cast<void*>(uintptr_t{0x1040});

  auto range = GetRpcMemDescriptorRange(
      raw_base,
      aligned_base,
      /*allocation_bytes=*/1087,
      /*usable_bytes=*/1024,
      /*tensor_offset_from_aligned=*/128,
      /*tensor_bytes=*/256);
  ASSERT_TRUE(range.has_value());
  EXPECT_EQ(range->total_bytes, size_t{1087});
  EXPECT_EQ(range->tensor_offset, size_t{61 + 128});

  auto final_byte = GetRpcMemDescriptorRange(
      raw_base,
      aligned_base,
      /*allocation_bytes=*/1087,
      /*usable_bytes=*/1024,
      /*tensor_offset_from_aligned=*/1023,
      /*tensor_bytes=*/1);
  ASSERT_TRUE(final_byte.has_value());
  EXPECT_EQ(final_byte->tensor_offset, size_t{1084});

  EXPECT_FALSE(GetRpcMemDescriptorRange(
                   raw_base,
                   aligned_base,
                   /*allocation_bytes=*/1087,
                   /*usable_bytes=*/1024,
                   /*tensor_offset_from_aligned=*/1024,
                   /*tensor_bytes=*/1)
                   .has_value());
}

TEST(QnnMemoryRangeTest, FdBaseAndDescriptorOffsetShareRawOrigin) {
  auto* raw_base = reinterpret_cast<void*>(uintptr_t{0x2003});
  auto* aligned_base = reinterpret_cast<void*>(uintptr_t{0x2040});
  std::unordered_map<void*, void*> restore_map{{aligned_base, raw_base}};

  void* fd_base = ResolveRpcAllocationBase(restore_map, aligned_base);
  ASSERT_EQ(fd_base, raw_base);
  auto range = GetRpcMemDescriptorRange(
      fd_base,
      aligned_base,
      /*allocation_bytes=*/1087,
      /*usable_bytes=*/1024,
      /*tensor_offset_from_aligned=*/256,
      /*tensor_bytes=*/512);
  ASSERT_TRUE(range.has_value());
  EXPECT_EQ(range->total_bytes, size_t{1087});
  EXPECT_EQ(range->tensor_offset, size_t{61 + 256});
}

TEST(QnnMemoryRangeTest, AddressOffsetRejectsInvalidOrdering) {
  auto* base = reinterpret_cast<void*>(uintptr_t{0x1000});
  auto* address = reinterpret_cast<void*>(uintptr_t{0x1040});
  auto offset = GetAddressOffset(base, address);
  ASSERT_TRUE(offset.has_value());
  EXPECT_EQ(*offset, size_t{64});
  EXPECT_FALSE(GetAddressOffset(nullptr, address).has_value());
  EXPECT_FALSE(GetAddressOffset(base, nullptr).has_value());
  EXPECT_FALSE(GetAddressOffset(address, base).has_value());
}

TEST(QnnMemoryRangeTest, RejectsInvalidRpcMemDescriptorRanges) {
  auto* raw_base = reinterpret_cast<void*>(uintptr_t{0x1003});
  auto* aligned_base = reinterpret_cast<void*>(uintptr_t{0x1040});
  EXPECT_FALSE(GetRpcMemDescriptorRange(nullptr, aligned_base, 1087, 1024, 0, 1)
                   .has_value());
  EXPECT_FALSE(GetRpcMemDescriptorRange(raw_base, nullptr, 1087, 1024, 0, 1)
                   .has_value());
  EXPECT_FALSE(
      GetRpcMemDescriptorRange(aligned_base, raw_base, 1087, 1024, 0, 1)
          .has_value());
  EXPECT_FALSE(
      GetRpcMemDescriptorRange(raw_base, aligned_base, 0, 0, 0, 1).has_value());
  auto empty_tensor =
      GetRpcMemDescriptorRange(raw_base, aligned_base, 1087, 1024, 1024, 0);
  ASSERT_TRUE(empty_tensor.has_value());
  EXPECT_EQ(empty_tensor->tensor_offset, size_t{1085});
  EXPECT_FALSE(GetRpcMemDescriptorRange(raw_base, aligned_base, 60, 1, 0, 1)
                   .has_value());
  EXPECT_FALSE(
      GetRpcMemDescriptorRange(raw_base, aligned_base, 1087, 1027, 0, 1)
          .has_value());
  EXPECT_FALSE(
      GetRpcMemDescriptorRange(raw_base, aligned_base, 1087, 1024, 1024, 1)
          .has_value());
  EXPECT_FALSE(GetRpcMemDescriptorRange(
                   raw_base,
                   aligned_base,
                   1087,
                   1024,
                   std::numeric_limits<size_t>::max(),
                   1)
                   .has_value());
}

TEST(QnnMemoryRangeTest, RejectsInvalidAndOverflowingRanges) {
  const uintptr_t max = std::numeric_limits<uintptr_t>::max();
  EXPECT_FALSE(IsAddressInRange(nullptr, reinterpret_cast<void*>(1), 1));
  EXPECT_FALSE(IsAddressInRange(reinterpret_cast<void*>(1), nullptr, 1));
  EXPECT_FALSE(IsAddressInRange(
      reinterpret_cast<void*>(1), reinterpret_cast<void*>(1), 0));
  EXPECT_FALSE(IsAddressInRange(
      reinterpret_cast<void*>(max - 1), reinterpret_cast<void*>(max - 8), 16));
}

TEST(QnnMemoryRangeTest, DeregistersOnlyMatchingHandlesAndRetainsFailures) {
  auto* first = reinterpret_cast<void*>(uintptr_t{0x1020});
  auto* failed = reinterpret_cast<void*>(uintptr_t{0x1040});
  auto* outside = reinterpret_cast<void*>(uintptr_t{0x2020});
  std::unordered_map<int, void*> handles{{1, first}, {2, failed}, {3, outside}};
  std::vector<int> released;

  EXPECT_FALSE(DeregisterHandlesInRange(
      handles,
      reinterpret_cast<void*>(uintptr_t{0x1000}),
      0x100,
      [](int handle) { return handle != 2; },
      [&](int handle) { released.push_back(handle); }));
  ASSERT_EQ(released.size(), 1);
  EXPECT_EQ(released.front(), 1);
  EXPECT_EQ(handles.count(1), 0);
  EXPECT_EQ(handles.count(2), 1);
  EXPECT_EQ(handles.count(3), 1);
}

} // namespace
} // namespace qnn
} // namespace backends
} // namespace executorch
