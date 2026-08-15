/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace executorch {
namespace backends {
namespace qnn {

template <typename Handle>
class QnnDelegateHandleRegistry final {
 private:
  struct Entry final {
    explicit Entry(Handle* value) : handle(value) {}
    Entry(Handle* value, std::string shared_identity)
        : handle(value), identity(std::move(shared_identity)) {}

    Handle* handle;
    std::optional<std::string> identity;
    size_t owner_count{1};
    size_t active_executions{0};
    bool retiring{false};
    std::mutex execution_mutex;
    std::condition_variable idle;
  };

 public:
  class ExecutionLease final {
   public:
    ExecutionLease() = default;
    ExecutionLease(const ExecutionLease&) = delete;
    ExecutionLease& operator=(const ExecutionLease&) = delete;
    ExecutionLease(ExecutionLease&& other) noexcept
        : registry_(other.registry_),
          entry_(std::move(other.entry_)),
          execution_lock_(std::move(other.execution_lock_)) {
      other.registry_ = nullptr;
    }
    ExecutionLease& operator=(ExecutionLease&& other) noexcept {
      if (this != &other) {
        reset();
        registry_ = other.registry_;
        entry_ = std::move(other.entry_);
        execution_lock_ = std::move(other.execution_lock_);
        other.registry_ = nullptr;
      }
      return *this;
    }
    ~ExecutionLease() {
      reset();
    }

    explicit operator bool() const {
      return entry_ != nullptr;
    }

    Handle* get() const {
      return entry_ == nullptr ? nullptr : entry_->handle;
    }

   private:
    friend class QnnDelegateHandleRegistry;
    ExecutionLease(
        QnnDelegateHandleRegistry* registry,
        std::shared_ptr<Entry> entry)
        : registry_(registry),
          entry_(std::move(entry)),
          execution_lock_(entry_->execution_mutex) {}

    void reset() {
      if (registry_ != nullptr && entry_ != nullptr) {
        if (execution_lock_.owns_lock()) {
          execution_lock_.unlock();
        }
        registry_->finish_execution(entry_);
      }
      registry_ = nullptr;
      entry_.reset();
    }

    QnnDelegateHandleRegistry* registry_{nullptr};
    std::shared_ptr<Entry> entry_;
    std::unique_lock<std::mutex> execution_lock_;
  };

  Handle* acquire(const std::string& identity) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto entry = shared_handles_.find(identity);
    if (entry == shared_handles_.end()) {
      return nullptr;
    }
    ++entry->second->owner_count;
    return entry->second->handle;
  }

  Handle* cache_or_acquire(const std::string& identity, Handle* candidate) {
    if (candidate == nullptr) {
      return nullptr;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    // A candidate already owned by another entry cannot safely be reused for a
    // second identity. In particular, it may be retiring outside this lock.
    if (entries_.count(candidate) != 0) {
      return nullptr;
    }
    auto entry = shared_handles_.find(identity);
    if (entry != shared_handles_.end()) {
      ++entry->second->owner_count;
      return entry->second->handle;
    }
    auto candidate_entry = std::make_shared<Entry>(candidate, identity);
    shared_handles_[identity] = candidate_entry;
    entries_[candidate] = candidate_entry;
    return candidate;
  }

  bool track_unique(Handle* handle) {
    if (handle == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    if (entries_.count(handle) != 0) {
      return false;
    }
    entries_[handle] = std::make_shared<Entry>(handle);
    return true;
  }

  ExecutionLease acquire_execution(Handle* handle) {
    if (handle == nullptr) {
      return {};
    }
    std::shared_ptr<Entry> selected;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      auto entry = entries_.find(handle);
      if (entry == entries_.end() || entry->second->retiring) {
        return {};
      }
      selected = entry->second;
      ++selected->active_executions;
    }
    return ExecutionLease(this, std::move(selected));
  }

  bool release(Handle* handle) {
    return release_and_destroy(handle, [](Handle*) {});
  }

  template <typename DestroyFn>
  bool release_and_destroy(Handle* handle, DestroyFn&& destroy) {
    if (handle == nullptr) {
      return false;
    }
    std::shared_ptr<Entry> entry;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      auto found = entries_.find(handle);
      if (found == entries_.end() || found->second->retiring) {
        return false;
      }
      entry = found->second;
      if (entry->owner_count > 1) {
        --entry->owner_count;
        return false;
      }
      entry->retiring = true;
      if (entry->identity.has_value()) {
        shared_handles_.erase(*entry->identity);
      }
      entry->idle.wait(
          lock, [&entry]() { return entry->active_executions == 0; });
    }

    // QNN teardown may call vendor code. Never execute it while holding the
    // registry lock, otherwise callbacks can deadlock future init/execute.
    std::forward<DestroyFn>(destroy)(handle);
    {
      std::lock_guard<std::mutex> guard(mutex_);
      auto found = entries_.find(handle);
      if (found != entries_.end() && found->second == entry) {
        entries_.erase(found);
      }
    }
    return true;
  }

  bool contains(Handle* handle) const {
    std::lock_guard<std::mutex> guard(mutex_);
    auto entry = entries_.find(handle);
    return entry != entries_.end() && !entry->second->retiring;
  }

 private:
  void finish_execution(const std::shared_ptr<Entry>& entry) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (--entry->active_executions == 0 && entry->retiring) {
      entry->idle.notify_all();
    }
  }

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Entry>> shared_handles_;
  std::unordered_map<Handle*, std::shared_ptr<Entry>> entries_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
