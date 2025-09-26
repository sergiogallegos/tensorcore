#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <cstdint>

namespace tensorcore {

/**
 * @brief Memory pool for efficient tensor allocation
 * 
 * This class provides a memory pool that pre-allocates memory blocks
 * and reuses them to avoid frequent malloc/free operations, which
 * can be expensive for large tensors.
 */
class MemoryPool {
public:
    // Constructor and destructor
    MemoryPool(size_t initial_capacity = 1024 * 1024); // 1MB initial capacity
    ~MemoryPool();
    
    // Disable copy constructor and assignment
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    // Memory allocation
    void* allocate(size_t size);
    void deallocate(void* ptr, size_t size);
    
    // Pool management
    void reserve(size_t capacity);
    void clear();
    size_t get_total_allocated() const;
    size_t get_total_capacity() const;
    
    // Statistics
    size_t get_allocation_count() const;
    size_t get_deallocation_count() const;
    double get_utilization_ratio() const;

private:
    struct Block {
        void* data;
        size_t size;
        bool in_use;
        Block* next;
    };
    
    // Memory blocks organized by size
    std::unordered_map<size_t, std::vector<std::unique_ptr<Block>>> free_blocks_;
    std::unordered_map<void*, std::unique_ptr<Block>> allocated_blocks_;
    
    // Statistics
    mutable std::mutex mutex_;
    size_t total_allocated_;
    size_t total_capacity_;
    size_t allocation_count_;
    size_t deallocation_count_;
    
    // Helper functions
    Block* find_free_block(size_t size);
    Block* create_new_block(size_t size);
    void return_block(Block* block);
    size_t align_size(size_t size);
};

/**
 * @brief Global memory pool instance
 */
extern MemoryPool& get_global_memory_pool();

/**
 * @brief RAII wrapper for memory pool allocations
 */
class PooledAllocation {
public:
    PooledAllocation(size_t size);
    ~PooledAllocation();
    
    void* get() const { return ptr_; }
    size_t size() const { return size_; }
    
    // Disable copy
    PooledAllocation(const PooledAllocation&) = delete;
    PooledAllocation& operator=(const PooledAllocation&) = delete;
    
    // Allow move
    PooledAllocation(PooledAllocation&& other) noexcept;
    PooledAllocation& operator=(PooledAllocation&& other) noexcept;

private:
    void* ptr_;
    size_t size_;
};

} // namespace tensorcore
