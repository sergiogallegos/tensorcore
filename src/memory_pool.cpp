#include "tensorcore/memory_pool.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace tensorcore {

MemoryPool::MemoryPool(size_t initial_capacity)
    : total_allocated_(0), total_capacity_(initial_capacity), 
      allocation_count_(0), deallocation_count_(0) {
    reserve(initial_capacity);
}

MemoryPool::~MemoryPool() {
    clear();
}

void* MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t aligned_size = align_size(size);
    
    // Try to find a free block of the right size
    Block* block = find_free_block(aligned_size);
    if (!block) {
        // Create a new block if none available
        block = create_new_block(aligned_size);
    }
    
    if (block) {
        block->in_use = true;
        allocated_blocks_[block->data] = std::unique_ptr<Block>(block);
        total_allocated_ += aligned_size;
        allocation_count_++;
        return block->data;
    }
    
    // Fallback to standard allocation if pool is full
    void* ptr = std::malloc(size);
    if (ptr) {
        total_allocated_ += size;
        allocation_count_++;
    }
    return ptr;
}

void MemoryPool::deallocate(void* ptr, size_t size) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocated_blocks_.find(ptr);
    if (it != allocated_blocks_.end()) {
        // Return to pool
        Block* block = it->second.release();
        block->in_use = false;
        return_block(block);
        allocated_blocks_.erase(it);
        total_allocated_ -= size;
        deallocation_count_++;
    } else {
        // Standard deallocation
        std::free(ptr);
        total_allocated_ -= size;
        deallocation_count_++;
    }
}

void MemoryPool::reserve(size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (capacity > total_capacity_) {
        total_capacity_ = capacity;
        
        // Pre-allocate some common block sizes
        std::vector<size_t> common_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
        
        for (size_t size : common_sizes) {
            if (size <= capacity) {
                // Create a few blocks of each size
                for (int i = 0; i < 3; ++i) {
                    Block* block = create_new_block(size);
                    if (block) {
                        block->in_use = false;
                        return_block(block);
                    }
                }
            }
        }
    }
}

void MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Free all allocated blocks
    for (auto& pair : allocated_blocks_) {
        std::free(pair.first);
    }
    allocated_blocks_.clear();
    
    // Free all free blocks
    for (auto& pair : free_blocks_) {
        for (auto& block : pair.second) {
            std::free(block->data);
        }
    }
    free_blocks_.clear();
    
    total_allocated_ = 0;
    total_capacity_ = 0;
}

size_t MemoryPool::get_total_allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_;
}

size_t MemoryPool::get_total_capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_capacity_;
}

size_t MemoryPool::get_allocation_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocation_count_;
}

size_t MemoryPool::get_deallocation_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return deallocation_count_;
}

double MemoryPool::get_utilization_ratio() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (total_capacity_ == 0) return 0.0;
    return static_cast<double>(total_allocated_) / static_cast<double>(total_capacity_);
}

MemoryPool::Block* MemoryPool::find_free_block(size_t size) {
    auto it = free_blocks_.find(size);
    if (it != free_blocks_.end() && !it->second.empty()) {
        Block* block = it->second.back().release();
        it->second.pop_back();
        return block;
    }
    return nullptr;
}

MemoryPool::Block* MemoryPool::create_new_block(size_t size) {
    void* data = std::malloc(size);
    if (!data) return nullptr;
    
    Block* block = new Block{data, size, false, nullptr};
    return block;
}

void MemoryPool::return_block(Block* block) {
    if (!block) return;
    
    size_t size = block->size;
    auto it = free_blocks_.find(size);
    if (it == free_blocks_.end()) {
        free_blocks_[size] = std::vector<std::unique_ptr<Block>>();
    }
    
    free_blocks_[size].push_back(std::unique_ptr<Block>(block));
}

size_t MemoryPool::align_size(size_t size) {
    // Align to 8-byte boundary for better performance
    return (size + 7) & ~7;
}

// Global memory pool instance
static MemoryPool global_pool(1024 * 1024 * 16); // 16MB initial capacity

MemoryPool& get_global_memory_pool() {
    return global_pool;
}

// PooledAllocation implementation
PooledAllocation::PooledAllocation(size_t size)
    : ptr_(get_global_memory_pool().allocate(size)), size_(size) {
}

PooledAllocation::~PooledAllocation() {
    if (ptr_) {
        get_global_memory_pool().deallocate(ptr_, size_);
    }
}

PooledAllocation::PooledAllocation(PooledAllocation&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

PooledAllocation& PooledAllocation::operator=(PooledAllocation&& other) noexcept {
    if (this != &other) {
        if (ptr_) {
            get_global_memory_pool().deallocate(ptr_, size_);
        }
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

} // namespace tensorcore
