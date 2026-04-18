#pragma once

#include <cstddef>

namespace OwnTensor
{
    class SizeClass {
        public:
            // rounding upto appropriate size grp
            static size_t round_size(size_t size) {
                if (size < 512) return 512;
                if (size <= kSmallSize) {
                    return ((size + 511) / 512) * 512;
                }
                
                size_t rounded = 1;
                while (rounded < size) rounded <<= 1;
                
                // If it's a power of 2, we are done
                if (rounded == size) return size;
                
                // Otherwise, divide the range [rounded/2, rounded] into 8 parts
                size_t prev_pow2 = rounded >> 1;
                size_t step = (rounded - prev_pow2) / 8;
                
                return prev_pow2 + ((size - prev_pow2 + step - 1) / step) * step;
            }

            /**
             * Round of 2 divisions
             * 
             * Basically instead of the standard 512 bytes alignment with the allocation,
             * we can use rounding based on the power of 2's
             * 
             * We can split stuffs in this ratio
             * [256: 1, 512: 2, 1024:4, >:8]
             * 
             * Example:
             *      For a 1200 byte allocation, since it is between 1024 (2^10) and 2048 (2^11)
             *      we do 4 divisions for this region so 1280, 1536, 1792 bytes instead of the usual
             *      512 bytes alignment
             * 
             *      So for 1200 bytes - instead of aligning to 1536 bytes, using this we can allocate just 
             *      1024 + 256 = 1280 bytes which gives less unused memory and reduce fragmentation
             */

            // to use small pool whenever possible
            static bool is_small(size_t size) {
                return size <= kSmallSize;
            }

            // Thresholds
            static constexpr size_t kSmallSize = 1048576;           // 1MB - small pool threshold
            static constexpr size_t kSmallBuffer = 1048576 * 2;     // 2MB
            static constexpr size_t kLargeBuffer = 1048576 * 20;    // 20MB
            static constexpr size_t kMinLargeAlloc = 1048576 * 10;  // 10MB
            static constexpr size_t kRoundLarge = 2097152;          // 2MB alignment
    };
}