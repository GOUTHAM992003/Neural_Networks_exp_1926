#pragma once

#include <cstddef>

namespace OwnTensor
{
    class SizeClass {
        public:
            // rounding upto appropriate size grp
            static size_t round_size(size_t size) {
                
                // DEFAULT:
                // if (size < kSmallSize) {
                //    // Rounding to 512 byte allocations for small allocations
                //     return ((size + 511) / 512) * 512;
                // } else if (size < kSmallBuffer) {
                //     // Round to 2Mb for medium allocations
                //     return kSmallBuffer;
                // } else if (size < kLargeBuffer) {
                //     // Round to 20MB for large allocations
                //     return kLargeBuffer;
                // } else {
                //     // Round to 2MB alignment for very large allocations
                //     return ((size + kRoundLarge - 1) / kRoundLarge) * kRoundLarge;
                // }

                // KURRA's OPTIMIZATION
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
            

                // CUSTOM SPLIT - WORKED BETTER ON PRE OPTIM 44M
                // if (size < MB_1) {
                //     return ((size + 511) / 512) * 512;
                // }
                // else if (size < MB_2) {
                //     return MB_2;
                // }
                // else if (size < MB_10) {
                //     return MB_10;
                // }
                // else if (size < MB_13) {
                //     return MB_13;
                // }
                // else if (size < MB_20) {
                //     return MB_20;
                // }
                // else {
                //     return ((size + ALIGN_MB_2 - 1) / ALIGN_MB_2) * ALIGN_MB_2;
                // }
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

            static constexpr size_t MB_1 = 1048576;
            static constexpr size_t MB_2 = 2097152;
            static constexpr size_t MB_10 = 10485760;
            static constexpr size_t MB_13 = 13631488;
            static constexpr size_t MB_20 = 20971520;
            static constexpr size_t ALIGN_MB_2 = 2097152;


    };
}