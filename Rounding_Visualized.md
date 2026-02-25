# Round to Nearest Even (RTNE) — Full Explanation

---

## Part 1: WHY not normal rounding?

### Normal Rounding (what we learn in school)

Normal rounding says: if the digit after the cutoff is >= 5, round UP. Otherwise round DOWN.

Example: Round these to the nearest integer:
- `2.3` → `2` (down)
- `2.7` → `3` (up)
- `2.5` → `3` (up, because >= 5)
- `3.5` → `4` (up, because >= 5)
- `4.5` → `5` (up, because >= 5)

### The Problem with Normal Rounding

Notice that `.5` **ALWAYS rounds UP**. Never down. Over millions of operations, this creates a **systematic upward drift**.

Example: Average these 4 numbers after rounding:
- Original: `0.5, 1.5, 2.5, 3.5`
- True average: `(0.5 + 1.5 + 2.5 + 3.5) / 4 = 2.0`
- Normal rounding: `1, 2, 3, 4`
- Rounded average: `(1 + 2 + 3 + 4) / 4 = 2.5` ← **WRONG! Drifted up by 0.5**

### Round to Nearest Even (RTNE) fixes this

RTNE says: if you are EXACTLY at the halfway point, round to the nearest **EVEN** number.

Same example with RTNE:
- `0.5` → `0` (0 is even, 1 is odd → pick 0)
- `1.5` → `2` (1 is odd, 2 is even → pick 2)
- `2.5` → `2` (2 is even, 3 is odd → pick 2)
- `3.5` → `4` (3 is odd, 4 is even → pick 4)
- Rounded average: `(0 + 2 + 2 + 4) / 4 = 2.0` ← **CORRECT!**

Half the time we round up, half the time we round down. **No drift.**

In deep learning, we do billions of multiplications and additions. Even a tiny drift destroys training. That is why IEEE 754 mandates RTNE as the default.

---

## Part 2: The Setup — What are we actually doing?

We have a 32-bit number `u`. We want to keep only the top 16 bits.

```text
Bit positions:   31  30  ...  17  16  |  15  14  ...  1   0
                 ←── KEEP (BF16) ──→  |  ←── DISCARD (Tail) ──→
```

- **Bit 16** = the LSB (Least Significant Bit) of the result we keep
- **Bits 15 to 0** = the "Tail" that gets chopped off

### What is an "even" or "odd" result?

- If **Bit 16 = 0**, the result is **EVEN**
- If **Bit 16 = 1**, the result is **ODD**

(Just like in decimal: a number ending in 0, 2, 4, 6, 8 is even)

### What is "Halfway"?

The tail (bits 15-0) can range from `0x0000` to `0xFFFF`.

- `0x0000` = the tail is zero (no rounding needed)
- `0xFFFF` = the tail is almost at the next number
- `0x8000` = **EXACTLY halfway** between "round down" and "round up"

```text
          0x0000         0x8000          0xFFFF
            |               |               |
  Round Down zone      HALFWAY       Round Up zone
```

---

## Part 3: The Code

```cpp
uint32_t lsb = (u >> 16) & 1u;           // Extract bit 16 (0 = even, 1 = odd)
uint32_t rounding_bias = 0x7FFFu;        // Default bias
if (lsb) rounding_bias = 0x8000u;        // If odd, use higher bias
u += rounding_bias;                       // Add bias to trigger carry (or not)
return static_cast<uint16_t>(u >> 16);    // Take top 16 bits
```

The trick: when we add the bias to `u`, if the tail + bias >= 0x10000, it creates a **carry** that increments bit 16. Otherwise bit 16 stays the same.

---

## Part 4: ALL 6 Cases — Fully Expanded

### CASE 1: Tail < Halfway, LSB = 0 (Even)

```text
Bit 16:  0
Tail:    0x3000  (less than 0x8000, so below halfway)
Bias:    0x7FFF  (because lsb = 0)

Addition of tail + bias:
  0x3000
+ 0x7FFF
--------
  0xAFFF  (this is less than 0x10000, so NO CARRY)

Result: Bit 16 stays 0. Rounded DOWN. ✓ (correct, tail was below halfway)
```

### CASE 2: Tail < Halfway, LSB = 1 (Odd)

```text
Bit 16:  1
Tail:    0x3000  (less than 0x8000, so below halfway)
Bias:    0x8000  (because lsb = 1)

Addition of tail + bias:
  0x3000
+ 0x8000
--------
  0xB000  (this is less than 0x10000, so NO CARRY)

Result: Bit 16 stays 1. Rounded DOWN. ✓ (correct, tail was below halfway)
```

### CASE 3: Tail > Halfway, LSB = 0 (Even)

```text
Bit 16:  0
Tail:    0xC000  (greater than 0x8000, so above halfway)
Bias:    0x7FFF  (because lsb = 0)

Addition of tail + bias:
  0xC000
+ 0x7FFF
--------
 0x13FFF  (this is >= 0x10000, so CARRY!)

Result: Bit 16 goes from 0 to 1. Rounded UP. ✓ (correct, tail was above halfway)
```

### CASE 4: Tail > Halfway, LSB = 1 (Odd)

```text
Bit 16:  1
Tail:    0xC000  (greater than 0x8000, so above halfway)
Bias:    0x8000  (because lsb = 1)

Addition of tail + bias:
  0xC000
+ 0x8000
--------
 0x14000  (this is >= 0x10000, so CARRY!)

Result: Bit 16 goes from 1 to 0 (1+1=10 in binary). Rounded UP. ✓ (correct, tail was above halfway)
```

### CASE 5: Tail = EXACTLY Halfway, LSB = 0 (Already Even)

```text
Bit 16:  0  ← already EVEN
Tail:    0x8000  (EXACTLY halfway)
Bias:    0x7FFF  (because lsb = 0)

Addition of tail + bias:
  0x8000
+ 0x7FFF
--------
  0xFFFF  (this is less than 0x10000, so NO CARRY)

Result: Bit 16 stays 0 (EVEN). Did NOT round up. ✓
Why? Because we are already EVEN, so RTNE says STAY HERE.
```

### CASE 6: Tail = EXACTLY Halfway, LSB = 1 (Currently Odd)

```text
Bit 16:  1  ← currently ODD
Tail:    0x8000  (EXACTLY halfway)
Bias:    0x8000  (because lsb = 1)

Addition of tail + bias:
  0x8000
+ 0x8000
--------
 0x10000  (this is >= 0x10000, so CARRY!)

Result: Bit 16 goes from 1 to 0 (now EVEN). Rounded UP. ✓
Why? Because we were ODD, so RTNE says ROUND UP to become EVEN.
```

---

## Part 5: Summary Table

| Case | Tail | LSB (Bit 16) | Bias | Tail + Bias | Carry? | Action | Final LSB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | < Half | 0 (Even) | 0x7FFF | < 0x10000 | No | Round Down | 0 |
| 2 | < Half | 1 (Odd) | 0x8000 | < 0x10000 | No | Round Down | 1 |
| 3 | > Half | 0 (Even) | 0x7FFF | >= 0x10000 | **Yes** | Round Up | 1 |
| 4 | > Half | 1 (Odd) | 0x8000 | >= 0x10000 | **Yes** | Round Up | 0 |
| **5** | **= Half** | **0 (Even)** | **0x7FFF** | **0xFFFF** | **No** | **Stay Even** | **0** |
| **6** | **= Half** | **1 (Odd)** | **0x8000** | **0x10000** | **Yes** | **Round to Even** | **0** |

**Cases 1-4**: Normal rounding (below half → down, above half → up). The LSB/bias difference doesn't matter here because the result is the same either way.

**Cases 5-6**: The RTNE tie-breaker. This is the ONLY place where the LSB matters. By choosing `0x7FFF` vs `0x8000` as the bias, we get EXACTLY the right behavior: stay even if already even, round up if currently odd.
