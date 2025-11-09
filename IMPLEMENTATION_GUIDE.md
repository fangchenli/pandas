# Implementation Guide for 'both' Option in Resample closed Parameter

## Quick Reference: Current 'left' vs 'right' Behavior

```
CLOSED='LEFT'
┌──────────────────────────────┐
│ Bin [start, end)             │
│ Includes start, excludes end │
└──────────────────────────────┘

Example: [2000-01-01 00:00, 2000-01-01 00:05)
- 2000-01-01 00:00:00  ✓ INCLUDED
- 2000-01-01 00:04:59  ✓ INCLUDED
- 2000-01-01 00:05:00  ✗ EXCLUDED


CLOSED='RIGHT'
┌──────────────────────────────┐
│ Bin (start, end]             │
│ Excludes start, includes end │
└──────────────────────────────┘

Example: (2000-01-01 00:00, 2000-01-01 00:05]
- 2000-01-01 00:00:00  ✗ EXCLUDED
- 2000-01-01 00:04:59  ✓ INCLUDED
- 2000-01-01 00:05:00  ✓ INCLUDED


CLOSED='BOTH' (proposed)
┌──────────────────────────────┐
│ Bin [start, end]             │
│ Includes both start and end  │
└──────────────────────────────┘

Example: [2000-01-01 00:00, 2000-01-01 00:05]
- 2000-01-01 00:00:00  ✓ INCLUDED
- 2000-01-01 00:04:59  ✓ INCLUDED
- 2000-01-01 00:05:00  ✓ INCLUDED
```

## Core Binning Logic Differences

### Current Implementation in lib.pyx

The `generate_bins_dt64()` function contains the core logic:

```cython
# For closed='right'
right_closed = (closed == "right")

if right_closed:
    # Use <= for upper bound comparison
    while j < lenidx and values[j] <= r_bin:
        j += 1
else:
    # Use < for upper bound comparison
    while j < lenidx and values[j] < r_bin:
        j += 1
```

### For 'both' Implementation

The 'both' option needs to include values at both boundaries:

```
closed='left':  values[j] < r_bin       (exclude right edge)
closed='right': values[j] <= r_bin      (include right edge)
closed='both':  values[j] <= r_bin      (include right edge)
                + handle left edge differently
```

But there's a complexity: with 'both', you get overlapping bins at boundaries.

**Example with 5-minute intervals:**
```
Time series: [00:00, 00:01, 00:05, 00:06, 00:10, 00:11]

closed='left':
  [00:00, 00:05): [00:00, 00:01]
  [00:05, 00:10): [00:05, 00:06]
  [00:10, 00:15): [00:10, 00:11]

closed='right':
  (00:00, 00:05]: [00:00, 00:01, 00:05]
  (00:05, 00:10]: [00:06, 00:10]
  (00:10, 00:15]: [00:11]

closed='both':
  [00:00, 00:05]: [00:00, 00:01, 00:05]
  [00:05, 00:10]: [00:05, 00:06, 00:10]
  [00:10, 00:15]: [00:10, 00:11]
```

Note: Values at boundaries appear in multiple bins!

## Implementation Strategy

### Step 1: Type Annotations (Easy)

Files to update:
- `pandas/core/generic.py:8903`
- `pandas/core/resample.py:2386`
- `pandas/core/resample.py:2850`
- `pandas/core/resample.py:2922`
- `pandas/core/resample.py:2996`

Change all occurrences of:
```python
closed: Literal["right", "left"] | None = None
```

To:
```python
closed: Literal["right", "left", "both"] | None = None
```

And:
```python
closed: Literal["right", "left"] = "left"
```

To:
```python
closed: Literal["right", "left", "both"] = "left"
```

### Step 2: Validation (Easy)

File: `pandas/core/resample.py:2404`

Change:
```python
if closed not in {None, "left", "right"}:
```

To:
```python
if closed not in {None, "left", "right", "both"}:
```

### Step 3: Core C Function (MOST CRITICAL)

File: `pandas/_libs/lib.pyx:900-958`

This is where the real work happens. Current logic:

```cython
if right_closed:
    # closed='right': <= comparison
    for i in range(0, lenbin - 1):
        r_bin = binner[i + 1]
        while j < lenidx and values[j] <= r_bin:
            j += 1
        bins[bc] = j
        bc += 1
else:
    # closed='left': < comparison
    for i in range(0, lenbin - 1):
        r_bin = binner[i + 1]
        while j < lenidx and values[j] < r_bin:
            j += 1
        bins[bc] = j
        bc += 1
```

For 'both', we need to handle both left and right boundaries:

**Option A: Use <= like 'right'**
```cython
elif closed == "both":
    # Include both left and right boundaries
    for i in range(0, lenbin - 1):
        r_bin = binner[i + 1]
        while j < lenidx and values[j] <= r_bin:
            j += 1
        bins[bc] = j
        bc += 1
```

But this alone doesn't handle the left boundary properly. You need additional logic.

**Option B: Manual iteration for 'both'**
```cython
elif closed == "both":
    # For 'both': include both left and right boundaries
    # This creates overlapping bins at boundaries
    for i in range(0, lenbin - 1):
        l_bin = binner[i]
        r_bin = binner[i + 1]
        bin_start = j
        # Count values in [l_bin, r_bin]
        while j < lenidx and values[j] <= r_bin:
            j += 1
        # But also include values from previous bin that equal l_bin
        if i > 0:
            # Need to adjust to not double-count
            pass
        bins[bc] = j
        bc += 1
```

### Step 4: Edge Adjustment Functions

Files to update with 'both' handling:
- `pandas/core/resample.py:_adjust_bin_edges` (lines 2632-2666)
- `pandas/core/resample.py:_get_time_delta_bins` (lines 2668-2705)
- `pandas/core/resample.py:_get_timestamp_range_edges` (lines 2845-2915)
- `pandas/core/resample.py:_adjust_dates_anchored` (lines 2992-3073)

**Pattern for all these:**
```python
if self.closed == "right":
    # RIGHT logic
    ...
elif self.closed == "left":
    # LEFT logic (might be in else clause currently)
    ...
elif self.closed == "both":
    # BOTH logic - probably combine aspects of left and right
    ...
```

### Step 5: Upsampling Logic

File: `pandas/core/resample.py:_adjust_binner_for_upsample` (lines 2098-2108)

Current:
```python
if self.closed == "right":
    binner = binner[1:]
else:
    binner = binner[:-1]
```

For 'both', you might not need to adjust:
```python
if self.closed == "right":
    binner = binner[1:]
elif self.closed == "left":
    binner = binner[:-1]
elif self.closed == "both":
    # Don't adjust for 'both'?
    pass
```

### Step 6: Label Handling

File: `pandas/core/resample.py:_get_datetime_bins` (lines 2613-2618)

Current logic handles 'right' specially, then checks label separately.

For 'both', need to determine if label handling changes.

### Step 7: Add Comprehensive Tests

Create tests in: `pandas/tests/resample/test_datetime_index.py`

Test cases needed:
```python
def test_resample_closed_both_basic():
    """Test basic functionality with closed='both'"""

def test_resample_closed_both_vs_left_right():
    """Compare results of 'both' with 'left' and 'right'"""

def test_resample_closed_both_boundary_values():
    """Test that boundary values are included in both bins"""

def test_resample_closed_both_upsampling():
    """Test upsampling with closed='both'"""

def test_resample_closed_both_with_different_frequencies():
    """Test with various frequency offsets"""

def test_resample_closed_both_timedelta_index():
    """Test with TimedeltaIndex"""

def test_resample_closed_both_period_index():
    """Test with PeriodIndex"""
```

## Key Decision Points

### 1. How to Handle Overlapping Bins

With `closed='both'`, values at bin boundaries will appear in multiple bins:

**Option 1: Keep in both bins (current IntervalIndex behavior)**
- Simpler to implement
- Aggregations will see values multiple times
- User must handle deduplication if needed

**Option 2: Decide on some precedence**
- Only left bin gets the value
- Only right bin gets the value
- Some other rule
- More complex but might be more intuitive

### 2. Default Value for 'both'

Should 'both' have a special default, or should it be explicitly specified?

Current defaults (in order of precedence):
1. If freq in {ME, YE, QE, BME, BYE, BQE, W}: default='right'
2. If origin in [end, end_day]: default='right'
3. Otherwise: default='left'

Probably 'both' should not be a default - users must specify it.

### 3. Interaction with 'label' Parameter

The 'label' parameter determines which bin edge to use for the label.

For 'both', what makes sense?
- label='left': use left edge of bin
- label='right': use right edge of bin
- label='both': use both? Not clear what this means

Probably 'label' should be constrained to 'left'/'right' even when closed='both'.

## Testing Strategy

Before implementing 'both', create a small test script:

```python
import pandas as pd
import numpy as np

# Create simple test data
dates = pd.date_range('2000-01-01', periods=11, freq='1min')
data = pd.Series(range(11), index=dates)

# Test current behavior
print("Original data:")
print(data)

print("\nclosed='left':")
print(data.resample('5min', closed='left').mean())

print("\nclosed='right':")
print(data.resample('5min', closed='right').mean())

# After implementation, test 'both'
# print("\nclosed='both':")
# print(data.resample('5min', closed='both').mean())
```

## References

- Issue: #1918 (mentioned in git branch name)
- Related: IntervalIndex already supports 'both' (see `pandas/core/arrays/interval.py`)
- Related: IntervalIndex closed parameter in `pandas/core/indexes/interval.py`

## Checklist for Implementation

- [ ] Update type annotations (7 locations)
- [ ] Update validation logic (1 location)
- [ ] Update C library function (1 critical location)
- [ ] Update edge adjustment functions (4 locations)
- [ ] Update upsampling logic (1 location)
- [ ] Update label handling (1 location)
- [ ] Add documentation/docstrings
- [ ] Add comprehensive tests (8+ test cases)
- [ ] Test with different index types (DatetimeIndex, TimedeltaIndex, PeriodIndex)
- [ ] Test edge cases (empty data, single value, etc.)
- [ ] Run existing test suite to ensure no regressions
- [ ] Create PR with clear description
