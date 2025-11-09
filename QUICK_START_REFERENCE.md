# Quick Start Reference: Understanding Resample closed Parameter

## The Key Files to Understand

### 1. Entry Point: pandas/core/generic.py (line 8903)
```python
def resample(
    self,
    rule,
    closed: Literal["right", "left"] | None = None,  # CURRENTLY: only 'left' or 'right'
    ...
```

### 2. Validation & Storage: pandas/core/resample.py (line 2404)
```python
if closed not in {None, "left", "right"}:
    raise ValueError(f"Unsupported value {closed} for `closed`")
```

### 3. THE CRITICAL FUNCTION: pandas/_libs/lib.pyx (lines 936-951)

This Cython function is where the actual binning happens:

```cython
if right_closed:
    # closed='right': INCLUDES right boundary
    for i in range(0, lenbin - 1):
        r_bin = binner[i + 1]
        while j < lenidx and values[j] <= r_bin:  # <-- LESS THAN OR EQUAL
            j += 1
        bins[bc] = j
        bc += 1
else:
    # closed='left': EXCLUDES right boundary
    for i in range(0, lenbin - 1):
        r_bin = binner[i + 1]
        while j < lenidx and values[j] < r_bin:  # <-- LESS THAN
            j += 1
        bins[bc] = j
        bc += 1
```

**This is the most important function to understand.**

### 4. TimedeltaIndex Handling: pandas/core/resample.py (lines 2688-2699)

Shows how closed affects binner calculation:

```python
start, end = ax.min(), ax.max()

if self.closed == "right":
    end += self.freq  # Different for right-closed

labels = binner = timedelta_range(
    start=start, end=end, freq=self.freq, name=ax.name
)

end_stamps = labels
if self.closed == "left":
    end_stamps += self.freq  # Different for left-closed

bins = ax.searchsorted(end_stamps, side=self.closed)  # <-- Uses closed directly!
```

### 5. DatetimeIndex Edge Adjustment: pandas/core/resample.py (lines 2648-2658)

Special handling for month-end and other special frequencies:

```python
if self.closed == "right":
    # GH 21459, GH 9119: Adjust bins for right-closed intervals
    edges_dti = binner.tz_localize(None)
    edges_dti = (
        edges_dti
        + Timedelta(days=1).as_unit(edges_dti.unit)
        - Timedelta(1, unit=edges_dti.unit).as_unit(edges_dti.unit)
    )
    bin_edges = edges_dti.tz_localize(binner.tz).asi8
else:
    bin_edges = binner.asi8
```

### 6. Upsampling: pandas/core/resample.py (lines 2104-2107)

How upsampling adjusts the binner:

```python
if self.closed == "right":
    binner = binner[1:]  # Remove first element
else:
    binner = binner[:-1]  # Remove last element
```

### 7. Complex Date Adjustment: pandas/core/resample.py (lines 3042-3066)

This function handles the complex interaction between closed and origin parameters:

```python
if closed == "right":
    if foffset > 0:
        fresult_int = first._value - foffset
    else:
        fresult_int = first._value - freq_value

    if loffset > 0:
        lresult_int = last._value + (freq_value - loffset)
    else:
        lresult_int = last._value
else:  # closed == 'left'
    if foffset > 0:
        fresult_int = first._value - foffset
    else:
        fresult_int = first._value

    if loffset > 0:
        lresult_int = last._value + (freq_value - loffset)
    else:
        lresult_int = last._value + freq_value
```

## Understanding the Behavior

### closed='left' [start, end)
- Lower bound: INCLUDED
- Upper bound: EXCLUDED
- Implemented via: `values[j] < r_bin` (strictly less than)
- Used for: Most frequencies by default (hourly, daily, etc.)

### closed='right' (start, end]
- Lower bound: EXCLUDED
- Upper bound: INCLUDED
- Implemented via: `values[j] <= r_bin` (less than or equal)
- Used for: Month-end, year-end, week-end frequencies by default

### closed='both' [start, end] (TO BE IMPLEMENTED)
- Lower bound: INCLUDED
- Upper bound: INCLUDED
- Implementation: Would need `values[j] <= r_bin` but with special handling
- Issue: Creates overlapping bins at boundaries

## How to Trace Through the Code

Example: `df.resample('5min', closed='left').mean()`

1. **Entry:** `df.resample(rule='5min', closed='left')` in `pandas/core/generic.py`

2. **Creates TimeGrouper:** `TimeGrouper(freq='5min', closed='left')` in `pandas/core/resample.py:2355`
   - Validates closed='left' is valid (line 2404)
   - Stores `self.closed = 'left'` (line 2459)

3. **Creates Resampler:** Resampler stores TimeGrouper and accesses `self.closed` via property

4. **Gets bins:** Calls appropriate method based on index type:
   - `_get_datetime_bins()` for DatetimeIndex
   - `_get_time_delta_bins()` for TimedeltaIndex
   - `_get_period_bins()` for PeriodIndex

5. **Core binning:** Calls `lib.generate_bins_dt64(ax_values, bin_edges, closed='left', ...)`
   - right_closed = False
   - Loop uses: `while j < lenidx and values[j] < r_bin:`

6. **Applies aggregation:** Uses bins to group data and apply mean()

7. **Returns result:** Series/DataFrame with resampled data

## Most Important Changes for 'both'

1. **Update validation** (1 line change):
   ```python
   if closed not in {None, "left", "right", "both"}:
   ```

2. **Update C function** (most complex):
   ```cython
   elif closed == "both":
       # Logic to include both boundaries
       # This will create overlapping bins
   ```

3. **Update edge calculations** (multiple locations):
   - Each function that has `if closed == "right"` / `else` needs an elif for "both"

4. **Update tests** (comprehensive):
   - Verify boundary values appear in both bins
   - Test with different frequencies
   - Test with different index types

## Testing Current Implementation

```python
import pandas as pd

# Create test data
dates = pd.date_range('2000-01-01 00:00:00', periods=11, freq='1min')
values = list(range(11))
df = pd.DataFrame({'value': values}, index=dates)

print("Original data:")
print(df)

print("\nResampled with closed='left':")
result_left = df.resample('5min', closed='left', label='left').mean()
print(result_left)

print("\nResampled with closed='right':")
result_right = df.resample('5min', closed='right', label='right').mean()
print(result_right)

# After implementation:
# print("\nResampled with closed='both':")
# result_both = df.resample('5min', closed='both', label='left').mean()
# print(result_both)
```

Expected output would show how boundary values are handled differently in each case.
