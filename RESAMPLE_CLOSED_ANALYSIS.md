# Resample Function `closed` Parameter Implementation Analysis

## Overview
The `closed` parameter in pandas resample controls which side of the bin interval is closed (inclusive) during resampling operations. Currently, it only supports 'left' and 'right' options. This document shows where it's defined and how these options are handled throughout the codebase.

## 1. Main Entry Point: NDFrame.resample()

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/generic.py` (lines 8900-8928)

```python
def resample(
    self,
    rule,
    closed: Literal["right", "left"] | None = None,  # TYPE ANNOTATION
    label: Literal["right", "left"] | None = None,
    ...
) -> Resampler:
    """
    Parameters
    ----------
    closed : {{'right', 'left'}}, default None
        Which side of bin interval is closed. The default is 'left'
        for all frequency offsets except for 'ME', 'YE', 'QE', 'BME',
        'BA', 'BQE', and 'W' which all have a default of 'right'.
    """
```

**Key Points:**
- Type annotation only accepts 'left' or 'right'
- Documentation mentions default behavior based on frequency type
- Default is None, which gets resolved to either 'left' or 'right' based on frequency

## 2. TimeGrouper Class: Definition & Validation

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py` (lines 2355-2496)

### Parameter Definition (lines 2386):
```python
def __init__(
    self,
    ...
    closed: Literal["left", "right"] | None = None,
    ...
) -> None:
```

### Validation (lines 2404-2405):
```python
if closed not in {None, "left", "right"}:
    raise ValueError(f"Unsupported value {closed} for `closed`")
```

**To support 'both' option, you would need to change this to:**
```python
if closed not in {None, "left", "right", "both"}:
    raise ValueError(f"Unsupported value {closed} for `closed`")
```

### Default Resolution Logic (lines 2434-2459):
```python
end_types = {"ME", "YE", "QE", "BME", "BYE", "BQE", "W"}
rule = freq.rule_code
if rule in end_types or ("-" in rule and rule[: rule.find("-")] in end_types):
    if closed is None:
        closed = "right"
    if label is None:
        label = "right"
else:
    # The backward resample sets ``closed`` to ``'right'`` by default
    # since the last value should be considered as the edge point for
    # the last bin. When origin in "end" or "end_day", the value for a
    # specific ``Timestamp`` index stands for the resample result from
    # the current ``Timestamp`` minus ``freq`` to the current
    # ``Timestamp`` with a right close.
    if origin in ["end", "end_day"]:
        if closed is None:
            closed = "right"
        if label is None:
            label = "right"
    else:
        if closed is None:
            closed = "left"
        if label is None:
            label = "left"

self.closed = closed  # STORED IN INSTANCE
```

## 3. Resampler Class: Attributes & Storage

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py` (lines 118-196)

### Attributes Definition (lines 148-155):
```python
_attributes = [
    "freq",
    "closed",      # <-- Stored as attribute
    "label",
    "convention",
    "origin",
    "offset",
]
```

### Retrieval via __getattr__ (lines 198-200):
```python
@final
def __getattr__(self, attr: str):
    if attr in self._internal_names_set:
        return object.__getattribute__(self, attr)
```

The `closed` parameter is accessed throughout the Resampler class via `self.closed`.

## 4. Core Binning Logic: Where 'left' and 'right' Diverge

### 4a. DatetimeIndex Binning (_get_datetime_bins)

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py` (lines 2574-2630)

```python
def _get_datetime_bins(self, ax: DatetimeIndex):
    # ... setup code ...

    # CRITICAL: Call to C library function with closed parameter
    bins = lib.generate_bins_dt64(
        ax_values, bin_edges, self.closed, hasnans=ax.hasnans
    )

    if self.closed == "right":
        labels = binner
        if self.label == "right":
            labels = labels[1:]
    elif self.label == "right":
        labels = labels[1:]
    # ...
```

**Edge adjustment for closed='right' (lines 2648-2658):**
```python
if self.closed == "right":
    # GH 21459, GH 9119: Adjust the bins relative to the wall time
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

### 4b. TimedeltaIndex Binning (_get_time_delta_bins)

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py` (lines 2668-2705)

```python
def _get_time_delta_bins(self, ax: TimedeltaIndex):
    start, end = ax.min(), ax.max()

    if self.closed == "right":
        end += self.freq  # <-- Different end point for right

    labels = binner = timedelta_range(
        start=start, end=end, freq=self.freq, name=ax.name
    )

    end_stamps = labels
    if self.closed == "left":
        end_stamps += self.freq  # <-- Different for left

    # KEY: searchsorted uses the closed parameter directly
    bins = ax.searchsorted(end_stamps, side=self.closed)
    # ...
```

### 4c. Upsampling Adjustment (_adjust_binner_for_upsample)

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py` (lines 2098-2108)

```python
def _adjust_binner_for_upsample(self, binner):
    """
    Adjust our binner when upsampling.
    The range of a new index should not be outside specified range
    """
    if self.closed == "right":
        binner = binner[1:]  # Remove first element
    else:
        binner = binner[:-1]  # Remove last element
    return binner
```

## 5. The Core C Library Function: generate_bins_dt64

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/_libs/lib.pyx` (lines 900-958)

This is the **CRITICAL function** that implements the binning logic:

```cython
def generate_bins_dt64(ndarray[int64_t, ndim=1] values,
                       const int64_t[:] binner,
                       object closed="left",
                       bint hasnans=False):
    """
    Int64 (datetime64) version of generic python version in ``groupby.py``.
    """
    cdef:
        Py_ssize_t lenidx, lenbin, i, j, bc
        ndarray[int64_t, ndim=1] bins
        int64_t r_bin, nat_count
        bint right_closed = closed == "right"  # <-- CRITICAL COMPARISON

    # ... setup code ...

    bins = np.empty(lenbin - 1, dtype=np.int64)
    j = 0  # index into values
    bc = 0  # bin count

    # LINEAR SCAN WITH DIFFERENT LOGIC FOR 'left' vs 'right'
    if right_closed:
        # For closed='right': values[j] <= r_bin (INCLUSIVE upper bound)
        for i in range(0, lenbin - 1):
            r_bin = binner[i + 1]
            while j < lenidx and values[j] <= r_bin:
                j += 1
            bins[bc] = j
            bc += 1
    else:
        # For closed='left': values[j] < r_bin (EXCLUSIVE upper bound)
        for i in range(0, lenbin - 1):
            r_bin = binner[i + 1]
            while j < lenidx and values[j] < r_bin:
                j += 1
            bins[bc] = j
            bc += 1

    # ... NaT handling ...
    return bins
```

**CRITICAL DIFFERENCE:**
- `closed='right'`: `while j < lenidx and values[j] <= r_bin:` (includes right boundary)
- `closed='left'`: `while j < lenidx and values[j] < r_bin:` (excludes right boundary)

## 6. Range Edge Calculations

### 6a. Timestamp Range Edges (_get_timestamp_range_edges)

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py` (lines 2845-2915)

```python
def _get_timestamp_range_edges(
    first: Timestamp,
    last: Timestamp,
    freq: BaseOffset,
    unit: TimeUnit,
    closed: Literal["right", "left"] = "left",
    origin: TimeGrouperOrigin = "start_day",
    offset: Timedelta | None = None,
) -> tuple[Timestamp, Timestamp]:
    # For Tick frequencies, delegates to _adjust_dates_anchored
    # For non-Tick frequencies:
    if closed == "left":
        first = Timestamp(freq.rollback(first))
    else:
        first = Timestamp(first - freq)
```

### 6b. Date Anchor Adjustment (_adjust_dates_anchored)

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py` (lines 2992-3073)

```python
def _adjust_dates_anchored(
    first: Timestamp,
    last: Timestamp,
    freq: Tick,
    closed: Literal["right", "left"] = "right",
    origin: TimeGrouperOrigin = "start_day",
    offset: Timedelta | None = None,
    unit: TimeUnit = "ns",
) -> tuple[Timestamp, Timestamp]:
    # ... complex logic for origin handling ...

    if closed == "right":
        # RIGHT SIDE CLOSED LOGIC
        if foffset > 0:
            fresult_int = first._value - foffset  # roll back
        else:
            fresult_int = first._value - freq_value

        if loffset > 0:
            lresult_int = last._value + (freq_value - loffset)  # roll forward
        else:
            lresult_int = last._value
    else:  # closed == 'left'
        # LEFT SIDE CLOSED LOGIC
        if foffset > 0:
            fresult_int = first._value - foffset
        else:
            fresult_int = first._value  # start of the road

        if loffset > 0:
            lresult_int = last._value + (freq_value - loffset)  # roll forward
        else:
            lresult_int = last._value + freq_value
```

## 7. Key Places Where closed='left' vs closed='right' Differ

| Operation | Location | closed='left' | closed='right' |
|-----------|----------|---------------|-----------------|
| **Value inclusion** | `lib.pyx:generate_bins_dt64` | `values[j] < r_bin` | `values[j] <= r_bin` |
| **End calculation** | `resample.py:_get_time_delta_bins` | `end_stamps += self.freq` | `end += self.freq` to binner |
| **Upsampling** | `resample.py:_adjust_binner_for_upsample` | `binner = binner[:-1]` | `binner = binner[1:]` |
| **Label handling** | `resample.py:_get_datetime_bins` | Conditional | Always applies label shift |
| **Anchor adjustment** | `resample.py:_adjust_dates_anchored` | Keep start/extend end | Roll back start/keep end |

## 8. Summary of Implementation Structure

The `closed` parameter flows through the following critical path:

```
NDFrame.resample(closed='left'/'right')
    ↓
TimeGrouper.__init__(closed)
    ↓ validates and stores
Resampler.__init__() stores via TimeGrouper
    ↓
_get_datetime_bins() / _get_time_delta_bins() / _get_period_bins()
    ↓
generate_bins_dt64(closed) [C library]
    ↓
Comparison: `<=` (right) vs `<` (left)
```

## 9. Files to Modify for 'both' Implementation

To add support for `closed='both'`, you would need to modify:

1. **Type Annotations** (multiple files):
   - `pandas/core/generic.py:8903` - Add 'both' to Literal type
   - `pandas/core/resample.py:2386` - Add 'both' to TimeGrouper type
   - Other files with type hints

2. **Validation Logic**:
   - `pandas/core/resample.py:2404` - Update validation check

3. **Default Resolution Logic**:
   - `pandas/core/resample.py:2437-2459` - Add 'both' handling to defaults

4. **C Library Function**:
   - `pandas/_libs/lib.pyx:900-958` - Add 'both' logic to generate_bins_dt64

5. **Edge Calculation Functions**:
   - `pandas/core/resample.py:_adjust_dates_anchored` - Add 'both' case
   - `pandas/core/resample.py:_adjust_bin_edges` - Add 'both' case
   - `pandas/core/resample.py:_get_time_delta_bins` - Add 'both' case

6. **Upsampling Logic**:
   - `pandas/core/resample.py:2098-2108` - Handle 'both' in _adjust_binner_for_upsample

7. **Label Handling**:
   - `pandas/core/resample.py:2613-2618` - Handle label logic for 'both'

8. **Tests**:
   - Add comprehensive tests in `pandas/tests/resample/test_datetime_index.py`

## 10. Expected Behavior for 'both'

Based on the IntervalIndex implementation (which already supports 'both'), the `closed='both'` option should:

- Include **both** the left and right boundaries in each bin
- Potentially create overlapping bins where consecutive values at boundaries belong to both bins
- Or handle boundary values specially (e.g., duplicate them in multiple bins)
