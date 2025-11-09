# Critical Code Sections for Resample closed Parameter

This document shows the EXACT code sections that implement the 'left' vs 'right' behavior.

## 1. THE MOST CRITICAL FUNCTION (THE CORE)

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/_libs/lib.pyx`
**Lines:** 900-958
**Importance:** CRITICAL - This is where the actual binning happens

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
        bint right_closed = closed == "right"  # <-- THIS LINE DECIDES

    nat_count = 0
    if hasnans:
        mask = values == NPY_NAT
        nat_count = np.sum(mask)
        values = values[~mask]

    lenidx = len(values)
    lenbin = len(binner)

    if lenidx <= 0 or lenbin <= 0:
        raise ValueError("Invalid length for values or for binner")

    if values[0] < binner[0]:
        raise ValueError("Values falls before first bin")

    if values[lenidx - 1] > binner[lenbin - 1]:
        raise ValueError("Values falls after last bin")

    bins = np.empty(lenbin - 1, dtype=np.int64)

    j = 0  # index into values
    bc = 0  # bin count

    # ============ THE CRITICAL DIFFERENCE ============
    if right_closed:
        # *** closed='right': INCLUDES right boundary ***
        for i in range(0, lenbin - 1):
            r_bin = binner[i + 1]
            while j < lenidx and values[j] <= r_bin:  # <-- <=
                j += 1
            bins[bc] = j
            bc += 1
    else:
        # *** closed='left': EXCLUDES right boundary ***
        for i in range(0, lenbin - 1):
            r_bin = binner[i + 1]
            while j < lenidx and values[j] < r_bin:  # <-- <
                j += 1
            bins[bc] = j
            bc += 1
    # ==================================================

    if nat_count > 0:
        bins = bins + nat_count
        bins = np.insert(bins, 0, nat_count)

    return bins
```

**KEY INSIGHT:** The ONLY difference is `<=` (includes) vs `<` (excludes).

---

## 2. VALIDATION

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py`
**Lines:** 2404-2405

```python
if closed not in {None, "left", "right"}:
    raise ValueError(f"Unsupported value {closed} for `closed`")
```

**To add 'both':**
```python
if closed not in {None, "left", "right", "both"}:
    raise ValueError(f"Unsupported value {closed} for `closed`")
```

---

## 3. TIMEDELTA INDEX BINNING

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py`
**Lines:** 2668-2705

```python
def _get_time_delta_bins(self, ax: TimedeltaIndex):
    if not isinstance(ax, TimedeltaIndex):
        raise TypeError(...)

    if not isinstance(self.freq, (Tick, Day)):
        raise ValueError(...)

    if not len(ax):
        binner = labels = TimedeltaIndex(data=[], freq=self.freq, name=ax.name)
        return binner, [], labels

    start, end = ax.min(), ax.max()

    # *** DIFFERENCE 1: End calculation differs for left vs right ***
    if self.closed == "right":
        end += self.freq
    # For 'left', end stays the same

    labels = binner = timedelta_range(
        start=start, end=end, freq=self.freq, name=ax.name
    )

    end_stamps = labels
    # *** DIFFERENCE 2: end_stamps calculation differs ***
    if self.closed == "left":
        end_stamps += self.freq
    # For 'right', end_stamps stays the same

    # *** DIFFERENCE 3: searchsorted uses closed parameter ***
    bins = ax.searchsorted(end_stamps, side=self.closed)

    if self.offset:
        labels += self.offset

    return binner, bins, labels
```

**Key differences:**
1. When `closed='right'`, end is extended by freq
2. When `closed='left'`, end_stamps is extended by freq (inverse logic)
3. searchsorted uses `side=self.closed` (passed as string)

---

## 4. DATETIME INDEX BINNING

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py`
**Lines:** 2608-2618

```python
# Call C function with closed parameter
bins = lib.generate_bins_dt64(
    ax_values, bin_edges, self.closed, hasnans=ax.hasnans
)

# *** Label handling differs for left vs right ***
if self.closed == "right":
    labels = binner
    if self.label == "right":
        labels = labels[1:]
elif self.label == "right":
    labels = labels[1:]
```

---

## 5. BIN EDGE ADJUSTMENT

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py`
**Lines:** 2648-2658

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

**Behavior:**
- `closed='right'`: Adjusts edges by adding/subtracting time units
- `closed='left'`: Uses binner.asi8 directly

---

## 6. UPSAMPLING ADJUSTMENT

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py`
**Lines:** 2104-2107

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

**Behavior:**
- `closed='right'`: Removes first element
- `closed='left'`: Removes last element

---

## 7. COMPLEX DATE ADJUSTMENT

**File:** `/Users/fangchenli/Workspace/pandas-fangchenli/pandas/core/resample.py`
**Lines:** 3042-3066

```python
if closed == "right":
    # *** RIGHT: Roll back start, keep end or extend ***
    if foffset > 0:
        fresult_int = first._value - foffset
    else:
        fresult_int = first._value - freq_value

    if loffset > 0:
        lresult_int = last._value + (freq_value - loffset)
    else:
        lresult_int = last._value
else:  # closed == 'left'
    # *** LEFT: Keep start or adjust, extend end ***
    if foffset > 0:
        fresult_int = first._value - foffset
    else:
        fresult_int = first._value  # Keep start

    if loffset > 0:
        lresult_int = last._value + (freq_value - loffset)
    else:
        lresult_int = last._value + freq_value  # Extend end
```

**Key differences:**
- `closed='right'`: May roll back start, keeps end as-is
- `closed='left'`: Keeps start as-is, always extends end

---

## Summary: The Minimum Changes Needed

To add `closed='both'` support, minimally you need to:

### 1. Validation (1 change)
```python
if closed not in {None, "left", "right", "both"}:  # ADD "both"
```

### 2. C Function (1 critical change)
```cython
elif closed == "both":
    # Include both left and right boundaries
    for i in range(0, lenbin - 1):
        r_bin = binner[i + 1]
        while j < lenidx and values[j] <= r_bin:  # Include right
            j += 1
        bins[bc] = j
        bc += 1
```

### 3. Edge Calculations (4 functions with new branches)
- Each `if self.closed == "right": ... else: ...`
- Becomes: `if ... elif ... else (both): ...`

### 4. Type Annotations (7 locations)
```python
closed: Literal["right", "left", "both"] | None = None  # ADD "both"
```

That's it! The complexity comes from deciding what the exact behavior should be for 'both' in each location.
