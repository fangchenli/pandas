# Code Reference: Where closed='left' vs closed='right' is Handled

## File 1: pandas/core/generic.py

### Location: Lines 8900-8928
```python
def resample(
    self,
    rule,
    closed: Literal["right", "left"] | None = None,  # <-- Type restriction
    label: Literal["right", "left"] | None = None,
    ...
) -> Resampler:
    """
    Parameters
    ----------
    closed : {{'right', 'left'}}, default None
        Which side of bin interval is closed.
        ...
    """
```

**Changes needed for 'both':**
- Change type annotation to: `Literal["right", "left", "both"] | None = None`
- Update docstring to mention 'both' option


## File 2: pandas/core/resample.py

### Location: Lines 2355-2406
```python
class TimeGrouper(Grouper):
    def __init__(
        self,
        obj: Grouper | None = None,
        freq: Frequency = "Min",
        key: str | None = None,
        closed: Literal["left", "right"] | None = None,  # <-- TYPE
        ...
    ) -> None:
        # Check for correctness of the keyword arguments
        if label not in {None, "left", "right"}:
            raise ValueError(f"Unsupported value {label} for `label`")
        if closed not in {None, "left", "right"}:  # <-- VALIDATION
            raise ValueError(f"Unsupported value {closed} for `closed`")
```

**Changes needed for 'both':**
- Type: `Literal["left", "right", "both"] | None = None`
- Validation: `if closed not in {None, "left", "right", "both"}:`


### Location: Lines 2434-2459
```python
end_types = {"ME", "YE", "QE", "BME", "BYE", "BQE", "W"}
rule = freq.rule_code
if rule in end_types or ("-" in rule and rule[: rule.find("-")] in end_types):
    if closed is None:
        closed = "right"
    if label is None:
        label = "right"
else:
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

self.closed = closed
```

**Note:** Default handling doesn't need to change unless 'both' should have a special default.


### Location: Lines 2098-2108 (_adjust_binner_for_upsample)
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

**Changes needed for 'both':**
```python
if self.closed == "right":
    binner = binner[1:]
elif self.closed == "left":
    binner = binner[:-1]
# For 'both', need to decide logic (maybe don't adjust?)
else:  # closed == "both"
    # Need to define behavior
    pass
```


### Location: Lines 2574-2630 (_get_datetime_bins)
```python
def _get_datetime_bins(self, ax: DatetimeIndex):
    # ... setup ...

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

**Changes needed for 'both':**
- C function must handle 'both' case
- Label handling logic needs expansion


### Location: Lines 2632-2666 (_adjust_bin_edges)
```python
def _adjust_bin_edges(
    self, binner: DatetimeIndex, ax_values: npt.NDArray[np.int64]
) -> tuple[DatetimeIndex, npt.NDArray[np.int64]]:
    # ...
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
    # ...
```

**Changes needed for 'both':**
- Add elif/else branch for 'both' case


### Location: Lines 2668-2705 (_get_time_delta_bins)
```python
def _get_time_delta_bins(self, ax: TimedeltaIndex):
    # ... validation ...

    start, end = ax.min(), ax.max()

    if self.closed == "right":
        end += self.freq  # <-- DIFFERENT CALCULATION

    labels = binner = timedelta_range(
        start=start, end=end, freq=self.freq, name=ax.name
    )

    end_stamps = labels
    if self.closed == "left":
        end_stamps += self.freq  # <-- DIFFERENT CALCULATION

    bins = ax.searchsorted(end_stamps, side=self.closed)
    # ...
```

**Changes needed for 'both':**
- Handle 'both' case in end calculation
- Handle 'both' case in end_stamps calculation
- Handle 'both' case in searchsorted (need custom logic)


### Location: Lines 2845-2915 (_get_timestamp_range_edges)
```python
def _get_timestamp_range_edges(
    first: Timestamp,
    last: Timestamp,
    freq: BaseOffset,
    unit: TimeUnit,
    closed: Literal["right", "left"] = "left",  # <-- TYPE
    origin: TimeGrouperOrigin = "start_day",
    offset: Timedelta | None = None,
) -> tuple[Timestamp, Timestamp]:
    # ...
    if isinstance(freq, Tick):
        # ... delegates to _adjust_dates_anchored ...
    else:
        first = first.normalize()
        last = last.normalize()

        if closed == "left":
            first = Timestamp(freq.rollback(first))
        else:
            first = Timestamp(first - freq)

        last = Timestamp(last + freq)

    return first, last
```

**Changes needed for 'both':**
- Update type to include 'both'
- Add branch for 'both' case


### Location: Lines 2992-3073 (_adjust_dates_anchored)
```python
def _adjust_dates_anchored(
    first: Timestamp,
    last: Timestamp,
    freq: Tick,
    closed: Literal["right", "left"] = "right",  # <-- TYPE
    ...
) -> tuple[Timestamp, Timestamp]:
    # ... complex logic ...

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

    # ... more logic ...
    return fresult, lresult
```

**Changes needed for 'both':**
- Update type to include 'both'
- Add branch for 'both' case with custom logic


## File 3: pandas/_libs/lib.pyx

### Location: Lines 900-958
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
        bint right_closed = closed == "right"  # <-- CRITICAL

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

    # LINEAR SCAN WITH DIFFERENT LOGIC
    if right_closed:
        # closed='right': includes right boundary (<=)
        for i in range(0, lenbin - 1):
            r_bin = binner[i + 1]
            while j < lenidx and values[j] <= r_bin:  # <-- <=
                j += 1
            bins[bc] = j
            bc += 1
    else:
        # closed='left': excludes right boundary (<)
        for i in range(0, lenbin - 1):
            r_bin = binner[i + 1]
            while j < lenidx and values[j] < r_bin:  # <-- <
                j += 1
            bins[bc] = j
            bc += 1

    if nat_count > 0:
        # shift bins by the number of NaT
        bins = bins + nat_count
        bins = np.insert(bins, 0, nat_count)

    return bins
```

**Changes needed for 'both':**
- Need to modify the binning logic to handle 'both'
- For 'both', might need to:
  - Return bins that can have duplicate indices for boundary values
  - Or create a different representation
  - Or aggregate results differently

This is the **MOST CRITICAL** function to modify.


## Summary of all locations requiring changes:

1. `pandas/core/generic.py:8903` - Type annotation
2. `pandas/core/resample.py:2386` - Type annotation
3. `pandas/core/resample.py:2404` - Validation
4. `pandas/core/resample.py:2434-2459` - Default handling (optional)
5. `pandas/core/resample.py:2098-2108` - Upsampling logic
6. `pandas/core/resample.py:2574-2630` - Datetime binning logic
7. `pandas/core/resample.py:2632-2666` - Bin edge adjustment
8. `pandas/core/resample.py:2668-2705` - Timedelta binning logic
9. `pandas/core/resample.py:2845-2915` - Timestamp range edges
10. `pandas/core/resample.py:2992-3073` - Date anchor adjustment
11. `pandas/_libs/lib.pyx:900-958` - Core C function (MOST CRITICAL)
12. Tests in `pandas/tests/resample/`
