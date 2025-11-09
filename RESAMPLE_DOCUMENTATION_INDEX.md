# Resample closed Parameter Documentation Index

This directory contains comprehensive documentation about the pandas resample function's `closed` parameter and how to implement support for the 'both' option (Issue #1918).

## Documents Created

### 1. QUICK_START_REFERENCE.md (6.1 KB) - START HERE
**Best for:** Quick understanding of the current implementation
- Entry point and validation
- The critical generate_bins_dt64 Cython function
- 7 key code locations
- Behavior differences between 'left' and 'right'
- Step-by-step code tracing
- Testing script

**Read this first if you want:** A quick understanding of how the system works

### 2. RESAMPLE_CLOSED_ANALYSIS.md (12 KB) - COMPREHENSIVE REFERENCE
**Best for:** Deep understanding of the entire system
- Overview of the closed parameter
- Main NDFrame.resample() entry point
- TimeGrouper class definition and validation
- Resampler class attributes and storage
- Core binning logic differences (4 detailed sections)
- The generate_bins_dt64 C library function with full code
- Range edge calculations
- Key places where 'left' vs 'right' differ (table)
- Implementation structure (flow diagram)
- Files to modify for 'both' implementation
- Expected behavior for 'both'

**Read this if you want:** Complete understanding of all aspects

### 3. CODE_REFERENCE.md (9.6 KB) - IMPLEMENTATION ROADMAP
**Best for:** Implementing the 'both' option
- Line-by-line code snippets for each location
- Shows what changes are needed for each file
- Organized by file and line numbers
- Specific code to change highlighted
- Summary checklist of all locations

**Read this if you want:** A detailed implementation guide with specific code changes

### 4. IMPLEMENTATION_GUIDE.md (9.7 KB) - STEP-BY-STEP PLAN
**Best for:** Actually implementing the feature
- Visual diagrams of 'left', 'right', and 'both' behavior
- Core binning logic differences
- Implementation strategy with 7 steps
- Detailed considerations for each step
- Key decision points
- Testing strategy
- References to related code
- Implementation checklist

**Read this if you want:** A structured plan to implement 'both'

## How to Use These Documents

### If you have 15 minutes:
1. Read QUICK_START_REFERENCE.md
2. Understand the 3 critical functions

### If you have 30 minutes:
1. Read QUICK_START_REFERENCE.md
2. Skim RESAMPLE_CLOSED_ANALYSIS.md sections 1-5
3. Review the table in section 7

### If you have 1 hour:
1. Read QUICK_START_REFERENCE.md thoroughly
2. Read all of RESAMPLE_CLOSED_ANALYSIS.md
3. Review CODE_REFERENCE.md for specific locations

### If you're implementing:
1. Start with IMPLEMENTATION_GUIDE.md Step 1-2 (type annotations)
2. Use CODE_REFERENCE.md to find exact lines to change
3. Implement Step 3 (C function) - the most critical part
4. Follow Steps 4-7 in IMPLEMENTATION_GUIDE.md
5. Use the testing checklist to validate

## Key Files in Pandas That Need Changes

### High Priority (core functionality)
- `pandas/_libs/lib.pyx` (lines 900-958)
  - **THE CRITICAL FUNCTION**: generate_bins_dt64
  - This is where the actual binning logic lives
  - Most important change for implementing 'both'

### Medium Priority (type annotations & validation)
- `pandas/core/generic.py` (line 8903)
  - Main entry point for resample()
  - Type annotation needs 'both'

- `pandas/core/resample.py` (lines 2355-2459)
  - TimeGrouper class definition
  - Validation and storage of closed parameter
  - Default handling logic

### Medium Priority (edge calculations)
- `pandas/core/resample.py` (lines 2098-2630)
  - _adjust_binner_for_upsample
  - _get_datetime_bins
  - _adjust_bin_edges

- `pandas/core/resample.py` (lines 2668-2705)
  - _get_time_delta_bins

### Lower Priority (complex calculations)
- `pandas/core/resample.py` (lines 2845-2915)
  - _get_timestamp_range_edges

- `pandas/core/resample.py` (lines 2992-3073)
  - _adjust_dates_anchored

### Testing
- `pandas/tests/resample/test_datetime_index.py`
  - Add comprehensive tests for 'both' option

## The Most Critical Understanding

The key insight is that resampling with `closed='left'` vs `closed='right'` differs in ONE COMPARISON:

```cython
# closed='left'
while j < lenidx and values[j] < r_bin:      # STRICT less-than
    j += 1

# closed='right'
while j < lenidx and values[j] <= r_bin:     # less-than-or-equal
    j += 1
```

This one character difference (< vs <=) determines whether boundary values are included.

For `closed='both'`, you need to include values at BOTH boundaries, which creates overlapping bins.

## Related Pandas Features

- **IntervalIndex** - Already supports 'both' in its `closed` parameter
  - See: `pandas/core/arrays/interval.py` and `pandas/core/indexes/interval.py`
  - This is a reference for how 'both' should work

- **Rolling windows** - Also uses similar closed parameter concepts
  - See: `pandas/core/window/rolling.py`

## Common Questions

### Q: Why do we need 'both'?
A: Use case #1918 describes a need for inclusive binning on both sides.

### Q: Will this break existing code?
A: No, 'both' is a new option. Existing code using 'left' or 'right' won't change.

### Q: How do overlapping bins work?
A: Boundary values appear in multiple bins. Aggregation functions will process them multiple times.

### Q: Should 'both' be a default?
A: No, users should explicitly specify it. Keep current defaults ('left' or 'right').

### Q: Does 'label' parameter work with 'both'?
A: Probably yes, with 'label' being 'left' or 'right' (not 'both').

## File Size Guide

- QUICK_START_REFERENCE.md: 200 lines - read in 10 minutes
- RESAMPLE_CLOSED_ANALYSIS.md: 380 lines - read in 20 minutes
- CODE_REFERENCE.md: 350 lines - read in 15 minutes, reference while coding
- IMPLEMENTATION_GUIDE.md: 370 lines - read in 25 minutes, follow while implementing

**Total time to understand everything: ~70 minutes**

## Next Steps

1. Choose a reference document based on your time and needs
2. Run the test script in QUICK_START_REFERENCE.md to see current behavior
3. Trace through the code using the line numbers provided
4. If implementing, follow IMPLEMENTATION_GUIDE.md step-by-step
5. Use CODE_REFERENCE.md to find exact lines to modify
6. Add tests and verify no regressions

## Document Change History

All documents created on: 2025-11-08

Based on pandas repository at: /Users/fangchenli/Workspace/pandas-fangchenli

Current branch: fix/issue-1918-bar-plot-dateformatter

## Questions or Issues?

Refer back to:
- QUICK_START_REFERENCE.md for basic concepts
- RESAMPLE_CLOSED_ANALYSIS.md for comprehensive details
- CODE_REFERENCE.md for implementation locations
- IMPLEMENTATION_GUIDE.md for implementation strategy
