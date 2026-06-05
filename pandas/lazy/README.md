# pandas.lazy — experimental lazy execution prototype

Opt-in lazy query execution for pandas: logical plans, a rule-based
optimizer, and an array-native physical engine with streaming and
spill-to-disk.

- **Design proposal (start here):** [docs/PROPOSAL.md](docs/PROPOSAL.md)
- **Runnable tour:** `python pandas/lazy/docs/examples.py`
- **Full documentation:** [docs/README.md](docs/README.md)

```python
import pandas as pd
from pandas.lazy import col, scan

result = (
    scan("events/*.parquet")
    .filter(col("value") > 100)
    .group_by("region")
    .agg(col("value").sum().alias("total"))
    .sort("total", descending=True)
    .collect()
)
```

This module is an experimental prototype circulated for design feedback —
see the proposal's open questions before relying on any API.
