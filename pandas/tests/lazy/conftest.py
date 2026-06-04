"""Shared fixtures and setup for lazy pandas tests."""

import contextlib
import warnings

# Pre-import pyarrow.acero (used by the arrow groupby backend) so the
# protobuf gencode/runtime version mismatch UserWarning emitted by the
# third-party substrait package on first import does not escalate to an
# error inside tests (pandas runs pytest with -W error). Once imported
# here, the module is cached and the warning is not raised again.
with contextlib.suppress(ImportError):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        import pyarrow.acero  # noqa: F401
