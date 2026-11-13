#!/usr/bin/env bash
# Run every compare_pandas.py arm in its own process, twice, in reversed order
# on the second pass to expose drift. Both Pythons should share NumPy and
# PyArrow versions; the script records them with every result.
#
#   RELEASE_PYTHON  Python with a released pandas (the eager reference)
#   BRANCH_PYTHON   Python whose pandas is the deferred-take build; it also
#                   runs the prototype, so library versions match
#   OUTPUT          JSON-lines file to append to
#   Extra arguments are passed to compare_pandas.py (e.g. --strings 32).
set -euo pipefail

: "${RELEASE_PYTHON:?}" "${BRANCH_PYTHON:?}" "${OUTPUT:?}"
here="$(cd "$(dirname "$0")" && pwd)"
script="$here/compare_pandas.py"
src="$here/../src"

arms=(
    pandas pandas-eager pandas-unshared pandas-deferred
    proto-eager proto-compact proto-tree
    proto-eager-numpy proto-compact-numpy proto-tree-numpy
)
# ARMS="a b" runs a subset, e.g. without a deferred-take build.
[[ -n "${ARMS:-}" ]] && read -r -a arms <<<"$ARMS"
run_arm() {
    local arm="$1"
    shift
    local python="$BRANCH_PYTHON"
    [[ "$arm" == pandas ]] && python="$RELEASE_PYTHON"
    # Run outside any source tree (-P, neutral cwd) so the checkout's pandas
    # directory cannot shadow the installed or editable pandas.
    (cd /tmp && PYTHONPATH="$src" "$python" -P "$script" --arm "$arm" \
        --output "$OUTPUT" "$@")
}

for arm in "${arms[@]}"; do run_arm "$arm" "$@"; done
for ((i = ${#arms[@]} - 1; i >= 0; i--)); do run_arm "${arms[i]}" "$@"; done
