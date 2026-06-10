"""
Cost-based join reordering.

A maximal block of inner equi-joins is commutative and associative, so its
relations may be joined in any order without changing the result (column order
is normalized downstream by name-based references). Picking a good order is
what keeps intermediate results small — the single biggest lever for multi-join
pipelines (a measured ~5x on a star join, enough to overtake Polars, which does
not cost-reorder multi-way joins at all).

Approach (greedy left-deep, System-R cardinality):

1. Flatten a connected block of inner joins into relations + join-graph edges.
2. Greedily build a left-deep tree: start from the smallest relation, then
   repeatedly add the *connected* relation whose join yields the smallest
   estimated output (``|L join R| = |L||R| / max(NDV_L, NDV_R)``).
3. Never introduce a cross product (only connected relations are candidates);
   bail to the original order on any ambiguity.

**Experimental, off by default** (``compute.lazy.join_reorder``). It speeds up
badly-ordered queries, but is not safe to enable unconditionally: per Leis et
al. (VLDB 2015) the estimate quality dominates, and the sampled-NDV model here
is not reliable enough — it can *under*estimate a fact column's NDV and so miss
that joining a filtered dimension early is restrictive, picking an order slower
than a well-written one (a confidence check on input vs greedy cost helps but
cannot save an estimator that is confidently backwards). Correctness is never at
risk — inner joins are commutative/associative and the block is left in its
original order whenever a guard trips — only runtime can regress, which is why
it is opt-in. Closing that needs reliable distinct-count estimation
(HyperLogLog-class) and bushy enumeration (GOO); see ROADMAP.md.

The model is conservative within those limits: floors at 1 row, NDV capped at
the relation's row count, and the least-selective single key per edge.
"""

from __future__ import annotations

from pandas.lazy.optimize.base import PlanVisitor
from pandas.lazy.plan import (
    Join,
    LogicalPlan,
)

# Fallback relation size when a source cannot be estimated.
_DEFAULT_ROWS = 10_000

# Skip reordering (and its cardinality sampling) unless some relation is at
# least this large — below it the joins are cheap and the planning cost of
# estimating NDVs would not pay for itself.
_MIN_REORDER_ROWS = 50_000


class JoinReorder(PlanVisitor):
    """Reorder maximal inner-join blocks by estimated cardinality."""

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        # Per-run schema cache keyed by id() — reset each call so reused object
        # ids from a previous plan can never return stale column sets.
        self._COLS_CACHE = {}
        return self.visit(plan)

    def visit_join(self, plan: Join) -> LogicalPlan:
        if plan.how == "inner":
            block = self._collect_block(plan)
            if block is not None:
                relations, edges = block
                if len(relations) >= 3:
                    rebuilt = self._reorder(relations, edges)
                    if rebuilt is not None:
                        return rebuilt
        # Not a reorderable block (or a guard tripped): default recursion so
        # nested blocks and non-join inputs are still optimized.
        return super().visit_join(plan)

    # -- block extraction ---------------------------------------------------
    def _join_keys(self, node: Join):
        """(left_col, right_col) pairs for a simple equi-join, else None."""
        if node.on is not None:
            return [(c, c) for c in node.on]
        if node.left_on is not None and node.right_on is not None:
            if len(node.left_on) != len(node.right_on):
                return None
            return list(zip(node.left_on, node.right_on, strict=True))
        return None

    def _collect_block(self, node: LogicalPlan):
        """Flatten a connected inner-join block.

        Returns ``(relations, edges)`` where ``relations`` is a list of leaf
        subplans (each already optimized) and ``edges`` is a list of
        ``(left_rel_idx, right_rel_idx, left_col, right_col)``. Returns ``None``
        to signal "bail" (keep the original order).
        """
        if not (isinstance(node, Join) and node.how == "inner"):
            # Leaf relation — optimize its internals (may reorder nested blocks).
            return [self.visit(node)], []

        keys = self._join_keys(node)
        if keys is None:
            return [self.visit(node)], []

        left = self._collect_block(node.left)
        right = self._collect_block(node.right)
        if left is None or right is None:
            return None
        lrels, ledges = left
        rrels, redges = right
        offset = len(lrels)

        new_edges = []
        for lk, rk in keys:
            li = self._find_rel(lrels, lk)
            ri = self._find_rel(rrels, rk)
            if li is None or ri is None:
                return None  # key not attributable to exactly one relation
            new_edges.append((li, offset + ri, lk, rk))

        relations = lrels + rrels
        edges = (
            list(ledges)
            + [(i + offset, j + offset, a, b) for (i, j, a, b) in redges]
            + new_edges
        )
        return relations, edges

    def _find_rel(self, relations, col):
        """Index of the single relation whose schema has ``col``; else None."""
        hits = [i for i, r in enumerate(relations) if col in self._cols(r)]
        return hits[0] if len(hits) == 1 else None

    _COLS_CACHE: dict = {}

    def _cols(self, rel) -> frozenset:
        key = id(rel)
        cached = self._COLS_CACHE.get(key)
        if cached is None:
            try:
                cached = frozenset(rel.resolve_schema().names)
            except Exception:
                cached = frozenset()
            self._COLS_CACHE[key] = cached
        return cached

    # -- cardinality --------------------------------------------------------
    def _rows(self, rel) -> int:
        try:
            n = rel.estimate_row_count()
        except Exception:
            n = None
        return max(1, n if n else _DEFAULT_ROWS)

    def _ndv(self, rel, col, rows: int) -> int:
        """NDV of ``col`` in ``rel``, capped at the relation's row count."""
        try:
            stats = rel.column_statistics(col)
        except Exception:
            stats = None
        ndv = stats.ndv if stats is not None and stats.ndv else None
        if not ndv:
            # Unknown: assume moderately selective but never below 1 row.
            return max(1, min(rows, max(1, rows // 10)))
        return max(1, min(int(ndv), rows))

    def _join_card(self, left_rows, right_rows, key_divisor) -> int:
        return max(1, int(left_rows * right_rows / max(1, key_divisor)))

    def _edges_between(self, joined: set, j: int, edges):
        """Key pairs (left_col_on_joined_side, right_col_on_j) linking j to the
        already-joined set."""
        pairs = []
        for a, b, lk, rk in edges:
            if a in joined and b == j:
                pairs.append((lk, rk))
            elif b in joined and a == j:
                pairs.append((rk, lk))
        return pairs

    # -- greedy reorder -----------------------------------------------------
    def _reorder(self, relations, edges):
        n = len(relations)

        # Guard: every non-key column must belong to exactly one relation, or
        # suffix assignment (_x/_y) could change with order. Bail otherwise.
        key_cols = set()
        for _, _, lk, rk in edges:
            key_cols.add(lk)
            key_cols.add(rk)
        seen = {}
        for r in relations:
            for c in self._cols(r):
                if c in key_cols:
                    continue
                seen[c] = seen.get(c, 0) + 1
                if seen[c] > 1:
                    return None

        rows = [self._rows(r) for r in relations]

        # Gate: only worth the cardinality-sampling cost when a big relation is
        # involved (small joins are fast regardless of order).
        if max(rows) < _MIN_REORDER_ROWS:
            return None

        # Greedy left-deep: start from the smallest relation.
        start = min(range(n), key=lambda i: rows[i])
        joined = {start}
        order = [start]
        cur_rows = rows[start]

        while len(joined) < n:
            best = None  # (j, est, key_pairs)
            for j in range(n):
                if j in joined:
                    continue
                kps = self._edges_between(joined, j, edges)
                if not kps:
                    continue  # not connected -> would be a cross product
                # Conservative divisor: the single most selective key only.
                divisor = 1
                for lk, rk in kps:
                    # NDV of the joined-side key: take it from whichever joined
                    # relation provides it (capped at current intermediate size).
                    lndv = self._key_ndv(relations, joined, lk, cur_rows)
                    rndv = self._ndv(relations[j], rk, rows[j])
                    divisor = max(divisor, lndv, rndv)
                est = self._join_card(cur_rows, rows[j], divisor)
                if best is None or est < best[1]:
                    best = (j, est, kps)
            if best is None:
                return None  # disconnected block; don't synthesize a cross join
            j, est, _ = best
            joined.add(j)
            order.append(j)
            cur_rows = est

        # If greedy reproduced the input order exactly, no point rewriting.
        if order == list(range(n)):
            return None

        # Confidence gate: greedy is left-deep and myopic, so it can pick an
        # order whose *total* estimated cost is actually worse than the input's
        # (it gets forced down a chain that blows up — e.g. a restrictive
        # filtered dimension that wants to stay a small separate subtree, which
        # only bushy plans capture). Evaluate both orders with the same model
        # and only rewrite when greedy is *confidently* cheaper, so estimate
        # noise can't trigger a regression on an already-good hand-written order.
        greedy_cost = self._order_cost(relations, edges, rows, order)
        input_cost = self._order_cost(relations, edges, rows, list(range(n)))
        if greedy_cost >= input_cost * 0.7:
            return None

        return self._build_left_deep(relations, edges, order)

    def _order_cost(self, relations, edges, rows, order) -> float:
        """Total estimated intermediate size for a left-deep ``order`` (sum of
        each join's output rows). ``inf`` if the order is disconnected."""
        joined = {order[0]}
        cur = rows[order[0]]
        total = 0.0
        for j in order[1:]:
            pairs = self._edges_between(joined, j, edges)
            if not pairs:
                return float("inf")
            divisor = 1
            for lk, rk in pairs:
                lndv = self._key_ndv(relations, joined, lk, cur)
                rndv = self._ndv(relations[j], rk, rows[j])
                divisor = max(divisor, lndv, rndv)
            cur = self._join_card(cur, rows[j], divisor)
            total += cur
            joined.add(j)
        return total

    def _key_ndv(self, relations, joined: set, col, cur_rows: int) -> int:
        for i in joined:
            if col in self._cols(relations[i]):
                return self._ndv(relations[i], col, cur_rows)
        return max(1, cur_rows // 10)

    def _build_left_deep(self, relations, edges, order):
        built = {order[0]}
        tree = relations[order[0]]
        for j in order[1:]:
            pairs = self._edges_between(built, j, edges)
            left_keys = tuple(lk for lk, _ in pairs)
            right_keys = tuple(rk for _, rk in pairs)
            # Preserve `on`-style when every key has the same name on both sides
            # (keeps a single key column, matching the original schema).
            if left_keys == right_keys:
                tree = Join(tree, relations[j], on=left_keys, how="inner")
            else:
                tree = Join(
                    tree,
                    relations[j],
                    left_on=left_keys,
                    right_on=right_keys,
                    how="inner",
                )
            built.add(j)
        return tree
