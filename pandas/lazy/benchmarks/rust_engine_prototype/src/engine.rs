//! General plan -> Arrow execution engine. A serialized LogicalPlan (JSON) is
//! executed over input Arrow RecordBatches: scan/filter/project/aggregate/
//! sort/limit + an expression interpreter. The point: queries route through
//! this, instead of a hand-written run_qN per query.
use std::collections::HashMap as StdMap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Float64Builder, Int64Array, Int64Builder,
    StringArray, StringBuilder, Scalar,
};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use serde::Deserialize;

type Tables = StdMap<String, RecordBatch>;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Plan {
    Scan { table: String, columns: Vec<String> },
    Filter { pred: Expr, input: Box<Plan> },
    Project { exprs: Vec<NamedExpr>, input: Box<Plan> },
    Aggregate {
        group: Vec<String>,
        aggs: Vec<Agg>,
        input: Box<Plan>,
    },
    Sort { keys: Vec<SortKey>, input: Box<Plan> },
    Limit { n: usize, input: Box<Plan> },
    Join {
        left: Box<Plan>,
        right: Box<Plan>,
        left_key: String,
        right_key: String,
        #[serde(default)]
        how: String,
        #[serde(default)]
        left_keys: Vec<String>,
        #[serde(default)]
        right_keys: Vec<String>,
    },
    Distinct { #[serde(default)] subset: Vec<String>, input: Box<Plan> },
}

#[derive(Deserialize)]
pub struct NamedExpr {
    expr: Expr,
    name: String,
}
#[derive(Deserialize)]
pub struct Agg {
    func: String,
    #[serde(default)]
    col: String, // input column (already projected)
    name: String,
}
#[derive(Deserialize)]
pub struct SortKey {
    col: String,
    #[serde(default)]
    desc: bool,
}

#[derive(Deserialize)]
#[serde(tag = "t", rename_all = "lowercase")]
pub enum Expr {
    Col { name: String },
    Liti { v: i64 },
    Litf { v: f64 },
    LitStr { v: String },
    Bin { op: String, l: Box<Expr>, r: Box<Expr> },
    Unary { op: String, a: Box<Expr> },
    Isin { a: Box<Expr>, #[serde(default)] ints: Vec<i64>, #[serde(default)] strs: Vec<String> },
    Case { cases: Vec<(Expr, Expr)>, otherwise: Box<Expr> },
    Str { op: String, a: Box<Expr>, pat: String },
    Slice { a: Box<Expr>, start: i64, #[serde(default)] stop: Option<i64> },
}

fn eval(expr: &Expr, b: &RecordBatch) -> ArrayRef {
    match expr {
        Expr::Col { name } => {
            let i = b.schema().index_of(name).unwrap();
            b.column(i).clone()
        }
        Expr::Liti { v } => Arc::new(Int64Array::new_scalar(*v).into_inner()),
        Expr::Litf { v } => Arc::new(Float64Array::new_scalar(*v).into_inner()),
        Expr::LitStr { v } => Arc::new(StringArray::from(vec![v.clone()])),
        Expr::Bin { op, l, r } => {
            let la = eval(l, b);
            let ra = eval(r, b);
            bin(op, &la, &ra)
        }
        Expr::Unary { op, a } => {
            let x = eval(a, b);
            match op.as_str() {
                "dt_year" => {
                    let ts = compute::cast(
                        &x,
                        &DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
                    ).unwrap();
                    let y = compute::date_part(&ts, compute::DatePart::Year).unwrap();
                    compute::cast(&y, &DataType::Int64).unwrap()
                }
                "is_null" => Arc::new(arrow::array::BooleanArray::from(
                    (0..x.len()).map(|i| x.is_null(i)).collect::<Vec<bool>>(),
                )),
                "invert" => {
                    let xb = x.as_any().downcast_ref::<BooleanArray>().unwrap();
                    Arc::new(compute::not(xb).unwrap())
                }
                _ => panic!("unary {op}"),
            }
        }
        Expr::Isin { a, ints, strs } => {
            let x = eval(a, b);
            let n = x.len();
            let mut out = Vec::with_capacity(n);
            if !ints.is_empty() {
                let set: hashbrown::HashSet<i64, BuildI64> = ints.iter().copied().collect();
                let xi = x.as_any().downcast_ref::<Int64Array>().unwrap();
                for i in 0..n { out.push(set.contains(&xi.value(i))); }
            } else {
                let set: std::collections::HashSet<&str> = strs.iter().map(|s| s.as_str()).collect();
                let xs = x.as_any().downcast_ref::<StringArray>().unwrap();
                for i in 0..n { out.push(set.contains(xs.value(i))); }
            }
            Arc::new(arrow::array::BooleanArray::from(out))
        }
        Expr::Case { cases, otherwise } => {
            let mut acc = eval(otherwise, b);
            for (cond, val) in cases.iter().rev() {
                let m = eval(cond, b);
                let mb = m.as_any().downcast_ref::<BooleanArray>().unwrap();
                let v = eval(val, b);
                // coerce v and acc to common numeric type if needed
                let (vc, ac) = if v.data_type() != acc.data_type()
                    && is_num(v.data_type()) && is_num(acc.data_type()) {
                    (compute::cast(&v, &DataType::Float64).unwrap(),
                     compute::cast(&acc, &DataType::Float64).unwrap())
                } else { (v.clone(), acc.clone()) };
                // broadcast scalars to full length
                let vf = broadcast(&vc, b.num_rows());
                let af = broadcast(&ac, b.num_rows());
                acc = arrow::compute::kernels::zip::zip(mb, &vf, &af).unwrap();
            }
            acc
        }
        Expr::Str { op, a, pat } => {
            let x = eval(a, b);
            let xs = x.as_any().downcast_ref::<StringArray>().unwrap();
            // contains is a regex (pandas str.contains default regex=True);
            // startswith/endswith are literal.
            let re = if op == "contains" { regex::Regex::new(pat).ok() } else { None };
            let out: Vec<bool> = (0..xs.len()).map(|i| {
                let s = xs.value(i);
                match op.as_str() {
                    "startswith" => s.starts_with(pat.as_str()),
                    "endswith" => s.ends_with(pat.as_str()),
                    "contains" => re.as_ref().map(|r| r.is_match(s)).unwrap_or(false),
                    _ => false,
                }
            }).collect();
            Arc::new(arrow::array::BooleanArray::from(out))
        }
        Expr::Slice { a, start, stop } => {
            let x = eval(a, b);
            let xs = x.as_any().downcast_ref::<StringArray>().unwrap();
            let mut bld = StringBuilder::new();
            for i in 0..xs.len() {
                if xs.is_null(i) { bld.append_null(); continue; }
                let s = xs.value(i);
                let st = (*start).max(0) as usize;
                let en = stop.map(|e| e.max(0) as usize).unwrap_or(s.len()).min(s.len());
                bld.append_value(s.get(st.min(s.len())..en).unwrap_or(""));
            }
            Arc::new(bld.finish())
        }
    }
}

fn broadcast(a: &ArrayRef, n: usize) -> ArrayRef {
    if a.len() == n { a.clone() }
    else {
        // length-1 scalar -> repeat via take
        let idx = Int64Array::from(vec![0i64; n]);
        compute::take(a, &idx, None).unwrap()
    }
}

fn datum(a: &ArrayRef) -> Box<dyn arrow::array::Datum + '_> {
    if a.len() == 1 {
        Box::new(Scalar::new(a))
    } else {
        Box::new(a)
    }
}

fn is_num(d: &DataType) -> bool {
    matches!(d, DataType::Int64 | DataType::Float64)
}

fn bin(op: &str, l: &ArrayRef, r: &ArrayRef) -> ArrayRef {
    // numeric type coercion: int literal vs float column -> both f64
    let (lc, rc): (ArrayRef, ArrayRef) = if l.data_type() != r.data_type()
        && is_num(l.data_type())
        && is_num(r.data_type())
    {
        (
            compute::cast(l, &DataType::Float64).unwrap(),
            compute::cast(r, &DataType::Float64).unwrap(),
        )
    } else {
        (l.clone(), r.clone())
    };
    let ld = datum(&lc);
    let rd = datum(&rc);
    match op {
        "add" => compute::kernels::numeric::add(&*ld, &*rd).unwrap(),
        "sub" => compute::kernels::numeric::sub(&*ld, &*rd).unwrap(),
        "mul" => compute::kernels::numeric::mul(&*ld, &*rd).unwrap(),
        "div" => compute::kernels::numeric::div(&*ld, &*rd).unwrap(),
        "lt" => Arc::new(compute::kernels::cmp::lt(&*ld, &*rd).unwrap()),
        "le" => Arc::new(compute::kernels::cmp::lt_eq(&*ld, &*rd).unwrap()),
        "gt" => Arc::new(compute::kernels::cmp::gt(&*ld, &*rd).unwrap()),
        "ge" => Arc::new(compute::kernels::cmp::gt_eq(&*ld, &*rd).unwrap()),
        "eq" => Arc::new(compute::kernels::cmp::eq(&*ld, &*rd).unwrap()),
        "ne" => Arc::new(compute::kernels::cmp::neq(&*ld, &*rd).unwrap()),
        "and" => {
            let lb = l.as_any().downcast_ref::<BooleanArray>().unwrap();
            let rb = r.as_any().downcast_ref::<BooleanArray>().unwrap();
            Arc::new(compute::kernels::boolean::and(lb, rb).unwrap())
        }
        "or" => {
            let lb = l.as_any().downcast_ref::<BooleanArray>().unwrap();
            let rb = r.as_any().downcast_ref::<BooleanArray>().unwrap();
            Arc::new(compute::kernels::boolean::or(lb, rb).unwrap())
        }
        _ => panic!("unknown op {op}"),
    }
}

pub fn execute(plan: &Plan, tables: &Tables) -> RecordBatch {
    match plan {
        Plan::Scan { table, columns } => {
            let b = &tables[table];
            let idx: Vec<usize> = columns.iter().map(|c| b.schema().index_of(c).unwrap()).collect();
            b.project(&idx).unwrap()
        }
        Plan::Filter { pred, input } => {
            let b = execute(input, tables);
            let mask = eval(pred, &b);
            let mb = mask.as_any().downcast_ref::<BooleanArray>().unwrap();
            compute::filter_record_batch(&b, mb).unwrap()
        }
        Plan::Project { exprs, input } => {
            let b = execute(input, tables);
            let mut cols: Vec<ArrayRef> = Vec::new();
            let mut fields: Vec<Field> = Vec::new();
            for ne in exprs {
                let a = broadcast(&eval(&ne.expr, &b), b.num_rows());
                fields.push(Field::new(&ne.name, a.data_type().clone(), true));
                cols.push(a);
            }
            RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
        }
        Plan::Aggregate { group, aggs, input } => {
            let nuniq = aggs.iter().any(|a| a.func == "n_unique");
            if nuniq {
                let b = execute(input, tables);
                return aggregate(&b, group, aggs);
            }
            if let Some(rb) = fused_join_aggregate(input, group, aggs, tables) {
                rb
            } else if let Some((scan, ops)) = peel(input, tables) {
                fused_aggregate(scan, ops, group, aggs)
            } else {
                let b = execute(input, tables);
                aggregate(&b, group, aggs)
            }
        }
        Plan::Sort { keys, input } => {
            let b = execute(input, tables);
            let cols: Vec<compute::SortColumn> = keys
                .iter()
                .map(|k| compute::SortColumn {
                    values: b.column(b.schema().index_of(&k.col).unwrap()).clone(),
                    options: Some(arrow::compute::SortOptions {
                        descending: k.desc,
                        nulls_first: false,
                    }),
                })
                .collect();
            let idx = compute::lexsort_to_indices(&cols, None).unwrap();
            let arrs: Vec<ArrayRef> = b
                .columns()
                .iter()
                .map(|c| compute::take(c, &idx, None).unwrap())
                .collect();
            RecordBatch::try_new(b.schema(), arrs).unwrap()
        }
        Plan::Limit { n, input } => {
            let b = execute(input, tables);
            let k = (*n).min(b.num_rows());
            b.slice(0, k)
        }
        Plan::Join { left, right, left_key, right_key, how, left_keys, right_keys } => {
            let lb = execute(left, tables);
            let rb = execute(right, tables);
            join_exec(&lb, &rb, left_key, right_key, how, left_keys, right_keys)
        }
        Plan::Distinct { subset, input } => {
            let b = execute(input, tables);
            distinct_exec(&b, subset)
        }
    }
}

// Keep first occurrence of each distinct (subset) key combination.
fn distinct_exec(b: &RecordBatch, subset: &[String]) -> RecordBatch {
    let n = b.num_rows();
    let cols: Vec<ArrayRef> = if subset.is_empty() {
        b.columns().to_vec()
    } else {
        subset.iter().map(|c| b.column(b.schema().index_of(c).unwrap()).clone()).collect()
    };
    let mut seen: hashbrown::HashSet<u64, BuildI64> = hashbrown::HashSet::default();
    let mut keep: Vec<i64> = Vec::new();
    for i in 0..n {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for ka in &cols { h = (h ^ keyhash(ka, i)).wrapping_mul(0x0000_0100_0000_01b3); }
        if seen.insert(h) { keep.push(i as i64); }
    }
    let idx = Int64Array::from(keep);
    let arrs: Vec<ArrayRef> = b.columns().iter().map(|c| compute::take(c, &idx, None).unwrap()).collect();
    RecordBatch::try_new(b.schema(), arrs).unwrap()
}

// Inner equi-join on a single i64 key, exact pd.merge (left-row) order, parallel
// probe. Output = all left columns + all right columns except the join key.
fn composite_key(b: &RecordBatch, keys: &[String]) -> Int64Array {
    let arrs: Vec<ArrayRef> = keys.iter()
        .map(|k| b.column(b.schema().index_of(k).unwrap()).clone()).collect();
    let n = b.num_rows();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for ka in &arrs { h = (h ^ keyhash(ka, i)).wrapping_mul(0x0000_0100_0000_01b3); }
        out.push(h as i64);
    }
    Int64Array::from(out)
}

#[allow(clippy::too_many_arguments)]
fn join_exec(lb: &RecordBatch, rb: &RecordBatch, lkey: &str, rkey: &str, how: &str,
             lkeys: &[String], rkeys: &[String]) -> RecordBatch {
    let (li, ri) = if how == "cross" {
        // cartesian product: each left row x every right row (right is usually a
        // 1-row scalar threshold in TPC-H)
        let (nl, nr) = (lb.num_rows(), rb.num_rows());
        let mut la = Vec::with_capacity(nl * nr);
        let mut ra = Vec::with_capacity(nl * nr);
        for i in 0..nl {
            for j in 0..nr { la.push(i as i64); ra.push(j as i64); }
        }
        (Int64Array::from(la), Int64Array::from(ra))
    } else {
        // multi-key: fold the key columns into one composite i64 hash (the engine
        // accepts 64-bit hashing at these cardinalities, as group-by does).
        let lk_owned: Int64Array = if lkeys.len() >= 2 {
            composite_key(lb, lkeys)
        } else {
            lb.column(lb.schema().index_of(lkey).unwrap())
                .as_any().downcast_ref::<Int64Array>().unwrap().clone()
        };
        let rk_owned: Int64Array = if rkeys.len() >= 2 {
            composite_key(rb, rkeys)
        } else {
            rb.column(rb.schema().index_of(rkey).unwrap())
                .as_any().downcast_ref::<Int64Array>().unwrap().clone()
        };
        if how == "left" { join_indices_left(&lk_owned, &rk_owned) }
        else { join_indices(&lk_owned, &rk_owned) }
    };
    let mut fields: Vec<Field> = Vec::new();
    let mut cols: Vec<ArrayRef> = Vec::new();
    for (i, f) in lb.schema().fields().iter().enumerate() {
        fields.push(f.as_ref().clone());
        cols.push(compute::take(lb.column(i), &li, None).unwrap());
    }
    let lsch = lb.schema();
    let lnames: std::collections::HashSet<&str> =
        lsch.fields().iter().map(|f| f.name().as_str()).collect();
    for (i, f) in rb.schema().fields().iter().enumerate() {
        if lnames.contains(f.name().as_str()) { continue; }
        fields.push(f.as_ref().clone());
        cols.push(compute::take(rb.column(i), &ri, None).unwrap());
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
}

// (left_idx, right_idx) for an inner join, pd.merge order: CSR multimap on the
// right (right-row order), parallel probe of left chunks concatenated in order.
fn join_indices(lk: &Int64Array, rk: &Int64Array) -> (Int64Array, Int64Array) {
    let m = rk.len();
    let n = lk.len();
    let mut first: hashbrown::HashMap<i64, u32, BuildI64> = hashbrown::HashMap::default();
    let mut counts: Vec<i64> = Vec::new();
    let mut group_of: Vec<u32> = Vec::with_capacity(m);
    let mut ng: u32 = 0;
    for j in 0..m {
        let k = rk.value(j);
        match first.get(&k) {
            Some(&g) => { group_of.push(g); counts[g as usize] += 1; }
            None => { first.insert(k, ng); group_of.push(ng); counts.push(1); ng += 1; }
        }
    }
    let mut offs: Vec<i64> = vec![0; ng as usize + 1];
    for g in 0..ng as usize { offs[g + 1] = offs[g] + counts[g]; }
    let mut grows: Vec<i64> = vec![0; m];
    let mut cur: Vec<i64> = offs[..ng as usize].to_vec();
    for (j, &g) in group_of.iter().enumerate() {
        let s = cur[g as usize];
        grows[s as usize] = j as i64;
        cur[g as usize] = s + 1;
    }
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(nthreads);
    let spans: Vec<(usize, usize)> = (0..nthreads)
        .map(|t| (t * chunk, ((t + 1) * chunk).min(n)))
        .filter(|(a, b)| a < b)
        .collect();
    let parts: Vec<(Vec<i64>, Vec<i64>)> = spans.par_iter().map(|&(lo, hi)| {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in lo..hi {
            if let Some(&g) = first.get(&lk.value(i)) {
                for kk in offs[g as usize]..offs[g as usize + 1] {
                    a.push(i as i64);
                    b.push(grows[kk as usize]);
                }
            }
        }
        (a, b)
    }).collect();
    let mut la = Vec::new();
    let mut ra = Vec::new();
    for (a, b) in &parts { la.extend_from_slice(a); ra.extend_from_slice(b); }
    (Int64Array::from(la), Int64Array::from(ra))
}

// Left outer join: every left row appears at least once; unmatched left rows get
// a NULL right index (take -> null right columns). pd.merge multiset semantics.
fn join_indices_left(lk: &Int64Array, rk: &Int64Array) -> (Int64Array, Int64Array) {
    let m = rk.len();
    let n = lk.len();
    let mut first: hashbrown::HashMap<i64, u32, BuildI64> = hashbrown::HashMap::default();
    let mut counts: Vec<i64> = Vec::new();
    let mut group_of: Vec<u32> = Vec::with_capacity(m);
    let mut ng: u32 = 0;
    for j in 0..m {
        let k = rk.value(j);
        match first.get(&k) {
            Some(&g) => { group_of.push(g); counts[g as usize] += 1; }
            None => { first.insert(k, ng); group_of.push(ng); counts.push(1); ng += 1; }
        }
    }
    let mut offs: Vec<i64> = vec![0; ng as usize + 1];
    for g in 0..ng as usize { offs[g + 1] = offs[g] + counts[g]; }
    let mut grows: Vec<i64> = vec![0; m];
    let mut cur: Vec<i64> = offs[..ng as usize].to_vec();
    for (j, &g) in group_of.iter().enumerate() {
        let s = cur[g as usize];
        grows[s as usize] = j as i64;
        cur[g as usize] = s + 1;
    }
    let mut la: Vec<i64> = Vec::with_capacity(n);
    let mut ra: Vec<Option<i64>> = Vec::with_capacity(n);
    for i in 0..n {
        match first.get(&lk.value(i)) {
            Some(&g) => {
                for kk in offs[g as usize]..offs[g as usize + 1] {
                    la.push(i as i64);
                    ra.push(Some(grows[kk as usize]));
                }
            }
            None => { la.push(i as i64); ra.push(None); }
        }
    }
    (Int64Array::from(la), Int64Array::from(ra))
}

// group-by + aggregates. Keys: int64/utf8. Aggs over a single column: sum/mean/
// count/min/max (numeric f64/i64 cast to f64).
fn aggregate(b: &RecordBatch, group: &[String], aggs: &[Agg]) -> RecordBatch {
    let n = b.num_rows();
    // group id per row via a composite key string (general; specialize later)
    let key_arrs: Vec<ArrayRef> = group
        .iter()
        .map(|g| b.column(b.schema().index_of(g).unwrap()).clone())
        .collect();
    let mut ids: Vec<usize> = vec![0; n];
    let mut first_row: Vec<usize> = Vec::new();
    // Fast group id: hash the key columns into a u64 per row (no per-row alloc);
    // bucket with the fast i64 hasher. (Prototype: relies on 64-bit hash being
    // collision-free at these cardinalities; a production path verifies keys.)
    let mut map: hashbrown::HashMap<u64, usize, BuildI64> =
        hashbrown::HashMap::default();
    for i in 0..n {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for ka in &key_arrs {
            let kh = keyhash(ka, i);
            h = (h ^ kh).wrapping_mul(0x0000_0100_0000_01b3);
        }
        let id = *map.entry(h).or_insert_with(|| {
            first_row.push(i);
            first_row.len() - 1
        });
        ids[i] = id;
    }
    let ng = first_row.len();
    // accumulate aggs
    let mut out_cols: Vec<ArrayRef> = Vec::new();
    let mut fields: Vec<Field> = Vec::new();
    // key columns (gather first_row)
    let fr_idx = Int64Array::from(first_row.iter().map(|&r| r as i64).collect::<Vec<_>>());
    for (g, ka) in group.iter().zip(&key_arrs) {
        let col = compute::take(ka, &fr_idx, None).unwrap();
        fields.push(Field::new(g, col.data_type().clone(), true));
        out_cols.push(col);
    }
    for a in aggs {
        if a.func == "n_unique" {
            let kc = b.column(b.schema().index_of(&a.col).unwrap()).clone();
            let mut sets: Vec<hashbrown::HashSet<u64, BuildI64>> =
                (0..ng).map(|_| hashbrown::HashSet::default()).collect();
            for i in 0..n { if kc.is_null(i) { continue; } sets[ids[i]].insert(keyhash(&kc, i)); }
            let mut bld = Int64Builder::new();
            for st in &sets { bld.append_value(st.len() as i64); }
            fields.push(Field::new(&a.name, DataType::Int64, true));
            out_cols.push(Arc::new(bld.finish()));
            continue;
        }
        let acol = b.column(b.schema().index_of(&a.col).unwrap()).clone();
        let vals = to_f64(&acol);
        let mut sum = vec![0f64; ng];
        let mut cnt = vec![0i64; ng];
        let mut mn = vec![f64::INFINITY; ng];
        let mut mx = vec![f64::NEG_INFINITY; ng];
        for i in 0..n {
            if acol.is_null(i) { continue; }
            let g = ids[i];
            let v = vals[i];
            sum[g] += v;
            cnt[g] += 1;
            if v < mn[g] { mn[g] = v; }
            if v > mx[g] { mx[g] = v; }
        }
        match a.func.as_str() {
            "sum" => {
                let mut bld = Float64Builder::new();
                for g in 0..ng { bld.append_value(sum[g]); }
                fields.push(Field::new(&a.name, DataType::Float64, true));
                out_cols.push(Arc::new(bld.finish()));
            }
            "mean" => {
                let mut bld = Float64Builder::new();
                for g in 0..ng { bld.append_value(sum[g] / cnt[g] as f64); }
                fields.push(Field::new(&a.name, DataType::Float64, true));
                out_cols.push(Arc::new(bld.finish()));
            }
            "count" => {
                let mut bld = Int64Builder::new();
                for g in 0..ng { bld.append_value(cnt[g]); }
                fields.push(Field::new(&a.name, DataType::Int64, true));
                out_cols.push(Arc::new(bld.finish()));
            }
            "min" => {
                let mut bld = Float64Builder::new();
                for g in 0..ng { bld.append_value(mn[g]); }
                fields.push(Field::new(&a.name, DataType::Float64, true));
                out_cols.push(Arc::new(bld.finish()));
            }
            "max" => {
                let mut bld = Float64Builder::new();
                for g in 0..ng { bld.append_value(mx[g]); }
                fields.push(Field::new(&a.name, DataType::Float64, true));
                out_cols.push(Arc::new(bld.finish()));
            }
            _ => panic!("agg {}", a.func),
        }
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), out_cols).unwrap()
}

fn keyhash(a: &ArrayRef, i: usize) -> u64 {
    match a.data_type() {
        DataType::Int64 => a.as_any().downcast_ref::<Int64Array>().unwrap().value(i) as u64,
        DataType::Float64 => a.as_any().downcast_ref::<Float64Array>().unwrap().value(i).to_bits(),
        DataType::Utf8 => {
            let s = a.as_any().downcast_ref::<StringArray>().unwrap().value(i);
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for &b in s.as_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
            h
        }
        _ => panic!("key dtype {:?}", a.data_type()),
    }
}

#[derive(Default, Clone)]
struct HashU64(u64);
impl std::hash::Hasher for HashU64 {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, _: &[u8]) { unreachable!() }
    fn write_u64(&mut self, i: u64) { self.0 = i.wrapping_mul(0x9E37_79B9_7F4A_7C15); }
}
#[derive(Default, Clone)]
struct BuildI64;
impl std::hash::BuildHasher for BuildI64 {
    type Hasher = HashU64;
    fn build_hasher(&self) -> HashU64 { HashU64(0) }
}

#[allow(dead_code)]
fn keyval(a: &ArrayRef, i: usize) -> String {
    match a.data_type() {
        DataType::Int64 => a.as_any().downcast_ref::<Int64Array>().unwrap().value(i).to_string(),
        DataType::Utf8 => a.as_any().downcast_ref::<StringArray>().unwrap().value(i).to_string(),
        DataType::Float64 => a.as_any().downcast_ref::<Float64Array>().unwrap().value(i).to_bits().to_string(),
        _ => panic!("key dtype {:?}", a.data_type()),
    }
}

fn to_f64(a: &ArrayRef) -> Vec<f64> {
    match a.data_type() {
        DataType::Float64 => a.as_any().downcast_ref::<Float64Array>().unwrap().values().to_vec(),
        DataType::Int64 => a
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&x| x as f64)
            .collect(),
        _ => panic!("agg input dtype {:?}", a.data_type()),
    }
}

// keep StringBuilder import used
#[allow(dead_code)]
fn _u(_: StringBuilder) {}


// ---------------------------------------------------------------------------
// Fused / morsel-pipelined aggregate: Aggregate <- (Filter|Project)* <- Scan.
// Each morsel runs the row-wise chain (filter+project) vectorized on a
// cache-resident slice and partial-aggregates into a thread-local table; merge
// at the end. Intermediates never hit DRAM; parallel over morsels (rayon).
// ---------------------------------------------------------------------------
use rayon::prelude::*;

enum RowOp<'a> {
    Filter(&'a Expr),
    Project(&'a [NamedExpr]),
}

// Peel a row-wise chain over a scan; None if the input isn't fuseable.
fn peel<'a>(plan: &'a Plan, tables: &Tables) -> Option<(RecordBatch, Vec<RowOp<'a>>)> {
    let mut ops: Vec<RowOp<'a>> = Vec::new();
    let mut cur = plan;
    loop {
        match cur {
            Plan::Filter { pred, input } => {
                ops.push(RowOp::Filter(pred));
                cur = input;
            }
            Plan::Project { exprs, input } => {
                ops.push(RowOp::Project(exprs));
                cur = input;
            }
            Plan::Scan { table, columns } => {
                let b = &tables[table];
                let idx: Vec<usize> =
                    columns.iter().map(|c| b.schema().index_of(c).unwrap()).collect();
                ops.reverse(); // bottom-up: scan -> filter -> project
                return Some((b.project(&idx).unwrap(), ops));
            }
            _ => return None,
        }
    }
}

fn apply_ops(mut b: RecordBatch, ops: &[RowOp]) -> RecordBatch {
    for op in ops {
        match op {
            RowOp::Filter(pred) => {
                let mask = eval(pred, &b);
                let mb = mask.as_any().downcast_ref::<BooleanArray>().unwrap();
                b = compute::filter_record_batch(&b, mb).unwrap();
            }
            RowOp::Project(exprs) => {
                let mut cols: Vec<ArrayRef> = Vec::new();
                let mut fields: Vec<Field> = Vec::new();
                for ne in *exprs {
                    let a = broadcast(&eval(&ne.expr, &b), b.num_rows());
                    fields.push(Field::new(&ne.name, a.data_type().clone(), true));
                    cols.push(a);
                }
                b = RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap();
            }
        }
    }
    b
}

#[derive(Clone)]
enum KeyVal {
    I(i64),
    F(u64),
    S(String),
}
fn keyval_enum(a: &ArrayRef, i: usize) -> KeyVal {
    match a.data_type() {
        DataType::Int64 => KeyVal::I(a.as_any().downcast_ref::<Int64Array>().unwrap().value(i)),
        DataType::Float64 => {
            KeyVal::F(a.as_any().downcast_ref::<Float64Array>().unwrap().value(i).to_bits())
        }
        DataType::Utf8 => {
            KeyVal::S(a.as_any().downcast_ref::<StringArray>().unwrap().value(i).to_string())
        }
        _ => panic!("key dtype"),
    }
}

#[derive(Clone)]
struct GState {
    keys: Vec<KeyVal>,
    acc: Vec<[f64; 4]>, // per agg: [sum, count, min, max]
}

fn fused_aggregate(scan: RecordBatch, ops: Vec<RowOp>, group: &[String], aggs: &[Agg]) -> RecordBatch {
    let n = scan.num_rows();
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(nthreads);
    let na = aggs.len();
    let partials: Vec<hashbrown::HashMap<u64, GState, BuildI64>> = (0..nthreads)
        .into_par_iter()
        .map(|t| {
            let lo = t * chunk;
            let hi = ((t + 1) * chunk).min(n);
            let mut map: hashbrown::HashMap<u64, GState, BuildI64> =
                hashbrown::HashMap::default();
            // process this thread's range in cache-resident sub-morsels so the
            // per-op intermediates stay in cache (true fusion).
            const MORSEL: usize = 65_536;
            let mut s = lo;
            while s < hi {
                let e_ = (s + MORSEL).min(hi);
                let m = apply_ops(scan.slice(s, e_ - s), &ops);
                let mn = m.num_rows();
                let gk: Vec<ArrayRef> = group
                    .iter()
                    .map(|g| m.column(m.schema().index_of(g).unwrap()).clone())
                    .collect();
                let av: Vec<Vec<f64>> = aggs
                    .iter()
                    .map(|a| to_f64(m.column(m.schema().index_of(&a.col).unwrap())))
                    .collect();
                for i in 0..mn {
                    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                    for ka in &gk {
                        h = (h ^ keyhash(ka, i)).wrapping_mul(0x0000_0100_0000_01b3);
                    }
                    let e = map.entry(h).or_insert_with(|| GState {
                        keys: gk.iter().map(|ka| keyval_enum(ka, i)).collect(),
                        acc: vec![[0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY]; na],
                    });
                    for (j, av_j) in av.iter().enumerate() {
                        let v = av_j[i];
                        let a = &mut e.acc[j];
                        a[0] += v;
                        a[1] += 1.0;
                        if v < a[2] { a[2] = v; }
                        if v > a[3] { a[3] = v; }
                    }
                }
                s = e_;
            }
            map
        })
        .collect();
    finalize_groups(partials, group, aggs)
}


// ---------------------------------------------------------------------------
// Fused join+aggregate: Aggregate <- [Project <-] Join(probe-chain, build).
// Build the hash once on the build side; morsel-probe the probe side and, per
// cache-resident morsel, gather the joined rows, apply the outer project, and
// partial-aggregate -- one fused pass over the probe side (the run_q3 shape).
// ---------------------------------------------------------------------------
struct JoinBuild {
    first: hashbrown::HashMap<i64, u32, BuildI64>,
    offs: Vec<i64>,
    grows: Vec<i64>,
}
impl JoinBuild {
    fn new(rk: &Int64Array) -> Self {
        let m = rk.len();
        let mut first: hashbrown::HashMap<i64, u32, BuildI64> = hashbrown::HashMap::default();
        let mut counts: Vec<i64> = Vec::new();
        let mut group_of: Vec<u32> = Vec::with_capacity(m);
        let mut ng: u32 = 0;
        for j in 0..m {
            let k = rk.value(j);
            match first.get(&k) {
                Some(&g) => { group_of.push(g); counts[g as usize] += 1; }
                None => { first.insert(k, ng); group_of.push(ng); counts.push(1); ng += 1; }
            }
        }
        let mut offs = vec![0i64; ng as usize + 1];
        for g in 0..ng as usize { offs[g + 1] = offs[g] + counts[g]; }
        let mut grows = vec![0i64; m];
        let mut cur: Vec<i64> = offs[..ng as usize].to_vec();
        for (j, &g) in group_of.iter().enumerate() {
            let s = cur[g as usize];
            grows[s as usize] = j as i64;
            cur[g as usize] = s + 1;
        }
        JoinBuild { first, offs, grows }
    }
    fn probe(&self, lk: &Int64Array) -> (Vec<i64>, Vec<i64>) {
        let mut pl = Vec::new();
        let mut bg = Vec::new();
        for i in 0..lk.len() {
            if let Some(&g) = self.first.get(&lk.value(i)) {
                for kk in self.offs[g as usize]..self.offs[g as usize + 1] {
                    pl.push(i as i64);
                    bg.push(self.grows[kk as usize]);
                }
            }
        }
        (pl, bg)
    }
}

#[allow(clippy::type_complexity)]
fn fused_join_aggregate(
    agg_input: &Plan,
    group: &[String],
    aggs: &[Agg],
    tables: &Tables,
) -> Option<RecordBatch> {
    // peel optional outer project then a join
    let (proj, join): (Option<&[NamedExpr]>, &Plan) = match agg_input {
        Plan::Project { exprs, input } => (Some(exprs), input),
        Plan::Join { .. } => (None, agg_input),
        _ => return None,
    };
    let (left, right, lkey, rkey) = match join {
        // only the inner-join fast path is fused here; left/cross fall to naive
        Plan::Join { left, right, left_key, right_key, how, left_keys, .. }
            if (how.is_empty() || how == "inner") && left_keys.len() < 2 =>
        {
            (left, right, left_key, right_key)
        }
        _ => return None,
    };
    let (probe_scan, probe_ops) = peel(left, tables)?;
    let build = execute(right, tables);
    let rkey_idx = build.schema().index_of(rkey).unwrap();
    let jb = JoinBuild::new(build.column(rkey_idx).as_any().downcast_ref::<Int64Array>().unwrap());
    // joined schema = probe fields + build fields (minus rkey)
    let na = aggs.len();
    let n = probe_scan.num_rows();
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(nthreads);
    let proj_owned: Option<Vec<RowOp>> = proj.map(|e| vec![RowOp::Project(e)]);
    let partials: Vec<hashbrown::HashMap<u64, GState, BuildI64>> = (0..nthreads)
        .into_par_iter()
        .map(|t| {
            let mut map: hashbrown::HashMap<u64, GState, BuildI64> = hashbrown::HashMap::default();
            let lo = t * chunk;
            let hi = ((t + 1) * chunk).min(n);
            const MORSEL: usize = 65_536;
            let mut s = lo;
            while s < hi {
                let e_ = (s + MORSEL).min(hi);
                let pm = apply_ops(probe_scan.slice(s, e_ - s), &probe_ops);
                let lk = pm.column(pm.schema().index_of(lkey).unwrap())
                    .as_any().downcast_ref::<Int64Array>().unwrap();
                let (pl, bg) = jb.probe(lk);
                if pl.is_empty() { s = e_; continue; }
                let pidx = Int64Array::from(pl);
                let bidx = Int64Array::from(bg);
                // gather joined morsel
                let mut fields: Vec<Field> = Vec::new();
                let mut cols: Vec<ArrayRef> = Vec::new();
                for (i, f) in pm.schema().fields().iter().enumerate() {
                    fields.push(f.as_ref().clone());
                    cols.push(compute::take(pm.column(i), &pidx, None).unwrap());
                }
                for (i, f) in build.schema().fields().iter().enumerate() {
                    if i == rkey_idx { continue; }
                    fields.push(f.as_ref().clone());
                    cols.push(compute::take(build.column(i), &bidx, None).unwrap());
                }
                let mut jm = RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap();
                if let Some(ref po) = proj_owned { jm = apply_ops(jm, po); }
                // partial-aggregate the joined morsel
                let mn = jm.num_rows();
                let gk: Vec<ArrayRef> = group.iter()
                    .map(|g| jm.column(jm.schema().index_of(g).unwrap()).clone()).collect();
                let av: Vec<Vec<f64>> = aggs.iter()
                    .map(|a| to_f64(jm.column(jm.schema().index_of(&a.col).unwrap()))).collect();
                for i in 0..mn {
                    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                    for ka in &gk { h = (h ^ keyhash(ka, i)).wrapping_mul(0x0000_0100_0000_01b3); }
                    let e = map.entry(h).or_insert_with(|| GState {
                        keys: gk.iter().map(|ka| keyval_enum(ka, i)).collect(),
                        acc: vec![[0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY]; na],
                    });
                    for (j, av_j) in av.iter().enumerate() {
                        let v = av_j[i];
                        let a = &mut e.acc[j];
                        a[0] += v; a[1] += 1.0;
                        if v < a[2] { a[2] = v; }
                        if v > a[3] { a[3] = v; }
                    }
                }
                s = e_;
            }
            map
        })
        .collect();
    Some(finalize_groups(partials, group, aggs))
}


fn finalize_groups(
    partials: Vec<hashbrown::HashMap<u64, GState, BuildI64>>,
    group: &[String],
    aggs: &[Agg],
) -> RecordBatch {
    let na = aggs.len();
    // merge
    let mut merged: hashbrown::HashMap<u64, GState, BuildI64> = hashbrown::HashMap::default();
    for part in partials {
        for (h, g) in part {
            let e = merged.entry(h).or_insert_with(|| GState {
                keys: g.keys.clone(),
                acc: vec![[0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY]; na],
            });
            for j in 0..na {
                e.acc[j][0] += g.acc[j][0];
                e.acc[j][1] += g.acc[j][1];
                if g.acc[j][2] < e.acc[j][2] { e.acc[j][2] = g.acc[j][2]; }
                if g.acc[j][3] > e.acc[j][3] { e.acc[j][3] = g.acc[j][3]; }
            }
        }
    }
    // build output (key columns + agg columns), group order arbitrary
    let groups: Vec<&GState> = merged.values().collect();
    let ng = groups.len();
    let mut fields: Vec<Field> = Vec::new();
    let mut out: Vec<ArrayRef> = Vec::new();
    for (gi, gname) in group.iter().enumerate() {
        // dtype from first group's keyval
        match &groups.first().map(|g| &g.keys[gi]) {
            Some(KeyVal::I(_)) => {
                let mut b = Int64Builder::new();
                for g in &groups { if let KeyVal::I(v) = g.keys[gi] { b.append_value(v); } }
                fields.push(Field::new(gname, DataType::Int64, true));
                out.push(Arc::new(b.finish()));
            }
            Some(KeyVal::S(_)) => {
                let mut b = StringBuilder::new();
                for g in &groups { if let KeyVal::S(ref v) = g.keys[gi] { b.append_value(v); } }
                fields.push(Field::new(gname, DataType::Utf8, true));
                out.push(Arc::new(b.finish()));
            }
            Some(KeyVal::F(_)) => {
                let mut b = Float64Builder::new();
                for g in &groups { if let KeyVal::F(v) = g.keys[gi] { b.append_value(f64::from_bits(v)); } }
                fields.push(Field::new(gname, DataType::Float64, true));
                out.push(Arc::new(b.finish()));
            }
            None => {}
        }
    }
    for (j, a) in aggs.iter().enumerate() {
        match a.func.as_str() {
            "count" => {
                let mut b = Int64Builder::new();
                for g in &groups { b.append_value(g.acc[j][1] as i64); }
                fields.push(Field::new(&a.name, DataType::Int64, true));
                out.push(Arc::new(b.finish()));
            }
            _ => {
                let mut b = Float64Builder::new();
                for g in &groups {
                    let ac = g.acc[j];
                    b.append_value(match a.func.as_str() {
                        "sum" => ac[0],
                        "mean" => ac[0] / ac[1],
                        "min" => ac[2],
                        "max" => ac[3],
                        _ => panic!("agg"),
                    });
                }
                fields.push(Field::new(&a.name, DataType::Float64, true));
                out.push(Arc::new(b.finish()));
            }
        }
    }
    let _ = ng;
    RecordBatch::try_new(Arc::new(Schema::new(fields)), out).unwrap()
}
