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
    Bin { op: String, l: Box<Expr>, r: Box<Expr> },
}

fn eval(expr: &Expr, b: &RecordBatch) -> ArrayRef {
    match expr {
        Expr::Col { name } => {
            let i = b.schema().index_of(name).unwrap();
            b.column(i).clone()
        }
        Expr::Liti { v } => Arc::new(Int64Array::new_scalar(*v).into_inner()),
        Expr::Litf { v } => Arc::new(Float64Array::new_scalar(*v).into_inner()),
        Expr::Bin { op, l, r } => {
            let la = eval(l, b);
            let ra = eval(r, b);
            bin(op, &la, &ra)
        }
    }
}

fn datum(a: &ArrayRef) -> Box<dyn arrow::array::Datum + '_> {
    if a.len() == 1 {
        Box::new(Scalar::new(a))
    } else {
        Box::new(a)
    }
}

fn bin(op: &str, l: &ArrayRef, r: &ArrayRef) -> ArrayRef {
    let ld = datum(l);
    let rd = datum(r);
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
                let a = eval(&ne.expr, &b);
                fields.push(Field::new(&ne.name, a.data_type().clone(), true));
                cols.push(a);
            }
            RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
        }
        Plan::Aggregate { group, aggs, input } => {
            let b = execute(input, tables);
            aggregate(&b, group, aggs)
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
    }
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
        let vals = to_f64(b.column(b.schema().index_of(&a.col).unwrap()));
        let mut sum = vec![0f64; ng];
        let mut cnt = vec![0i64; ng];
        let mut mn = vec![f64::INFINITY; ng];
        let mut mx = vec![f64::NEG_INFINITY; ng];
        for i in 0..n {
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
