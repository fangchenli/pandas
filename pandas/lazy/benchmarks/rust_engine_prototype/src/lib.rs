use arrow::array::{Array, ArrayRef, Float64Array, Float64Builder, Int64Array, Int64Builder, StringArray, StringBuilder};
use arrow::pyarrow::PyArrowType;
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;

mod engine;

/// Round-trip a RecordBatch pandas->Rust->pandas (proves the Arrow C-data
/// boundary works and measures its cost).
#[pyfunction]
fn roundtrip(batch: PyArrowType<RecordBatch>) -> PyResult<PyArrowType<RecordBatch>> {
    Ok(PyArrowType(batch.0))
}

/// Trivial compute in Rust: sum a float64 column (proves we operate on the
/// Arrow buffers in Rust, GIL released).
#[pyfunction]
fn sum_col(py: Python<'_>, batch: PyArrowType<RecordBatch>, col: usize) -> PyResult<f64> {
    let b = batch.0;
    let arr: ArrayRef = b.column(col).clone();
    let s = py.allow_threads(|| {
        let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        let mut acc = 0.0f64;
        for i in 0..a.len() {
            acc += a.value(i);
        }
        acc
    });
    Ok(s)
}


/// Execute a serialized LogicalPlan (JSON) over named input Arrow tables.
#[pyfunction]
fn execute(
    py: Python<'_>,
    plan_json: String,
    tables: std::collections::HashMap<String, PyArrowType<RecordBatch>>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let plan: engine::Plan = serde_json::from_str(&plan_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let tbls: std::collections::HashMap<String, RecordBatch> =
        tables.into_iter().map(|(k, v)| (k, v.0)).collect();
    let out = py.allow_threads(|| engine::execute(&plan, &tbls));
    Ok(PyArrowType(out))
}

#[pymodule]
fn lazy_engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(sum_col, m)?)?;
    m.add_function(wrap_pyfunction!(run_q1, m)?)?;
    m.add_function(wrap_pyfunction!(run_q3, m)?)?;
    m.add_function(wrap_pyfunction!(execute, m)?)?;
    Ok(())
}


use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use rayon::prelude::*;
use hashbrown::HashMap;

/// Fast multiply-shift hasher for i64 keys (avalanche), like Polars/FxHash —
/// the generic hasher dominates large hash-probe scans.
#[derive(Default, Clone)]
struct I64Hasher(u64);
impl std::hash::Hasher for I64Hasher {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, _: &[u8]) { unreachable!() }
    fn write_i64(&mut self, i: i64) { self.0 = (i as u64).wrapping_mul(0x9E3779B97F4A7C15); }
    fn write_u64(&mut self, i: u64) { self.0 = i.wrapping_mul(0x9E3779B97F4A7C15); }
    fn write_usize(&mut self, i: usize) { self.0 = (i as u64).wrapping_mul(0x9E3779B97F4A7C15); }
}
#[derive(Default, Clone)]
struct I64Build;
impl std::hash::BuildHasher for I64Build {
    type Hasher = I64Hasher;
    fn build_hasher(&self) -> I64Hasher { I64Hasher(0) }
}
type IMap<V> = HashMap<i64, V, I64Build>;
type ISet = hashbrown::HashSet<i64, I64Build>;

/// TPC-H q1 end-to-end in Rust on Arrow: filter l_shipdate<=cutoff, group by
/// (returnflag, linestatus), multi-agg, sorted. Columns in order:
/// 0 shipdate(i64) 1 returnflag(utf8) 2 linestatus(utf8) 3 quantity(f64)
/// 4 extendedprice(f64) 5 discount(f64) 6 tax(f64).
#[pyfunction]
fn run_q1(py: Python<'_>, batch: PyArrowType<RecordBatch>, cutoff: i64) -> PyResult<PyArrowType<RecordBatch>> {
    let b = batch.0;
    let shipdate = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let rflag = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let lstatus = b.column(2).as_any().downcast_ref::<StringArray>().unwrap();
    let qty = b.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
    let price = b.column(4).as_any().downcast_ref::<Float64Array>().unwrap();
    let disc = b.column(5).as_any().downcast_ref::<Float64Array>().unwrap();
    let tax = b.column(6).as_any().downcast_ref::<Float64Array>().unwrap();
    let n = b.num_rows();

    // accumulators per group: [sum_qty, sum_price, sum_disc_price, sum_charge, sum_disc, count]
    type Acc = [f64; 6];
    let merged: HashMap<(u8, u8), Acc> = py.allow_threads(|| {
        let nthreads = rayon::current_num_threads().max(1);
        let chunk = n.div_ceil(nthreads);
        let partials: Vec<HashMap<(u8, u8), Acc>> = (0..nthreads)
            .into_par_iter()
            .map(|t| {
                let lo = t * chunk;
                let hi = ((t + 1) * chunk).min(n);
                let mut local: HashMap<(u8, u8), Acc> = HashMap::new();
                for i in lo..hi {
                    if shipdate.value(i) <= cutoff {
                        let rf = rflag.value(i).as_bytes()[0];
                        let ls = lstatus.value(i).as_bytes()[0];
                        let q = qty.value(i);
                        let p = price.value(i);
                        let d = disc.value(i);
                        let tx = tax.value(i);
                        let dp = p * (1.0 - d);
                        let acc = local.entry((rf, ls)).or_insert([0.0; 6]);
                        acc[0] += q;
                        acc[1] += p;
                        acc[2] += dp;
                        acc[3] += dp * (1.0 + tx);
                        acc[4] += d;
                        acc[5] += 1.0;
                    }
                }
                local
            })
            .collect();
        let mut merged: HashMap<(u8, u8), Acc> = HashMap::new();
        for part in partials {
            for (k, v) in part {
                let acc = merged.entry(k).or_insert([0.0; 6]);
                for j in 0..6 {
                    acc[j] += v[j];
                }
            }
        }
        merged
    });

    // sort groups by (returnflag, linestatus)
    let mut keys: Vec<(u8, u8)> = merged.keys().copied().collect();
    keys.sort_unstable();

    let mut rf_b = StringBuilder::new();
    let mut ls_b = StringBuilder::new();
    let mut sum_qty = Float64Builder::new();
    let mut sum_price = Float64Builder::new();
    let mut sum_dp = Float64Builder::new();
    let mut sum_charge = Float64Builder::new();
    let mut avg_qty = Float64Builder::new();
    let mut avg_price = Float64Builder::new();
    let mut avg_disc = Float64Builder::new();
    let mut cnt = Int64Builder::new();
    for k in &keys {
        let a = merged[k];
        let c = a[5];
        rf_b.append_value(std::str::from_utf8(&[k.0]).unwrap());
        ls_b.append_value(std::str::from_utf8(&[k.1]).unwrap());
        sum_qty.append_value(a[0]);
        sum_price.append_value(a[1]);
        sum_dp.append_value(a[2]);
        sum_charge.append_value(a[3]);
        avg_qty.append_value(a[0] / c);
        avg_price.append_value(a[1] / c);
        avg_disc.append_value(a[4] / c);
        cnt.append_value(c as i64);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_returnflag", DataType::Utf8, false),
        Field::new("l_linestatus", DataType::Utf8, false),
        Field::new("sum_qty", DataType::Float64, false),
        Field::new("sum_base_price", DataType::Float64, false),
        Field::new("sum_disc_price", DataType::Float64, false),
        Field::new("sum_charge", DataType::Float64, false),
        Field::new("avg_qty", DataType::Float64, false),
        Field::new("avg_price", DataType::Float64, false),
        Field::new("avg_disc", DataType::Float64, false),
        Field::new("count_order", DataType::Int64, false),
    ]));
    let out = RecordBatch::try_new(schema, vec![
        Arc::new(rf_b.finish()), Arc::new(ls_b.finish()),
        Arc::new(sum_qty.finish()), Arc::new(sum_price.finish()),
        Arc::new(sum_dp.finish()), Arc::new(sum_charge.finish()),
        Arc::new(avg_qty.finish()), Arc::new(avg_price.finish()),
        Arc::new(avg_disc.finish()), Arc::new(cnt.finish()),
    ]).unwrap();
    Ok(PyArrowType(out))
}


use hashbrown::HashSet;

/// TPC-H q3 end-to-end in Rust: filter customer(mktsegment) -> semijoin orders
/// (custkey + orderdate<cut) -> join lineitem (orderkey + shipdate>cut),
/// group by orderkey, sum revenue, top-10 by (revenue desc, orderdate asc).
/// cust: [c_custkey i64, c_mktsegment utf8]
/// ord:  [o_orderkey i64, o_custkey i64, o_orderdate i64, o_shippriority i64]
/// line: [l_orderkey i64, l_extendedprice f64, l_discount f64, l_shipdate i64]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_q3(
    py: Python<'_>,
    cust: PyArrowType<RecordBatch>,
    ord: PyArrowType<RecordBatch>,
    line: PyArrowType<RecordBatch>,
    cut: i64,
    segment: String,
) -> PyResult<PyArrowType<RecordBatch>> {
    let cust = cust.0;
    let ord = ord.0;
    let line = line.0;
    let out = py.allow_threads(|| {
        // 1. customer: custkeys in segment
        let ck = cust.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let seg = cust.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let mut building: ISet = ISet::default();
        for i in 0..cust.num_rows() {
            if seg.value(i) == segment {
                building.insert(ck.value(i));
            }
        }
        // 2. orders: orderdate<cut & custkey in building -> map orderkey ->(date,prio)
        let ok = ord.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let ocust = ord.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let odate = ord.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        let oprio = ord.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
        let mut orders: IMap<(i64, i64)> = IMap::default();
        for i in 0..ord.num_rows() {
            if odate.value(i) < cut && building.contains(&ocust.value(i)) {
                orders.insert(ok.value(i), (odate.value(i), oprio.value(i)));
            }
        }
        // 3. lineitem: shipdate>cut & orderkey in orders -> sum revenue per orderkey
        let lok = line.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let lprice = line.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let ldisc = line.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let lship = line.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
        let n = line.num_rows();
        let nthreads = rayon::current_num_threads().max(1);
        let chunk = n.div_ceil(nthreads);
        let partials: Vec<IMap<f64>> = (0..nthreads)
            .into_par_iter()
            .map(|t| {
                let lo = t * chunk;
                let hi = ((t + 1) * chunk).min(n);
                let mut local: IMap<f64> = IMap::default();
                for i in lo..hi {
                    if lship.value(i) > cut {
                        let k = lok.value(i);
                        if orders.contains_key(&k) {
                            let rev = lprice.value(i) * (1.0 - ldisc.value(i));
                            *local.entry(k).or_insert(0.0) += rev;
                        }
                    }
                }
                local
            })
            .collect();
        let mut rev: IMap<f64> = IMap::default();
        for part in partials {
            for (k, v) in part {
                *rev.entry(k).or_insert(0.0) += v;
            }
        }
        // 4. assemble + sort (revenue desc, orderdate asc) + top 10
        let mut rows: Vec<(i64, f64, i64, i64)> = rev
            .into_iter()
            .map(|(k, r)| {
                let (d, p) = orders[&k];
                (k, r, d, p)
            })
            .collect();
        rows.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap().then(a.2.cmp(&b.2))
        });
        rows.truncate(10);
        rows
    });

    let mut k_b = Int64Builder::new();
    let mut r_b = Float64Builder::new();
    let mut d_b = Int64Builder::new();
    let mut p_b = Int64Builder::new();
    for (k, r, d, p) in &out {
        k_b.append_value(*k);
        r_b.append_value(*r);
        d_b.append_value(*d);
        p_b.append_value(*p);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_orderkey", DataType::Int64, false),
        Field::new("revenue", DataType::Float64, false),
        Field::new("o_orderdate", DataType::Int64, false),
        Field::new("o_shippriority", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(k_b.finish()),
            Arc::new(r_b.finish()),
            Arc::new(d_b.finish()),
            Arc::new(p_b.finish()),
        ],
    )
    .unwrap();
    Ok(PyArrowType(batch))
}
