use arrow::array::{Array, ArrayRef, Float64Array, Float64Builder, Int64Array, Int64Builder, StringArray, StringBuilder};
use arrow::pyarrow::PyArrowType;
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;

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

#[pymodule]
fn lazy_engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(sum_col, m)?)?;
    m.add_function(wrap_pyfunction!(run_q1, m)?)?;
    Ok(())
}


use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use rayon::prelude::*;
use hashbrown::HashMap;

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
