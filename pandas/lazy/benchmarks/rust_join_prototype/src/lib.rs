use hashbrown::HashMap;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Fused parallel inner join + column-major gather, unique build (right) side.
/// right_payload is (P, m) C-contiguous. Real threads (rayon), GIL released,
/// no index round-trip, direct write to the final output buffers (no concat).
#[pyfunction]
fn fused_join_gather<'py>(
    py: Python<'py>,
    left_keys: PyReadonlyArray1<i64>,
    right_keys: PyReadonlyArray1<i64>,
    right_payload: PyReadonlyArray2<f64>,
    preserve_order: bool,
) -> PyResult<(Vec<Bound<'py, PyArray1<f64>>>, Bound<'py, PyArray1<i64>>)> {
    let lk = left_keys.as_slice()?;
    let rk = right_keys.as_slice()?;
    let rpay = right_payload.as_slice()?;
    let m = rk.len();
    let p = if m > 0 { rpay.len() / m } else { 0 };
    let n = lk.len();

    let (out_cols, out_lrow): (Vec<Vec<f64>>, Vec<i64>) = py.allow_threads(|| {
        let mut table: HashMap<i64, i64> = HashMap::with_capacity(m);
        for (j, &k) in rk.iter().enumerate() {
            table.insert(k, j as i64);
        }
        let nthreads = rayon::current_num_threads().max(1);
        let chunk = n.div_ceil(nthreads);
        let spans: Vec<(usize, usize)> = (0..nthreads)
            .map(|t| (t * chunk, ((t + 1) * chunk).min(n)))
            .filter(|(lo, hi)| lo < hi)
            .collect();

        // Phase 1: parallel probe, collect indices only (small, cache-resident).
        let idx: Vec<(Vec<i64>, Vec<u32>)> = spans
            .par_iter()
            .map(|&(lo, hi)| {
                let mut lrows: Vec<i64> = Vec::with_capacity(hi - lo);
                let mut rrows: Vec<u32> = Vec::with_capacity(hi - lo);
                for i in lo..hi {
                    if let Some(&rr) = table.get(&lk[i]) {
                        lrows.push(i as i64);
                        rrows.push(rr as u32);
                    }
                }
                (lrows, rrows)
            })
            .collect();

        // Exclusive prefix sum of per-span match counts.
        let counts: Vec<usize> = idx.iter().map(|x| x.0.len()).collect();
        let total: usize = counts.iter().sum();
        let mut offs = vec![0usize; idx.len() + 1];
        for i in 0..idx.len() {
            offs[i + 1] = offs[i] + counts[i];
        }

        // Phase 2: write directly to the final buffers (no concat double-copy).
        let mut out_lrow = vec![0i64; total];
        let mut out_cols: Vec<Vec<f64>> = (0..p).map(|_| vec![0f64; total]).collect();
        // left rows
        // gather per column, each span writing its disjoint global range
        let lrow_ptr = out_lrow.as_mut_ptr() as usize;
        let col_ptrs: Vec<usize> = out_cols.iter_mut().map(|c| c.as_mut_ptr() as usize).collect();
        idx.par_iter().enumerate().for_each(|(s, (lrows, rrows))| {
            let base = offs[s];
            unsafe {
                let lp = (lrow_ptr as *mut i64).add(base);
                std::ptr::copy_nonoverlapping(lrows.as_ptr(), lp, lrows.len());
                for c in 0..p {
                    let cbase = c * m;
                    let dst = (col_ptrs[c] as *mut f64).add(base);
                    for (k, &r) in rrows.iter().enumerate() {
                        *dst.add(k) = rpay[cbase + r as usize];
                    }
                }
            }
        });
        (out_cols, out_lrow)
    });

    let (out_cols, out_lrow) = if preserve_order {
        let mut order: Vec<usize> = (0..out_lrow.len()).collect();
        order.par_sort_unstable_by_key(|&i| out_lrow[i]);
        let ol: Vec<i64> = order.iter().map(|&i| out_lrow[i]).collect();
        let oc: Vec<Vec<f64>> = out_cols
            .par_iter()
            .map(|col| order.iter().map(|&i| col[i]).collect())
            .collect();
        (oc, ol)
    } else {
        (out_cols, out_lrow)
    };

    let cols_py: Vec<_> = out_cols.into_iter().map(|v| v.into_pyarray_bound(py)).collect();
    Ok((cols_py, out_lrow.into_pyarray_bound(py)))
}

#[pymodule]
fn lazyjoin_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fused_join_gather, m)?)?;
    m.add_function(wrap_pyfunction!(join_indices_i64, m)?)?;
    Ok(())
}


/// General inner-join indexers on a single i64 key (non-unique build OK), in
/// exact pd.merge order: build a CSR multimap on the RIGHT (right-row order
/// preserved), probe the LEFT in row order in parallel chunks, concat in chunk
/// order (= left-row order). Returns (left_idx, right_idx). Handles any payload
/// dtype because the caller gathers; this only computes the indices fast.
#[pyfunction]
fn join_indices_i64<'py>(
    py: Python<'py>,
    left_keys: PyReadonlyArray1<i64>,
    right_keys: PyReadonlyArray1<i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let lk = left_keys.as_slice()?;
    let rk = right_keys.as_slice()?;
    let n = lk.len();

    let (left_idx, right_idx): (Vec<i64>, Vec<i64>) = py.allow_threads(|| {
        // CSR multimap on right: key -> contiguous slice of right rows (in
        // right-row order). Two passes: count, then fill.
        let mut firstpos: HashMap<i64, u32> = HashMap::with_capacity(rk.len());
        let mut counts: Vec<i64> = Vec::with_capacity(rk.len());
        let mut group_of: Vec<u32> = Vec::with_capacity(rk.len());
        let mut ngroups: u32 = 0;
        for &k in rk.iter() {
            match firstpos.get(&k) {
                Some(&g) => {
                    group_of.push(g);
                    counts[g as usize] += 1;
                }
                None => {
                    firstpos.insert(k, ngroups);
                    group_of.push(ngroups);
                    counts.push(1);
                    ngroups += 1;
                }
            }
        }
        let mut offsets: Vec<i64> = vec![0; ngroups as usize + 1];
        for g in 0..ngroups as usize {
            offsets[g + 1] = offsets[g] + counts[g];
        }
        let mut group_rows: Vec<i64> = vec![0; rk.len()];
        let mut cursor: Vec<i64> = offsets[..ngroups as usize].to_vec();
        for (j, &g) in group_of.iter().enumerate() {
            let slot = cursor[g as usize];
            group_rows[slot as usize] = j as i64;
            cursor[g as usize] = slot + 1;
        }

        // Parallel probe over contiguous left chunks; concat in chunk order.
        let nthreads = rayon::current_num_threads().max(1);
        let chunk = n.div_ceil(nthreads);
        let spans: Vec<(usize, usize)> = (0..nthreads)
            .map(|t| (t * chunk, ((t + 1) * chunk).min(n)))
            .filter(|(lo, hi)| lo < hi)
            .collect();
        let parts: Vec<(Vec<i64>, Vec<i64>)> = spans
            .par_iter()
            .map(|&(lo, hi)| {
                let mut li: Vec<i64> = Vec::with_capacity(hi - lo);
                let mut ri: Vec<i64> = Vec::with_capacity(hi - lo);
                for i in lo..hi {
                    if let Some(&g) = firstpos.get(&lk[i]) {
                        let s = offsets[g as usize];
                        let e = offsets[g as usize + 1];
                        for k in s..e {
                            li.push(i as i64);
                            ri.push(group_rows[k as usize]);
                        }
                    }
                }
                (li, ri)
            })
            .collect();
        let total: usize = parts.iter().map(|x| x.0.len()).sum();
        let mut left_idx = Vec::with_capacity(total);
        let mut right_idx = Vec::with_capacity(total);
        for (li, ri) in &parts {
            left_idx.extend_from_slice(li);
            right_idx.extend_from_slice(ri);
        }
        (left_idx, right_idx)
    });
    Ok((left_idx.into_pyarray_bound(py), right_idx.into_pyarray_bound(py)))
}
