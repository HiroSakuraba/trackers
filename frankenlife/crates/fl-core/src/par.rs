//! Data-parallel helpers for the per-organism loops.
//!
//! Every helper gives the answer the plain serial loop gives, bit for bit: results come back in
//! index order, nothing is reduced in floating point across threads, and the work is split
//! by fixed-size chunks, not by the number of workers. `workers <= 1`, a short loop, or the
//! browser build run the loop serially, in the calling thread.

/// Loops shorter than this are not worth waking threads for.
pub const MIN_PAR: usize = 2048;

/// Whether a loop of `n` items should run on the thread pool.
pub fn threaded(workers: usize, n: usize) -> bool {
    cfg!(not(target_arch = "wasm32")) && workers > 1 && n >= MIN_PAR
}

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// `(0..n).map(f).collect()`.
pub fn map_collect<R: Send, F: Fn(usize) -> R + Sync + Send>(workers: usize, n: usize, f: F) -> Vec<R> {
    #[cfg(not(target_arch = "wasm32"))]
    if threaded(workers, n) {
        return (0..n).into_par_iter().with_min_len(512).map(f).collect();
    }
    let _ = workers;
    (0..n).map(f).collect()
}

/// Sum of `f(i)` over `0..n`, componentwise (integers, so the order cannot matter).
pub fn sum2<F: Fn(usize) -> (u64, u64) + Sync + Send>(workers: usize, n: usize, f: F) -> (u64, u64) {
    #[cfg(not(target_arch = "wasm32"))]
    if threaded(workers, n) {
        return (0..n).into_par_iter().with_min_len(1024).map(f).reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    }
    let _ = workers;
    (0..n).map(f).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
}

/// `for (i, a) in a.iter_mut().enumerate() { f(i, a) }`.
pub fn for_each_mut<A: Send, F: Fn(usize, &mut A) + Sync + Send>(workers: usize, a: &mut [A], f: F) {
    #[cfg(not(target_arch = "wasm32"))]
    if threaded(workers, a.len()) {
        a.par_iter_mut().enumerate().with_min_len(512).for_each(|(i, a)| f(i, a));
        return;
    }
    let _ = workers;
    for (i, a) in a.iter_mut().enumerate() {
        f(i, a);
    }
}

/// Two equally long slices walked together.
pub fn for_each_mut2<A: Send, B: Send, F: Fn(usize, &mut A, &mut B) + Sync + Send>(workers: usize, a: &mut [A], b: &mut [B], f: F) {
    assert_eq!(a.len(), b.len());
    #[cfg(not(target_arch = "wasm32"))]
    if threaded(workers, a.len()) {
        a.par_iter_mut().zip(b.par_iter_mut()).enumerate().with_min_len(512).for_each(|(i, (a, b))| f(i, a, b));
        return;
    }
    let _ = workers;
    for (i, (a, b)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
        f(i, a, b);
    }
}

/// Row `k` of three flat buffers (`sa`, `sb`, `sc` elements per row) for each of `n` rows.
#[allow(clippy::too_many_arguments)]
pub fn for_each_rows3<T: Send, F: Fn(usize, &mut [T], &mut [T], &mut [T]) + Sync + Send>(workers: usize, n: usize, a: &mut [T], sa: usize, b: &mut [T], sb: usize, c: &mut [T], sc: usize, f: F) {
    assert!(a.len() == n * sa && b.len() == n * sb && c.len() == n * sc);
    #[cfg(not(target_arch = "wasm32"))]
    if threaded(workers, n) {
        a.par_chunks_mut(sa.max(1)).zip(b.par_chunks_mut(sb.max(1))).zip(c.par_chunks_mut(sc.max(1))).enumerate().with_min_len(512).for_each(|(k, ((a, b), c))| f(k, a, b, c));
        return;
    }
    let _ = workers;
    for k in 0..n {
        f(k, &mut a[k * sa..(k + 1) * sa], &mut b[k * sb..(k + 1) * sb], &mut c[k * sc..(k + 1) * sc]);
    }
}

/// Three equally long slices walked together.
pub fn for_each_mut3<A: Send, B: Send, C: Send, F: Fn(usize, &mut A, &mut B, &mut C) + Sync + Send>(workers: usize, a: &mut [A], b: &mut [B], c: &mut [C], f: F) {
    assert!(a.len() == b.len() && a.len() == c.len());
    #[cfg(not(target_arch = "wasm32"))]
    if threaded(workers, a.len()) {
        a.par_iter_mut().zip(b.par_iter_mut()).zip(c.par_iter_mut()).enumerate().with_min_len(512).for_each(|(i, ((a, b), c))| f(i, a, b, c));
        return;
    }
    let _ = workers;
    for (i, ((a, b), c)) in a.iter_mut().zip(b.iter_mut()).zip(c.iter_mut()).enumerate() {
        f(i, a, b, c);
    }
}

/// Four equally long slices walked together.
pub fn for_each_mut4<A: Send, B: Send, C: Send, D: Send, F: Fn(usize, &mut A, &mut B, &mut C, &mut D) + Sync + Send>(workers: usize, a: &mut [A], b: &mut [B], c: &mut [C], d: &mut [D], f: F) {
    assert!(a.len() == b.len() && a.len() == c.len() && a.len() == d.len());
    #[cfg(not(target_arch = "wasm32"))]
    if threaded(workers, a.len()) {
        a.par_iter_mut().zip(b.par_iter_mut()).zip(c.par_iter_mut()).zip(d.par_iter_mut()).enumerate().with_min_len(512).for_each(|(i, (((a, b), c), d))| f(i, a, b, c, d));
        return;
    }
    let _ = workers;
    for (i, (((a, b), c), d)) in a.iter_mut().zip(b.iter_mut()).zip(c.iter_mut()).zip(d.iter_mut()).enumerate() {
        f(i, a, b, c, d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_match_the_serial_loop() {
        let n = 10_000;
        let serial: Vec<u64> = (0..n).map(|i| (i as u64).wrapping_mul(0x9e37_79b9)).collect();
        assert_eq!(map_collect(8, n, |i| (i as u64).wrapping_mul(0x9e37_79b9)), serial);
        let mut a = vec![1u64; n];
        let mut b = vec![2u64; n];
        let mut c = vec![3u64; n];
        let mut d = vec![4u64; n];
        for_each_mut4(8, &mut a, &mut b, &mut c, &mut d, |i, a, b, c, d| {
            *a += i as u64;
            *b += *a;
            *c += *b;
            *d += *c;
        });
        for i in 0..n {
            assert_eq!((a[i], b[i], c[i], d[i]), (1 + i as u64, 3 + i as u64, 6 + i as u64, 10 + i as u64));
        }
        for_each_mut(8, &mut a, |i, v| *v = i as u64);
        for_each_mut2(8, &mut a, &mut b, |_, x, y| *y = *x);
        assert_eq!(a, b);
        assert!(!threaded(1, n) && !threaded(8, 100) && (cfg!(target_arch = "wasm32") || threaded(8, n)));
    }
}
