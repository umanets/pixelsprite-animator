//! Silhouette edge flags and the demo's row budgets.

pub(super) const EDGE_UNKNOWN: usize = 3;

/// The demo's row budget per network step on a desktop with WebGPU (`rowsFor` in app.js): 24 rows at 16x16 (8
/// sprites of a two-caption mix in one pass), 8 at 24 / 32.
pub fn demo_rows_gpu(size: usize) -> usize {
    if size == 16 { 24 } else { 8 }
}

/// The demo's row budget on the CPU (WebAssembly): 2 rows x 8 sprites at 16x16, 2 x 2 at 24 / 32.
pub fn demo_rows_cpu(size: usize) -> usize {
    if size == 16 { 16 } else { 4 }
}

/// Edge flag per canvas pixel: 0 inside, 1 on the boundary (4-neighbourhood, canvas edge = outside), 2 outside,
/// 3 unknown (no silhouette), as `edgeFlags` in engine.js.
pub fn edge_flags(mask: Option<&[u8]>, s: usize) -> Vec<u8> {
    let Some(m) = mask else { return vec![EDGE_UNKNOWN as u8; s * s] };
    let mut e = vec![0u8; s * s];
    for y in 0..s {
        for x in 0..s {
            let p = y * s + x;
            if m[p] == 0 {
                e[p] = 2;
                continue;
            }
            let inner = y > 0 && m[p - s] != 0 && y < s - 1 && m[p + s] != 0 && x > 0 && m[p - 1] != 0 && x < s - 1 && m[p + 1] != 0;
            e[p] = if inner { 0 } else { 1 };
        }
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_budgets_follow_the_demo() {
        assert_eq!((demo_rows_gpu(16), demo_rows_gpu(24), demo_rows_gpu(32)), (24, 8, 8));
        assert_eq!((demo_rows_cpu(16), demo_rows_cpu(24), demo_rows_cpu(32)), (16, 4, 4));
    }

    #[test]
    fn edge_flags_without_a_silhouette_are_unknown() {
        assert_eq!(edge_flags(None, 3), vec![EDGE_UNKNOWN as u8; 9]);
    }

    #[test]
    fn edge_flags_mark_inside_boundary_and_outside() {
        // a 3x3 solid square on a 5x5 canvas: the centre is inside, its ring is the boundary, the rest is outside
        let s = 5;
        let mut mask = vec![0u8; s * s];
        for y in 1..4 {
            for x in 1..4 {
                mask[y * s + x] = 1;
            }
        }
        let e = edge_flags(Some(&mask), s);
        assert_eq!(e[2 * s + 2], 0);
        assert_eq!(e[s + 1], 1);
        assert_eq!(e[3 * s + 3], 1);
        assert_eq!(e[0], 2);
        // a filled pixel on the canvas edge counts as boundary (the canvas edge is outside)
        let full = edge_flags(Some(&vec![1u8; 9]), 3);
        assert_eq!(full[0], 1);
        assert_eq!(full[4], 0);
    }
}
