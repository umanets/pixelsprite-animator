use super::*;

// ---------------------------------------------------------------------------------------------- the filter
/// Sizes of the 8-connected pieces of the non-transparent pixels, largest first.
pub fn pieces(img: &[i16], s: usize) -> Vec<usize> {
    let mut seen = vec![false; s * s];
    let mut sizes = Vec::new();
    let mut q = Vec::with_capacity(s * s);
    for p0 in 0..s * s {
        if img[p0] < 0 || seen[p0] {
            continue;
        }
        q.clear();
        q.push(p0);
        seen[p0] = true;
        let mut head = 0;
        while head < q.len() {
            let p = q[head];
            head += 1;
            let (y, x) = ((p / s) as i32, (p % s) as i32);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (yy, xx) = (y + dy, x + dx);
                    if yy < 0 || xx < 0 || yy >= s as i32 || xx >= s as i32 {
                        continue;
                    }
                    let pp = yy as usize * s + xx as usize;
                    if img[pp] >= 0 && !seen[pp] {
                        seen[pp] = true;
                        q.push(pp);
                    }
                }
            }
        }
        sizes.push(q.len());
    }
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes
}

/// Draw again: an empty canvas or a detached piece of 1-2 pixels.
pub fn rejected(img: &[i16], s: usize) -> bool {
    let sizes = pieces(img, s);
    sizes.is_empty() || (sizes.len() >= 2 && sizes[1..].iter().min().copied().unwrap_or(99) <= 2)
}

pub(super) fn crop(img: &[i16], v: View) -> Vec<i16> {
    if v.w == v.s {
        return img.to_vec();
    }
    let mut out = vec![-1i16; v.w * v.w];
    for y in 0..v.w {
        for x in 0..v.w {
            out[y * v.w + x] = img[(y + v.o) * v.s + x + v.o];
        }
    }
    out
}

/// Crop of a canvas in progress (for live previews).
pub fn crop_view(img: &[i16], v: View) -> Vec<i16> {
    crop(img, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4x4 canvas from rows of '#' (drawn) and '.' (transparent).
    fn img(rows: [&str; 4]) -> Vec<i16> {
        rows.iter().flat_map(|r| r.chars().map(|c| if c == '#' { 1 } else { -1 })).collect()
    }

    #[test]
    fn pieces_counts_eight_connected_components_largest_first() {
        // the 2x2 block and the pixel at (2,2) touch diagonally, so they are one piece of 5; (3,0) is separate
        let i = img(["##..", "##..", "..#.", "#..."]);
        assert_eq!(pieces(&i, 4), vec![5, 1]);
        assert!(pieces(&img(["....", "....", "....", "...."]), 4).is_empty());
    }

    #[test]
    fn rejected_flags_empty_canvases_and_tiny_detached_pieces() {
        assert!(rejected(&img(["....", "....", "....", "...."]), 4));
        assert!(!rejected(&img(["##..", "##..", "....", "...."]), 4));
        // a one-pixel speck away from the body, and a two-pixel one
        assert!(rejected(&img(["##..", "##..", "....", "...#"]), 4));
        assert!(rejected(&img(["##..", "##..", "....", "..##"]), 4));
        // a detached piece of three pixels is allowed
        assert!(!rejected(&img(["##..", "##..", "....", ".###"]), 4));
    }

    #[test]
    fn crop_takes_the_window_and_leaves_a_full_canvas_alone() {
        let s = 4;
        let full: Vec<i16> = (0..16).collect();
        assert_eq!(crop(&full, View { s, o: 0, w: 4 }), full);
        // a 2x2 window at offset 1: rows 1-2, columns 1-2
        assert_eq!(crop(&full, View { s, o: 1, w: 2 }), vec![5, 6, 9, 10]);
        assert_eq!(crop_view(&full, View { s, o: 1, w: 2 }), vec![5, 6, 9, 10]);
    }
}
