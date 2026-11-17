use crate::sprite::Frame;

// ------------------------------------------------------------------------------------------------ quality check
/// The per-frame flags of `pack.py::qc` (32 px frames in the dataset).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QcFlags {
    /// The silhouette touches the canvas border.
    pub clipped: bool,
    /// Opaque area outside 0.4 .. 2.5 of the first frame of the same direction.
    pub area: bool,
    /// A second 4-connected piece larger than 5 % of the largest one.
    pub broken: bool,
}

impl QcFlags {
    pub fn any(self) -> bool {
        self.clipped || self.area || self.broken
    }
}

/// `pack.py::qc` for one frame (same float comparisons as numpy, 4-connectivity as `scipy.ndimage.label`).
pub fn qc_frame(f: &Frame, ref_area: usize) -> QcFlags {
    let s = f.size;
    let opaque = |x: usize, y: usize| f.px[y * s + x] >= 0;
    let area = f.opaque_count();
    let clipped = (0..s).any(|i| opaque(i, 0) || opaque(i, s - 1) || opaque(0, i) || opaque(s - 1, i));
    let ratio = if ref_area > 0 { area as f64 / ref_area as f64 } else { 0.0 };
    let mut sizes = components(f);
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let broken = sizes.len() > 1 && sizes[1] as f64 > 0.05 * sizes[0] as f64;
    QcFlags { clipped, area: ratio < 0.4 || ratio > 2.5, broken }
}

/// Sizes of the 4-connected opaque pieces.
fn components(f: &Frame) -> Vec<usize> {
    let s = f.size;
    let mut seen = vec![false; s * s];
    let mut sizes = Vec::new();
    let mut stack = Vec::new();
    for start in 0..s * s {
        if seen[start] || f.px[start] < 0 {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let mut n = 0;
        while let Some(i) = stack.pop() {
            n += 1;
            let (x, y) = (i % s, i / s);
            let mut visit = |j: usize| {
                if !seen[j] && f.px[j] >= 0 {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < s {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - s);
            }
            if y + 1 < s {
                visit(i + s);
            }
        }
        sizes.push(n);
    }
    sizes
}

/// The dataset's verdict on one action: QC flags over all its frames in all directions (each direction's area
/// compared with its own first frame).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QcVerdict {
    pub action: String,
    pub total: usize,
    pub flagged: usize,
    pub clipped: usize,
    pub area: usize,
    pub broken: usize,
}

impl QcVerdict {
    pub fn of(action: &str, directions: &[Vec<Frame>]) -> Self {
        let mut v = QcVerdict { action: action.to_string(), ..Default::default() };
        for frames in directions {
            let Some(first) = frames.first() else { continue };
            let ref_area = first.opaque_count();
            for f in frames {
                let q = qc_frame(f, ref_area);
                v.total += 1;
                v.flagged += q.any() as usize;
                v.clipped += q.clipped as usize;
                v.area += q.area as usize;
                v.broken += q.broken as usize;
            }
        }
        v
    }

    /// `pack.py` drops an action when more than 30 % of its frames are flagged.
    pub fn dropped(&self) -> bool {
        self.total > 0 && self.flagged as f64 > 0.3 * self.total as f64
    }

    /// "spin_kick: 24/64 frames flagged (20 touch the border, 3 area, 1 broken)".
    pub fn describe(&self) -> String {
        let mut why = Vec::new();
        if self.clipped > 0 {
            why.push(format!("{} touch the border", self.clipped));
        }
        if self.area > 0 {
            why.push(format!("{} area jumps", self.area));
        }
        if self.broken > 0 {
            why.push(format!("{} in pieces", self.broken));
        }
        let why = if why.is_empty() { String::new() } else { format!(" ({})", why.join(", ")) };
        format!("{}: {}/{} frames flagged{why}", self.action, self.flagged, self.total)
    }
}
