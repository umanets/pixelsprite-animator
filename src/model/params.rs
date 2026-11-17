/// Generation settings (defaults = the web demo's).
#[derive(Debug, Clone)]
pub struct GenParams {
    pub vector: Option<Vec<f32>>,
    pub n: usize,
    pub size: usize,
    pub seed: u32,
    pub g_shape: f64,
    pub g_colour: f64,
    pub temperature: f64,
    pub top_p: f64,
    pub filter: bool,
    /// Sprites per chunk when `rows` is 0 (the demo's `chunk`; 0 = 8).
    pub batch: usize,
    /// Network rows per step (the demo's row budget, `rowsFor`): sprites are packed into a chunk while their rows
    /// fit (at least one sprite per chunk). 0 = `batch` sprites of the first sprite's strategy.
    pub rows: usize,
    /// Silhouette (size x size, 1 = inside): the pixels inside are always filled, the rest stay transparent, and
    /// the filter is off (the demo's "draw" / "one shape" modes).
    pub mask: Option<Vec<u8>>,
}

impl Default for GenParams {
    fn default() -> Self {
        GenParams { vector: None, n: 8, size: 16, seed: 0, g_shape: 1.0, g_colour: 2.0, temperature: 1.0, top_p: 1.0, filter: true, batch: 0, rows: 0, mask: None }
    }
}

/// Canvas size S, window offset o and width w (24 = a 24 window in the middle of the 32 canvas).
#[derive(Debug, Clone, Copy)]
pub struct View {
    pub s: usize,
    pub o: usize,
    pub w: usize,
}

pub struct Progress<'a> {
    /// Sprite ids of the batch being drawn.
    pub ids: &'a [usize],
    /// Pixel just drawn (canvas index).
    pub t: usize,
    pub canvases: &'a [Vec<i16>],
    pub view: View,
    pub done: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Default)]
pub struct GenStats {
    pub sampled: usize,
    pub rejected: usize,
    pub rounds: usize,
    /// Network steps and their total time (for the timing line).
    pub steps: usize,
    pub step_secs: f64,
    /// Chunks drawn and the most rows one of them had.
    pub chunks: usize,
    pub max_rows: usize,
}

#[derive(Debug, Clone)]
pub struct GenOutput {
    pub images: Vec<Vec<i16>>,
    pub size: usize,
    pub stats: GenStats,
    pub secs: f64,
}

#[cfg(test)]
mod tests {
    use super::GenParams;

    #[test]
    fn gen_params_default_is_the_demos() {
        let p = GenParams::default();
        assert_eq!((p.n, p.size, p.seed), (8, 16, 0));
        assert_eq!((p.g_shape, p.g_colour, p.temperature, p.top_p), (1.0, 2.0, 1.0, 1.0));
        assert!(p.filter);
        // 0 = "pick the default" (engine batch size / sprite-count chunking), no caption, no silhouette
        assert_eq!((p.batch, p.rows), (0, 0));
        assert!(p.vector.is_none() && p.mask.is_none());
    }
}
