//! Cooperative cancellation: the user's "Stop" is an error type of its own, not a magic message.
//!
//! Long jobs (the sprite model, the ONNX animator) check a stop flag between pixels and return [`Cancelled`] as an
//! `anyhow` error. Code that crosses a thread boundary turns the error into a [`Failure`], which keeps "the user
//! stopped it" apart from "it broke" without any string matching.

use std::fmt;

/// The user stopped the job; finished work (cached clips, kept cards) is still valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("stopped")
    }
}

impl std::error::Error for Cancelled {}

/// True when `e` (or a cause in its chain) is a [`Cancelled`].
pub fn is_cancelled(e: &anyhow::Error) -> bool {
    e.chain().any(|c| c.downcast_ref::<Cancelled>().is_some())
}

/// How a background job ended badly, as sent to the UI thread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    Cancelled,
    Failed(String),
}

impl From<anyhow::Error> for Failure {
    fn from(e: anyhow::Error) -> Self {
        if is_cancelled(&e) { Failure::Cancelled } else { Failure::Failed(format!("{e:#}")) }
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Failure::Cancelled => f.write_str("stopped"),
            Failure::Failed(why) => f.write_str(why),
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    #[test]
    fn cancelled_survives_context() {
        let e = anyhow::Error::new(Cancelled).context("drawing row 3");
        assert!(is_cancelled(&e));
        assert_eq!(Failure::from(e), Failure::Cancelled);
    }

    #[test]
    fn other_errors_are_failures_even_when_they_mention_stopped() {
        let e = anyhow::anyhow!("the process stopped responding");
        assert!(!is_cancelled(&e));
        assert_eq!(Failure::from(e), Failure::Failed("the process stopped responding".into()));
        let r: anyhow::Result<()> = Err(Cancelled).context("x");
        assert!(r.is_err());
    }
}
