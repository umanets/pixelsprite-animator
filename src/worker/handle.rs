use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use crate::config::Config;
use crate::ep::DllReport;
use super::{Cmd, Msg};
use super::state::State;

pub struct Worker {
    tx: Sender<Cmd>,
    pub rx: Receiver<Msg>,
    pub cancel: Arc<AtomicBool>,
}

impl Worker {
    pub fn spawn(cfg: Config, dll: DllReport, repaint: Box<dyn Fn() + Send>) -> Self {
        let (tx, crx) = channel::<Cmd>();
        let (mtx, rx) = channel::<Msg>();
        let cancel = Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        std::thread::Builder::new()
            .name("models".into())
            .spawn(move || {
                let mut st = State::new(cfg, dll, mtx, repaint, c2);
                for cmd in crx {
                    st.handle(cmd);
                }
            })
            .expect("spawn worker");
        Worker { tx, rx, cancel }
    }

    pub fn send(&self, cmd: Cmd) {
        if matches!(cmd, Cmd::Generate(_)) {
            self.cancel.store(false, Ordering::Relaxed);
        }
        let _ = self.tx.send(cmd);
    }

    pub fn stop(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
