//! The UI's animation thread for the ONNX animator: a fight scene at 32 px is about a minute of GPU work, so nothing
//! of it runs on the UI thread.
//!
//! Three kinds of request, each with one waiting slot: the single-action preview, the 8-direction sheet and the
//! fight scene. A newer request of a kind replaces the one waiting in its slot; the UI stops a running request
//! with its [`Ticket`]'s flag. The thread takes the preview first, then the sheet, then the scene, so the preview
//! never waits behind a long scene. Results and progress come back as [`JobMsg`]s; the clips stay in the
//! animator's cache, so the scene and the preview never draw the same clip twice.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use super::onnx::{AnimProgress, Ctl, OnnxAnimator, OnnxWith};
use super::scene::{self, Scene, Script, Unavailable};
use super::{AZIMUTHS, AnimRequest, Animation, Animator, Kind};
use crate::cancel::Failure;
use crate::sprite::Sprite;

pub enum Job {
    Preview { sprite: Sprite, req: AnimRequest, refk: Option<usize> },
    Directions { sprite: Sprite, req: AnimRequest, refk: Option<usize> },
    Scene { sprite: Sprite, kind: Kind, size: usize, script: Script, refk: Option<usize> },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum JobKind {
    Preview,
    Directions,
    Scene,
}

impl Job {
    pub fn kind(&self) -> JobKind {
        match self {
            Job::Preview { .. } => JobKind::Preview,
            Job::Directions { .. } => JobKind::Directions,
            Job::Scene { .. } => JobKind::Scene,
        }
    }
}

pub enum JobResult {
    Preview(Result<Animation, Failure>),
    Directions(Result<Vec<Animation>, Failure>),
    Scene(Result<Scene, Failure>),
}

pub enum JobEvent {
    Progress(AnimProgress),
    Done(JobResult),
}

pub struct JobMsg {
    pub id: u64,
    pub event: JobEvent,
}

/// A submitted request: its id (to match messages) and its stop flag.
#[derive(Clone, Debug)]
pub struct Ticket {
    pub id: u64,
    pub cancel: Arc<AtomicBool>,
}

impl Ticket {
    pub fn stop(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

#[derive(Default)]
struct Queue {
    waiting: HashMap<JobKind, (u64, Job, Arc<AtomicBool>)>,
    quit: bool,
}

pub struct AnimJobs {
    pub anim: Arc<OnnxAnimator>,
    shared: Arc<(Mutex<Queue>, Condvar)>,
    pub rx: Receiver<JobMsg>,
    next: u64,
}

impl AnimJobs {
    pub fn spawn(anim: Arc<OnnxAnimator>, wake: Box<dyn Fn() + Send + Sync>) -> Self {
        let shared = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        let (tx, rx) = channel::<JobMsg>();
        let (sh, an) = (shared.clone(), anim.clone());
        let wake: Arc<dyn Fn() + Send + Sync> = Arc::from(wake);
        std::thread::Builder::new()
            .name("animator".into())
            .spawn(move || run(sh, an, tx, wake))
            .expect("spawning the animation thread");
        AnimJobs { anim, shared, rx, next: 1 }
    }

    /// Queues a request (replacing a waiting one of the same kind, which is dropped without a message).
    pub fn submit(&mut self, job: Job) -> Ticket {
        let id = self.next;
        self.next += 1;
        let cancel = Arc::new(AtomicBool::new(false));
        let (lock, cv) = &*self.shared;
        let mut q = lock_q(lock);
        q.waiting.insert(job.kind(), (id, job, cancel.clone()));
        cv.notify_one();
        Ticket { id, cancel }
    }
}

impl Drop for AnimJobs {
    fn drop(&mut self) {
        request_quit(&self.shared);
    }
}

/// The queue holds plain data that stays consistent across a panic, so a poisoned lock is recovered, not propagated.
fn lock_q(m: &Mutex<Queue>) -> MutexGuard<'_, Queue> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Tells the animation thread to finish (never panics: it runs in `Drop`).
fn request_quit(shared: &(Mutex<Queue>, Condvar)) {
    let (lock, cv) = shared;
    lock_q(lock).quit = true;
    cv.notify_one();
}

/// The scene builder's reason, as it crosses the thread boundary.
fn scene_failure(why: Unavailable) -> Failure {
    if why.cancelled { Failure::Cancelled } else { Failure::Failed(why.summary()) }
}

fn run(shared: Arc<(Mutex<Queue>, Condvar)>, anim: Arc<OnnxAnimator>, tx: Sender<JobMsg>, wake: Arc<dyn Fn() + Send + Sync>) {
    // set when the UI's receiver is gone: nobody waits for results any more, so the thread ends after this job
    let gone = Arc::new(AtomicBool::new(false));
    loop {
        let (id, job, cancel) = {
            let (lock, cv) = &*shared;
            let mut q = lock_q(lock);
            loop {
                if q.quit || gone.load(Ordering::Relaxed) {
                    return;
                }
                let pick = [JobKind::Preview, JobKind::Directions, JobKind::Scene].into_iter().find(|k| q.waiting.contains_key(k));
                if let Some(k) = pick {
                    break q.waiting.remove(&k).unwrap();
                }
                q = cv.wait(q).unwrap_or_else(PoisonError::into_inner);
            }
        };
        let send = {
            let (tx, wake, gone) = (tx.clone(), wake.clone(), gone.clone());
            move |event: JobEvent| {
                if tx.send(JobMsg { id, event }).is_err() {
                    gone.store(true, Ordering::Relaxed);
                }
                wake();
            }
        };
        let progress = {
            let send = send.clone();
            move |p: &AnimProgress| send(JobEvent::Progress(p.clone()))
        };
        let progress: &(dyn Fn(&AnimProgress) + Send + Sync) = &progress;
        let refk = match &job {
            Job::Preview { refk, .. } | Job::Directions { refk, .. } | Job::Scene { refk, .. } => *refk,
        };
        let with = OnnxWith { anim: &anim, ctl: Ctl { cancel: Some(&cancel), progress: Some(progress) }, refk };
        let result = match job {
            Job::Preview { sprite, req, .. } => JobResult::Preview(with.animate(&sprite, &req).map_err(Failure::from)),
            Job::Directions { sprite, req, .. } => JobResult::Directions(with.animate_directions(&sprite, &req, &AZIMUTHS).map_err(Failure::from)),
            Job::Scene { sprite, kind, size, script, .. } => JobResult::Scene(scene::build(&with, &sprite, kind, size, &script, false).map_err(scene_failure)),
        };
        send(JobEvent::Done(result));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poisoned() -> Arc<(Mutex<Queue>, Condvar)> {
        let shared = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        let s = shared.clone();
        let _ = std::thread::spawn(move || {
            let _g = s.0.lock().unwrap();
            panic!("poison the queue");
        })
        .join();
        assert!(shared.0.is_poisoned());
        shared
    }

    #[test]
    fn lock_q_recovers_a_poisoned_queue() {
        let shared = poisoned();
        assert!(!lock_q(&shared.0).quit);
    }

    #[test]
    fn quit_request_does_not_panic_on_a_poisoned_queue() {
        let shared = poisoned();
        request_quit(&shared);
        assert!(lock_q(&shared.0).quit);
    }

    #[test]
    fn unavailable_maps_to_failure() {
        let stopped = Unavailable { cancelled: true, errors: vec!["x".into()], ..Default::default() };
        assert_eq!(scene_failure(stopped), Failure::Cancelled);
        let broken = Unavailable { missing: vec!["walk".into()], ..Default::default() };
        assert_eq!(scene_failure(broken), Failure::Failed("no walk".into()));
    }
}
