use std::sync::Arc;

use anyhow::{Result, bail};

use super::*;
use crate::animator::{ActionSpec, AnimRequest, Animation, Animator, Kind, standard_actions};
use crate::sprite::{Frame, Sprite};
use crate::animator::mock::MockAnimator;

/// A 32 px frame with an opaque box (x0..x1, y0..y1 inclusive).
fn boxed(size: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> Frame {
    let mut px = vec![-1i16; size * size];
    for y in y0..=y1 {
        for x in x0..=x1 {
            px[y * size + x] = 0;
        }
    }
    Sprite::new(size, px, Arc::new(vec![[200, 50, 50]]))
}

/// Clean in-place clips (a 6 px wide body in the middle); `spin_kick` reaches the border if `bad_kick`.
struct Fake {
    actions: Vec<&'static str>,
    bad_kick: bool,
}

impl Animator for Fake {
    fn name(&self) -> &str {
        "fake"
    }
    fn actions(&self, kind: Kind) -> Vec<ActionSpec> {
        standard_actions(kind).into_iter().filter(|a| self.actions.contains(&a.name.as_str())).collect()
    }
    fn animate(&self, sprite: &Sprite, req: &AnimRequest) -> Result<Animation> {
        let Some(spec) = self.actions(req.kind).into_iter().find(|a| a.name == req.action) else { bail!("no {}", req.action) };
        let s = req.size.max(sprite.size);
        let frames = (0..spec.frames)
            .map(|i| {
                if self.bad_kick && req.action == "spin_kick" && i % 2 == 1 {
                    boxed(s, s / 2 - 3, s - 1, s / 4, s - 2)
                } else {
                    boxed(s, s / 2 - 3, s / 2 + 2, s / 4, s - 2)
                }
            })
            .collect();
        Ok(Animation { frames, note: String::new() })
    }
}

const ALL4: [&str; 4] = ["walk", "punch", "spin_kick", "run"];

#[test]
fn needs_and_missing_actions() {
    let script = fight();
    assert_eq!(script.actions(), ALL4.to_vec());
    let needs = script.needs();
    assert!(needs.contains(&("walk", AZ_RIGHT)) && needs.contains(&("run", AZ_RIGHT)));
    let sprite = boxed(32, 13, 18, 8, 30);
    let fake = Fake { actions: vec!["idle", "walk", "punch", "run"], bad_kick: false };
    let why = build(&fake, &sprite, Kind::Human, 32, &script, true).unwrap_err();
    assert_eq!(why.missing, vec!["spin_kick".to_string()]);
    // birds have none of it
    let why = build(&MockAnimator, &sprite, Kind::Bird, 32, &script, false).unwrap_err();
    assert_eq!(why.missing.len(), 4);
}

#[test]
fn quality_check_decides() {
    let script = fight();
    let sprite = boxed(32, 13, 18, 8, 30);
    let clean = Fake { actions: ALL4.to_vec(), bad_kick: false };
    let scene = build(&clean, &sprite, Kind::Human, 32, &script, true).expect("clean clips give a scene");
    assert_eq!(scene.qc.len(), 4);
    assert!(scene.qc.iter().all(|q| q.flagged == 0 && q.total > 0));
    // half the spin-kick frames touch the border: 32 of 64 > 30 % -> dropped, as pack.py would
    let bad = Fake { actions: ALL4.to_vec(), bad_kick: true };
    let why = build(&bad, &sprite, Kind::Human, 16, &script, true).unwrap_err();
    assert_eq!(why.failed_qc.len(), 1);
    assert_eq!((why.failed_qc[0].action.as_str(), why.failed_qc[0].flagged, why.failed_qc[0].total), ("spin_kick", 32, 64));
    assert!(why.summary().contains("spin_kick"));
    // with the check off the same clips play
    assert!(build(&bad, &sprite, Kind::Human, 16, &script, false).is_ok());
}

#[test]
fn qc_flags_match_pack_py() {
    let f = boxed(32, 10, 20, 5, 28);
    let a = f.opaque_count();
    assert_eq!(qc_frame(&f, a), QcFlags::default());
    assert!(qc_frame(&boxed(32, 0, 20, 5, 28), a).clipped);
    assert!(qc_frame(&f, a * 3).area); // 1/3 < 0.4
    assert!(!qc_frame(&f, a * 2).area); // 0.5
    let mut two = f.clone();
    for y in 2..4 {
        for x in 2..10 {
            two.px[y * 32 + x] = 0; // 16 px apart from 253: 6.3 % -> broken
        }
    }
    two.px[4 * 32 + 9] = -1;
    assert!(qc_frame(&two, a).broken);
    let mut speck = f.clone();
    speck.px[2 * 32 + 2] = 0; // a lone pixel is < 5 %
    assert!(!qc_frame(&speck, a).broken);
}

#[test]
fn fight_timeline() {
    for size in [16usize, 24, 32] {
        let script = fight();
        let sprite = boxed(size, 0, 1, 0, 1);
        let scene = build(&Fake { actions: ALL4.to_vec(), bad_kick: false }, &sprite, Kind::Human, size, &script, true).unwrap();
        let w = 4 * size as i32;
        assert_eq!(scene.stage_w, w as usize);
        let action = |i: usize| scene.shots[i].sprite.map(|(c, _)| scene.clips[c].action);
        let az = |i: usize| scene.shots[i].sprite.map(|(c, _)| scene.clips[c].azimuth);
        // the order of the script, with the native frame counts (punch 4, spin_kick 8)
        let mut runs: Vec<(Option<&str>, usize)> = Vec::new();
        for i in 0..scene.shots.len() {
            match runs.last_mut() {
                Some((a, n)) if *a == action(i) => *n += 1,
                _ => runs.push((action(i), 1)),
            }
        }
        let names: Vec<Option<&str>> = runs.iter().map(|r| r.0).collect();
        assert_eq!(names, vec![Some("walk"), Some("punch"), Some("spin_kick"), Some("punch"), Some("run"), None]);
        assert_eq!((runs[1].1, runs[2].1, runs[3].1), (4, 8, 8));
        let centre = (w - size as i32) / 2;
        let walk: Vec<usize> = (0..scene.shots.len()).filter(|&i| action(i) == Some("walk")).collect();
        // enters from the left edge (body first fully hidden, then visible), facing right, ends at the centre
        let body_x1 = size as i32 / 2 + 2;
        assert!(scene.shots[walk[0]].x + body_x1 >= 0 && scene.shots[walk[0]].x + body_x1 < 3);
        assert!(walk.iter().all(|&i| az(i) == Some(AZ_RIGHT)));
        assert_eq!(scene.shots[*walk.last().unwrap()].x, centre);
        assert!(walk.windows(2).all(|p| scene.shots[p[1]].x >= scene.shots[p[0]].x));
        let step = (scene.shots[*walk.last().unwrap()].x - scene.shots[walk[0]].x) as f32 / (walk.len() - 1) as f32;
        assert!((step - WALK_SPEED * size as f32 / 32.0).abs() < 0.2, "walk step {step}");
        // fight in place at the centre, facing right
        for i in 0..scene.shots.len() {
            if matches!(action(i), Some("punch") | Some("spin_kick")) {
                assert_eq!((scene.shots[i].x, az(i)), (centre, Some(AZ_RIGHT)));
            }
        }
        // no turn: run on to the right (same facing) until the body is off the stage
        let run: Vec<usize> = (0..scene.shots.len()).filter(|&i| action(i) == Some("run")).collect();
        assert!(run.iter().all(|&i| az(i) == Some(AZ_RIGHT)));
        assert!(run.windows(2).all(|p| scene.shots[p[1]].x > scene.shots[p[0]].x));
        let last = &scene.shots[*run.last().unwrap()];
        assert!(last.x + (size as i32 / 2 - 3) >= w, "ends off the stage");
        assert!(run.iter().all(|&i| (scene.shots[i].ticks - RUN_TEMPO).abs() < 1e-6));
        // length: walk + 4 + 8 + 8 ticks + run frames * 70/110 + the pause
        let total = walk.len() as f32 + 20.0 + run.len() as f32 * RUN_TEMPO + 5.0;
        assert!((scene.total_ticks() - total).abs() < 1e-3);
        // playback lookup and GIF delays
        assert_eq!(scene.shot_at(0.0), 0);
        assert_eq!(scene.shot_at(scene.start_of(walk.len()) as f64 + 0.01), walk.len());
        assert_eq!(scene.shot_at(scene.total_ticks() as f64 + 0.5), 0);
        assert_eq!(scene.shot_at(-0.01), scene.shots.len() - 1);
        let d = scene.delays_cs(8.0);
        assert_eq!(d.len(), scene.shots.len());
        let sum: u32 = d.iter().map(|&x| x as u32).sum();
        assert!((sum as f32 - scene.total_ticks() / 8.0 * 100.0).abs() <= 1.0);
        assert!(d.iter().all(|&x| x >= 2));
        // pixels land where the shot says
        let rgba = scene.rgba(walk[walk.len() - 1]);
        let y = size / 2;
        let at = |x: i32| rgba[(y * scene.stage_w + x as usize) * 4 + 3];
        assert_eq!((at(centre + size as i32 / 2 - 4), at(centre + size as i32 / 2 - 3)), (0, 255));
    }
}

#[test]
fn mock_scene_without_qc() {
    let sprite = boxed(16, 4, 11, 1, 15);
    let scene = build(&MockAnimator, &sprite, Kind::Human, 16, &fight(), false).expect("mock offers every human action");
    assert!(scene.qc.is_empty());
    assert!(scene.total_ticks() > 20.0);
}
