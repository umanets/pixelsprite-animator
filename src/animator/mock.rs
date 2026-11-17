//! Placeholder animator: procedural motion made from the sprite itself, so loops, directions and exports can be
//! tried before the animation model exists. Every frame is the reference sprite pushed around:
//!
//! * part shifts inside the opaque bounding box (legs = bottom band split into front / back halves, the "front arm"
//!   = middle band on the facing side, wings = outer columns, upper body = top 60 %),
//! * a lean (rows shifted more towards the top, feet fixed),
//! * squash / stretch about the feet, rotation (roll, death), hops,
//! * a colour flash (hurt, cast) mapped back to the nearest palette colour,
//! * direction: the sprite is taken as facing right at 90°; 180°-315° mirror it, front / back views (0° / 180°)
//!   swap the sideways leg swing for alternating leg lifts, the back view is slightly darker.
//!
//! None of this knows anatomy; it only makes the loop visible. The UI labels it "MOCK".

use std::collections::HashMap;
use std::f32::consts::PI;

use anyhow::{Result, bail};

use super::{ActionSpec, AnimRequest, Animation, Animator, Kind, standard_actions};
use crate::sprite::Sprite;

pub struct MockAnimator;

impl Animator for MockAnimator {
    fn name(&self) -> &str {
        "Mock (procedural placeholder)"
    }
    fn is_mock(&self) -> bool {
        true
    }
    fn actions(&self, kind: Kind) -> Vec<ActionSpec> {
        standard_actions(kind)
    }
    fn animate(&self, sprite: &Sprite, req: &AnimRequest) -> Result<Animation> {
        let specs = standard_actions(req.kind);
        let Some(spec) = specs.iter().find(|s| s.name == req.action) else {
            bail!("mock: no action '{}' for {}", req.action, req.kind.label());
        };
        let base = if sprite.size == req.size { sprite.clone() } else { resize_nearest(sprite, req.size) };
        let Some(bb) = base.bbox() else { bail!("the sprite is empty") };
        let n = if req.n_frames == 0 { spec.frames } else { req.n_frames }.clamp(1, 64);
        let az = (req.azimuth % 360) as f32 * PI / 180.0;
        let mirror = (180..360).contains(&(req.azimuth % 360));
        let shade = 0.18 * (-az.cos()).max(0.0);
        let mut nearest = NearestCache::default();
        let frames = (0..n)
            .map(|i| {
                let ph = if spec.looping || n == 1 { i as f32 / n as f32 } else { i as f32 / (n - 1) as f32 };
                let pose = pose(req.kind, &req.action, ph, az, bb);
                let mut f = render(&base, bb, &pose, &mut nearest);
                if shade > 0.0 {
                    tint(&mut f, [0, 0, 0], shade, &mut nearest);
                }
                if mirror { f.mirrored() } else { f }
            })
            .collect();
        Ok(Animation {
            frames,
            note: format!(
                "MOCK motion: {} {} at {}°, {n} frames{}",
                req.kind.label(),
                req.action,
                req.azimuth,
                if mirror { ", mirrored" } else { "" }
            ),
        })
    }
}

/// Parameters of one frame (pixels are already multiplied by the sprite's scale unit).
#[derive(Debug, Clone, Copy)]
struct Pose {
    tx: f32,
    ty: f32,
    sx: f32,
    sy: f32,
    rot: f32,
    pivot_feet: bool,
    lie_down: bool,
    shear: f32,
    torso_dy: f32,
    legs_band: f32,
    legs_side: f32,
    legs_front: f32,
    kick_dx: f32,
    kick_dy: f32,
    arm_dx: f32,
    arm_dy: f32,
    wing_dy: f32,
    flip: bool,
    flash: f32,
    flash_rgb: [u8; 3],
}

impl Default for Pose {
    fn default() -> Self {
        Pose {
            tx: 0.0,
            ty: 0.0,
            sx: 1.0,
            sy: 1.0,
            rot: 0.0,
            pivot_feet: false,
            lie_down: false,
            shear: 0.0,
            torso_dy: 0.0,
            legs_band: 0.32,
            legs_side: 0.0,
            legs_front: 0.0,
            kick_dx: 0.0,
            kick_dy: 0.0,
            arm_dx: 0.0,
            arm_dy: 0.0,
            wing_dy: 0.0,
            flip: false,
            flash: 0.0,
            flash_rgb: [255, 255, 255],
        }
    }
}

/// 0 at |ph - c| >= w, 1 at c.
fn bump(ph: f32, c: f32, w: f32) -> f32 {
    (1.0 - (ph - c).abs() / w).max(0.0)
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn pose(kind: Kind, action: &str, ph: f32, az: f32, bb: (i32, i32, i32, i32)) -> Pose {
    let h = (bb.3 - bb.1 + 1) as f32;
    let u = (h / 14.0).max(1.0); // one "pixel" of motion per 14 px of height
    let side = az.sin().abs();
    let front = az.cos().abs();
    let cyc = 2.0 * PI * ph;
    let (s1, c1) = (cyc.sin(), cyc.cos());
    let breathe = 0.9 * u * (1.0 - c1) / 2.0;
    let mut p = Pose::default();
    match (kind, action) {
        (Kind::Human, "idle") => p.torso_dy = breathe,
        (Kind::Human, "walk") => {
            p.legs_side = side * u * s1;
            p.legs_front = front * u * s1;
            p.torso_dy = 0.9 * u * s1.abs();
            p.arm_dx = -0.8 * side * u * s1;
        }
        (Kind::Human, "run") => {
            p.legs_side = side * 2.0 * u * s1;
            p.legs_front = front * 1.5 * u * s1;
            p.ty = -0.9 * u * c1.abs();
            p.shear = side * u;
            p.arm_dx = -1.5 * side * u * s1;
        }
        (Kind::Human, "jump") => {
            if ph < 1.0 / 6.0 {
                p.sy = 0.85;
                p.sx = 1.08;
            } else if ph > 5.0 / 6.0 {
                p.sy = 0.9;
                p.sx = 1.05;
            } else {
                let a = (PI * (ph - 1.0 / 6.0) / (4.0 / 6.0)).sin();
                p.ty = -3.0 * u * a;
                p.sy = 1.06;
                p.sx = 0.96;
                p.legs_side = -0.5 * side * u * a;
                p.arm_dy = -u * a;
            }
        }
        (Kind::Human, "crouch") => {
            p.sy = 0.78;
            p.sx = 1.06;
            p.torso_dy = breathe;
        }
        (Kind::Human, "crouch_walk") => {
            p.sy = 0.8;
            p.sx = 1.05;
            p.legs_side = side * u * s1;
            p.legs_front = front * 0.7 * u * s1;
        }
        (Kind::Human, "roll") => {
            p.rot = 2.0 * PI * ph;
            p.sx = 0.85;
            p.sy = 0.8;
            p.ty = 0.1 * h;
        }
        (Kind::Human, "punch") => {
            let b = bump(ph, 0.25, 0.3);
            p.arm_dx = 2.0 * u * b;
            p.shear = side * u * b;
        }
        (Kind::Human, "kick") => {
            let b = bump(ph, 0.4, 0.35);
            p.kick_dx = 2.5 * u * b;
            p.kick_dy = -1.5 * u * b;
            p.shear = -0.8 * u * b;
        }
        (Kind::Human, "spin_kick") => {
            p.flip = (0.25..0.75).contains(&ph);
            let b = bump(ph, 0.5, 0.3);
            p.kick_dx = 2.5 * u * b;
            p.kick_dy = -1.5 * u * b;
            p.ty = -u * (PI * ph).sin();
        }
        (Kind::Human, "slash") => {
            let (ax, ay) = if ph < 1.0 / 3.0 {
                let k = ph * 3.0;
                (-k, -k)
            } else if ph < 2.0 / 3.0 {
                (2.0, 1.0)
            } else {
                let k = (1.0 - ph) * 3.0;
                (2.0 * k, k)
            };
            p.arm_dx = ax * u;
            p.arm_dy = ay * u;
            p.shear = if ax > 1.0 { side * u } else { 0.0 };
        }
        (Kind::Human, "cast") => {
            let b = bump(ph, 0.5, 0.5);
            p.sy = 1.0 + 0.06 * b;
            p.arm_dy = -1.5 * u * b;
            p.ty = -0.5 * u * b;
            p.flash = 0.45 * b;
            p.flash_rgb = [140, 200, 255];
        }
        (Kind::Human, "block") => {
            p.arm_dx = u;
            p.shear = -0.8 * u * side;
            p.sx = 0.96;
            p.torso_dy = if ph >= 0.5 { 0.9 * u } else { 0.0 };
        }
        (Kind::Human, "hurt") => {
            let k = 1.0 - ph;
            p.flash = 0.7 * k;
            p.flash_rgb = [255, 60, 60];
            p.tx = -1.5 * u * k * side;
            p.shear = -u * k;
        }
        (Kind::Human, "death") => {
            p.pivot_feet = true;
            p.lie_down = true;
            p.rot = -0.5 * PI * smoothstep(ph * 1.15);
        }
        (Kind::Human, "cheer") => {
            p.ty = -u * (2.0 * PI * ph).sin().abs();
            p.arm_dy = -u;
            p.arm_dx = 0.8 * u * s1;
        }
        (Kind::Human, "wave") => {
            p.arm_dy = -u;
            p.arm_dx = u * s1;
        }
        (Kind::Quadruped, "idle") => {
            p.legs_band = 0.35;
            p.torso_dy = breathe;
        }
        (Kind::Quadruped, "walk") => {
            p.legs_band = 0.35;
            p.legs_side = side * u * s1;
            p.legs_front = front * u * s1;
            p.torso_dy = 0.9 * u * s1.abs() * 0.6;
        }
        (Kind::Quadruped, "run") => {
            p.legs_band = 0.35;
            p.legs_side = side * 2.0 * u * s1;
            p.legs_front = front * 1.5 * u * s1;
            p.sx = 1.0 + 0.08 * s1 * side;
            p.ty = -u * c1.max(0.0);
            p.rot = 0.07 * s1 * side;
        }
        (Kind::Bird, "fly") => {
            p.wing_dy = -2.0 * u * s1;
            p.ty = 0.6 * u * s1;
        }
        (Kind::Bird, "glide") => {
            p.ty = breathe;
            p.rot = 0.05 * s1;
            p.wing_dy = 0.6 * u * s1;
        }
        _ => {}
    }
    p
}

fn render(base: &Sprite, bb: (i32, i32, i32, i32), p: &Pose, near: &mut NearestCache) -> Sprite {
    let src = if p.flip { base.mirrored() } else { base.clone() };
    let bb = if p.flip { src.bbox().unwrap_or(bb) } else { bb };
    let (x0, y0, x1, y1) = bb;
    let (w, h) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
    let s = src.size as i32;

    // 1. part shifts (forward mapping; moved parts are drawn on top)
    let mut g = src.blank_like();
    let mut moved = Vec::new();
    for y in 0..s {
        for x in 0..s {
            let k = src.get(x, y);
            if k < 0 {
                continue;
            }
            let ry = (y - y0) as f32 / (h - 1.0).max(1.0);
            let rx = (x - x0) as f32 / (w - 1.0).max(1.0);
            let (mut dx, mut dy) = (0.0f32, 0.0f32);
            let leg_top = 1.0 - p.legs_band;
            if ry > leg_top {
                let depth = (ry - leg_top) / p.legs_band;
                dx += if rx >= 0.5 { p.legs_side * depth } else { -p.legs_side * depth };
                if rx < 0.5 {
                    dy -= p.legs_front.max(0.0);
                } else {
                    dy -= (-p.legs_front).max(0.0);
                }
                if rx >= 0.5 {
                    dx += p.kick_dx * depth;
                    dy += p.kick_dy * depth;
                }
            }
            if ry < 0.62 {
                dy += p.torso_dy;
            }
            if (0.25..0.7).contains(&ry) && rx > 0.6 {
                dx += p.arm_dx;
                dy += p.arm_dy;
            }
            let edge = (rx - 0.5).abs() * 2.0;
            if edge > 0.35 && ry < 0.8 {
                dy += p.wing_dy * (edge - 0.35) / 0.65;
            }
            let (dx, dy) = (dx.round() as i32, dy.round() as i32);
            if dx == 0 && dy == 0 {
                g.set(x, y, k);
            } else {
                moved.push((x + dx, y + dy, k));
            }
        }
    }
    for (x, y, k) in moved {
        g.set(x, y, k);
    }

    // 2. lean: rows shifted proportionally to their height above the feet
    if p.shear.abs() >= 0.5 {
        let mut l = g.blank_like();
        for y in 0..s {
            let off = (p.shear * (y1 - y) as f32 / (h - 1.0).max(1.0)).round() as i32;
            for x in 0..s {
                let k = g.get(x, y);
                if k >= 0 {
                    l.set(x + off, y, k);
                }
            }
        }
        g = l;
    }

    // 3. squash / stretch about the feet, rotation, translation (inverse mapping, nearest)
    let mut ty = p.ty;
    if p.lie_down {
        ty -= 0.5 * w * p.rot.sin().abs();
    }
    let identity = p.sx == 1.0 && p.sy == 1.0 && p.rot == 0.0 && p.tx.round() == 0.0 && ty.round() == 0.0;
    let mut out = if identity {
        g
    } else {
        let mut o = g.blank_like();
        let (ax, ay) = ((x0 + x1 + 1) as f32 / 2.0, (y1 + 1) as f32);
        let (px, py) = if p.pivot_feet { (ax, ay) } else { ((x0 + x1 + 1) as f32 / 2.0, (y0 + y1 + 1) as f32 / 2.0) };
        let (sn, cs) = p.rot.sin_cos();
        let (tx, ty) = (p.tx.round(), ty.round());
        for y in 0..s {
            for x in 0..s {
                let (mut qx, mut qy) = (x as f32 + 0.5 - tx, y as f32 + 0.5 - ty);
                // inverse rotation about the pivot
                let (dx, dy) = (qx - px, qy - py);
                qx = px + dx * cs + dy * sn;
                qy = py - dx * sn + dy * cs;
                // inverse scale about the feet
                qx = ax + (qx - ax) / p.sx;
                qy = ay + (qy - ay) / p.sy;
                let k = g.get(qx.floor() as i32, qy.floor() as i32);
                if k >= 0 {
                    o.set(x, y, k);
                }
            }
        }
        o
    };

    // 4. colour flash
    if p.flash > 0.01 {
        tint(&mut out, p.flash_rgb, p.flash, near);
    }
    out
}

/// Blends every colour towards `rgb` by `amount` and maps it to the nearest palette colour.
fn tint(f: &mut Sprite, rgb: [u8; 3], amount: f32, near: &mut NearestCache) {
    let pal = f.palette.clone();
    for k in f.px.iter_mut() {
        if *k < 0 {
            continue;
        }
        let c = pal[*k as usize];
        let t = [0, 1, 2].map(|i| (c[i] as f32 + (rgb[i] as f32 - c[i] as f32) * amount).round() as u8);
        *k = near.get(&pal, t);
    }
}

#[derive(Default)]
struct NearestCache(HashMap<[u8; 3], i16>);

impl NearestCache {
    fn get(&mut self, pal: &[[u8; 3]], c: [u8; 3]) -> i16 {
        *self.0.entry(c).or_insert_with(|| {
            let d = |p: &[u8; 3]| (0..3).map(|i| (p[i] as i32 - c[i] as i32).pow(2)).sum::<i32>();
            pal.iter().enumerate().min_by_key(|(_, p)| d(p)).map(|(i, _)| i as i16).unwrap_or(0)
        })
    }
}

/// Nearest-neighbour resize of a whole sprite canvas.
pub fn resize_nearest(s: &Sprite, size: usize) -> Sprite {
    let mut out = Sprite::new(size, vec![-1; size * size], s.palette.clone());
    for y in 0..size {
        for x in 0..size {
            out.px[y * size + x] = s.px[(y * s.size / size) * s.size + x * s.size / size];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::animator::AZIMUTHS;

    fn blob() -> Sprite {
        let mut px = vec![-1i16; 16 * 16];
        for y in 2..15 {
            for x in 5..11 {
                px[y * 16 + x] = ((x + y) % 3) as i16;
            }
        }
        Sprite::new(16, px, Arc::new(vec![[200, 30, 30], [30, 200, 30], [30, 30, 200]]))
    }

    #[test]
    fn every_action_and_direction_draws_frames() {
        let a = MockAnimator;
        let s = blob();
        for kind in Kind::ALL {
            for spec in a.actions(kind) {
                for az in AZIMUTHS {
                    let req = AnimRequest { kind, action: spec.name.clone(), azimuth: az, n_frames: 0, size: 16 };
                    let anim = a.animate(&s, &req).unwrap();
                    assert_eq!(anim.frames.len(), spec.frames);
                    assert!(anim.frames.iter().all(|f| f.opaque_count() > 0), "{kind:?} {} {az}", spec.name);
                }
            }
        }
    }

    #[test]
    fn walk_moves() {
        let a = MockAnimator;
        let req = AnimRequest { kind: Kind::Human, action: "walk".into(), azimuth: 90, n_frames: 8, size: 16 };
        let anim = a.animate(&blob(), &req).unwrap();
        assert!(anim.frames.windows(2).any(|w| w[0].px != w[1].px));
    }
}
