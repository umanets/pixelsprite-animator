# Pixel Sprite Studio — developer notes

How the app is built, how it was verified, and what is still rough. For download and first-start instructions see
the [README](../README.md).

Paths such as `pilot3d/…`, `release/…` and `docs/demo/…` point into the author's training workspace and the
[pixelsprite](https://github.com/umanets/pixelsprite) repository; they are not part of this repository. The
self-test modes that compare against PyTorch or the 3D-farm renders need those files and are for reference.

A native Windows desktop app (Rust, egui/eframe, ONNX Runtime). Flow: type a caption, generate sprites with
**pixelsprite canvas v2** (or load a PNG sprite), pick one, decide whether it can be animated and as what (human /
quadruped / bird), set the animation conditions and watch the loop, then export a sprite sheet or a GIF.

**Status:** sprite generation is the real model, a port of the web demo's `engine.js` + `mix.js` with every stage-1
feature of the published demo (v=12): caption mixing ("burger skull" draws skull-burgers, each sprite its own way),
silhouettes (draw / one shape), the example captions, guidance / temperature / top-p, the filter, up to 32 sprites with
the demo's row budget, and saving with the demo's file names. For a caption, size, seed and settings it draws **the
web demo's sprites**, pixel for pixel (checked against the demo's own JavaScript run headless; see *Parity with the web
demo*). It runs on CUDA, DirectML or the CPU. **Animation is the real model too:** AnimGPT
(`pilot3d/train/anim_model.py`, run `anim39`) in ONNX Runtime, on the sprite model's provider, drawn on a background
thread; it turns the character to each direction and then draws the action frame by frame (see *The animation
model*). The procedural mock (labelled MOCK) and playback of the 3D-farm renders (Library) are still there.

## Build and run

```
cd animator_ui
cargo build --release
target\release\animator_ui.exe
```

* Rust 1.92+ (tested with 1.99), Windows 11 x64, MSVC toolchain.
* `ort` downloads pyke's ONNX Runtime 1.31 build `cuda13,tensorrt,nvrtx,directml` at build time. It is linked into
  the exe, and the provider DLLs (`onnxruntime_providers_cuda.dll`, `DirectML.dll`, …) are copied next to it.
  Keep them together if you move the exe.
* First start: the models are downloaded with Windows' own `curl.exe` (`src/fetch.rs`, `src/files.rs`) into
  `models\`: the sprite model `ihor-umanets/pixelsprite-canvas-v2` (at the commit the web demo pins; `decoder.f16`
  is expanded once to a float32 `decoder.data`, 153 MB), the text encoder
  `Xenova/paraphrase-multilingual-MiniLM-L12-v2`, the demo's lexicon and example captions (from the pixelsprite
  repository) and the animation model `ihor-umanets/pixelsprite-animgpt` (`onnx/*`, 157 MB). In a development
  checkout next to the training workspace the sprite model and the lexicon are copied from `..\release\` instead.
* `scripts\package.ps1` builds the release zip (`dist\`).
* `target\release\selftest.exe` is a headless check of the same code paths (see *Verification*).

### GPU (CUDA) requirements

The CUDA provider is a **CUDA 13** build (the only CUDA build pyke ships for Windows with ORT 1.31; it supports
Blackwell / sm_120). It needs:

* an NVIDIA driver with CUDA 13 support (tested with 610.88);
* the CUDA 13 runtime DLLs (`cudart64_13`, `cublas64_13`, `cublasLt64_13`, `cufft64_12`). A CUDA 13 toolkit puts
  them on PATH (`…\CUDA\v13.x\bin\x64`); the app also searches `CUDA_PATH` and `C:\Program Files\NVIDIA GPU
  Computing Toolkit\CUDA\v13*`;
* **cuDNN 9 for CUDA 13** (`cudnn64_9.dll`). This is required at run time: the CUDA `BiasSoftmax` kernel asks for a
  cuDNN handle.

Most machines have neither the runtime nor cuDNN. When an NVIDIA driver is installed and a library is missing, the
status bar and **Settings** show **"Download CUDA libraries (~1027 MB)"**. That button (or
`selftest.exe --download-cudnn`) fetches NVIDIA's own wheels from PyPI (`nvidia-cuda-runtime` 13.0.96,
`nvidia-cublas` 13.1.0.3, `nvidia-cufft` 12.0.0.61, `nvidia-cudnn-cu13` 9.28.1.4), checks their SHA-256, keeps only
the DLLs in `models\cuda\` (1.4 GB) and deletes the wheels. Restart the app afterwards. Only what is missing is
downloaded: a system cuDNN (`CUDNN_PATH` / `CUDNN_HOME` or `C:\Program Files\NVIDIA\CUDNN\v9.*\bin\13.*`) or a
CUDA 13 toolkit is used if present. `ANIMATOR_UI_IGNORE_SYSTEM_CUDA=1` makes the app ignore every system copy (this
is how the release was tested as on a clean machine).

DirectML needs nothing extra (any DX12 GPU). The CPU always works.

## Execution providers

Bottom status bar: a selector (**Auto (CUDA > DirectML > CPU)**, CUDA, DirectML, CPU), the active provider, and
**"(!) N provider(s) skipped"**. Hover over that line to see the exact error of each skipped provider.

Every provider is proved with a two-step warm-up (finite logits) before it is used. If a provider fails, the next one
is tried. A forced GPU provider still falls back to the CPU so the app keeps working.

| Provider | Graph / cache | Why |
|---|---|---|
| CUDA | `decoder_static.onnx`, fixed-size key/value cache **on the GPU**: two cache sets are allocated once per batch and swapped every step through an `IoBinding`. Only `x`, `pos` and the logits cross the bus. | no per-step allocation; ~5–6 ms per step |
| DirectML | `decoder.onnx` (growing cache), cache kept on the GPU through an `IoBinding` | the static graph fails on DirectML in a Reshape node (`node_Reshape_560`, HRESULT 8007023E) |
| CPU | `decoder.onnx`, growing cache, prefix in one call | the static graph's full-cache `Where` per step made the CPU ~6× slower (117 vs 20 ms per step) |

Batches (the demo's row budget): sprites are packed into chunks while their network rows fit the budget (one caption
with guidance = 2 rows per sprite, a mix of two words = 3, of three = 4, no guidance = 1). The default is the demo's
on a desktop with WebGPU, 24 rows at 16×16 and 8 at 24/32 (*More settings → Rows per step*; "as the demo on the CPU"
= 16 / 4, or a custom number). The random numbers are drawn chunk by chunk, so the budget is part of what a seed means.
CUDA uses a 2 GB arena limit (the GPU is shared; override with `cuda_mem_limit_mb`) and TF32 is off, so the matmuls
are plain fp32 as in the web demo. The arena never returns freed memory to the device, and every chunk shape leaves
regions of its own size, so the sampler releases the arena's free regions (a one-row run with ONNX Runtime's
`memory.enable_memory_arena_shrinkage`) whenever the chunk shape changes, and if a chunk still runs out it frees the
arena and draws that chunk again with the same random numbers.

The animation model loads on the provider the sprite model ended up on (CPU as its fallback); see below.

## The animation model (`src/animator/onnx/`, `src/animator/jobs.rs`)

**Model:** AnimGPT, checkpoint `pilot3d/train/runs/anim39/best.pt` (step 136 655 of the 600-minute run, 39.2 M
parameters; the v2 transformer warm-started, plus condition / segment embeddings). One network row draws one frame of
one direction: `[null text, size, S unknown-mask rows] [condition: kind + action + azimuth + elevation + frame +
frame count + reference type] [reference S×S] [previous S×S] [target S×S, shifted]`, colours = v2's global palette
(1024 + EMPTY). Tasks: **view** (previous = reference: the character turned to an azimuth at elevation 0) and
**animation** (frame 0: previous = the view at that azimuth; frame k: previous = frame k-1).

**Files** (`models/animator/`, written by `pilot3d/train/export_anim_onnx.py`, ~155 MB). To export again (CPU,
~1 min + ~5 min of checks): put `onnx`, `onnxscript` and `onnxruntime` in a folder (`uv pip install --target <dir>
onnx onnxscript onnxruntime`), then `PYTHONPATH=<dir> .venv\Scripts\python.exe pilot3d/train/export_anim_onnx.py`
(writes the graphs, tables and `parity.json`) and `... --reference` (writes `parity_ref.json`, ~7 min).

| File | What |
|---|---|
| `prefill.onnx` | v2's growing-cache decoder: `x` (B,T,512) + `past_*` (B,8,P,64) → logits of the last token, `present_*` (B,8,P+T,64). The app feeds the whole *context* (prefix, condition, reference, previous canvas and the target's BOS token: C = 2 + S + 1 + 2S² + 1 = 531 / 1179 / 2084 tokens at 16 / 24 / 32) in **one call per row**, so the context costs 5–40 ms (16 px) / 30–240 ms (32 px, 1–8 rows) instead of 2 000 single-token calls. |
| `step.onnx` | one target pixel per call: `x` (B,1,512), `pos`, `ctx_k_i/ctx_v_i` (B,8,C,64) read only, `past_*` (B,8,S²−1,64) → logits, `present_*` with the new key / value written at `pos`. Attention is one softmax over [context, target] computed in two parts, so the context cache is never copied. GPUs. |
| `step_grow.onnx` | the same with a growing target cache (CPU). |
| `anim.data` | float32 weights shared by the three graphs (external data). |
| `embeddings.bin/.json` | input tables (the app sums rows, as for v2), actions, kinds, azimuths, frame counts, the v2 palette, the layout. |
| `parity.json`, `parity_ref.json` | the exporter's ONNX-vs-PyTorch numbers; PyTorch reference outputs for `selftest --onnx-anim`. |

**How the app runs it:**

* **Caches:** the context cache (B,8,C,64 per layer) and the two target-cache sets are allocated once per (batch,
  size) and reused frame after frame; the prefill writes row j straight into row j of the context cache (its
  outputs are bound to views of it). Nothing grows during a request, so the CUDA arena does not fragment.
* **CUDA graphs** (CUDA only): `step.onnx` gets its own session per (batch padded to 1 / 2 / 4 / 8, size) with the
  symbolic dimensions fixed, all inputs / outputs bound to fixed device buffers, and two captured graphs (cache set
  A → B and B → A, run option `gpu_graph_id`). A step is then one graph launch plus three small copies (x and pos
  in, logits out) instead of ~600 kernel launches: **~7 ms → 1.6–5 ms per step** with identical output. Only the latest
  shape's session is kept (VRAM); a new shape costs ~1 s to set up. `ANIMATOR_UI_NO_CUDA_GRAPH=1` turns them off; a failure while setting
  one up falls back to the plain stepping (shown in the status line).
* **Sampling:** softmax(logits / T) over the reference's colours + EMPTY only, one mulberry32 number per pixel,
  inverse CDF; T = 0.8 by default (the training's sample sheets). Every clip (one view, or all frames of one action
  at one azimuth) has its own random stream (`row_seed(seed, kind, action, azimuth, size)`), so a clip does not
  depend on what else is in its batch, and the same seed gives the same frames.
* **Order and batching:** `draw_clips` draws the views of every azimuth it needs first (one batch), then all clips
  together frame by frame (one batch per frame index, ≤ 8 rows; clips with fewer frames leave early). The fight
  scene is 5 views + 7 clips (walk / punch / spin_kick at 270°, run at 315 / 0 / 45 / 90°) = 9 batched frames.
* **Reference:** the sprite is mapped to v2 tokens (exact colour, else the nearest v2 colour) and frames come back
  in the sprite's own palette. Reference type: "original sprite" at 16 / 32 px (what the app draws), "front render"
  at 24 px (the model never saw an original 24 px sprite) and for library entries (their sprite is the farm's front
  render); the panel can override it.
* **Cache:** every clip is kept per (sprite, size, kind, reference type, action, azimuth, seed, temperature), so the
  preview, the fight scene and the 8-direction export never draw the same clip twice. Finished clips survive Stop.
* **Threading:** `jobs.rs` runs the model on an "animator" thread with three slots (preview, 8-direction sheet,
  scene; the preview goes first). The UI only submits requests, shows the progress bar and can stop a request; it
  never waits on the model. Frame counts are the dataset's (`frames` in `embeddings.json` = `standard_actions`).
* **VRAM:** each animation session has its own CUDA arena capped at `anim_cuda_mem_limit_mb` (default 3072). With
  8 rows at 32 px the caches are 0.8 GB (context) + 0.8 GB (target sets); measured process total: see
  *Verification*.

## Configuration (optional `animator_ui.json` in this folder)

```json
{
  "model_src": "C:/work/pixelsprite/release/hf",
  "demo_src": "C:/work/pixelsprite/release/github/docs/demo",
  "library_roots": ["C:/work/pixelsprite/pilot3d/farm_out_trial", "C:/work/pixelsprite/pilot3d/farm_out_test",
                    "C:/work/pixelsprite/pilot3d/farm_out_full"],
  "cuda_dll_dirs": ["C:/path/to/cudnn/bin"],
  "ep": "auto",
  "cuda_mem_limit_mb": 2048,
  "anim_cuda_mem_limit_mb": 3072,
  "ref_from_walk_front": false
}
All keys are optional; the defaults are the values shown, relative to this folder. `ANIMATOR_UI_HOME` overrides
where the app folder is.

## The UI

The window is a 1:1 implementation of a Figma design: the frames **Studio / Main** and **Studio / Generating**,
both 1440 × 1060. The exported frames and the tools used to compare them with the app are not part of this
repository; mentions of `design/` below refer to them.

The layout is the mock's: a 52 px top bar, a 320 px Generate + Library column down the whole body, a 780 px centre
(gallery + the 487 px Preview stage), a 340 px Animate inspector, and a 200 px Fight scene dock that spans the
centre and the inspector above a 28 px status bar. Panel content sits 16 px from the edge (17 in the inspector,
20 in the centre); cards are radius 12, the stage 14, controls 32 high, chips 24, the Generate / Stop button 44.
Every colour, radius, gap and type size is a token in `src/theme.rs`, and every control is a `src/ui_kit/`
component (primary / ghost / danger buttons, chips, a radius-9 segmented control, badges, dropdowns, the 16 px
checkbox, the 34 × 20 switch, the 100 px slider, the dashed dropzone, and a painted check mark — the ✓ glyph is
missing from Geist Medium, so ticks are drawn). Geist and Geist Mono are embedded from `fonts/`; Ubuntu-Light
stays behind them as the Cyrillic fallback, and numbers, ids, timings, seeds and fps are always Geist Mono.

* **Top bar:** the pixel logo (a 28 px accent square with the 4 px cross), "Pixel Sprite Studio" + "animator" in
  mono, the three-step stepper in its input-coloured pill (done = green ✓, current = accent-soft with an accent
  border, todo = raised dot), the provider chip (dot, provider, last ms/step; hover for the full status),
  **Settings** (the provider state, cuDNN, the animation model's advanced settings) and **Export** (the GIF).
* **Left column, "1 · Generate sprites":** the caption (EN / RU / UA, Ctrl+Enter generates; the well's frame turns
  accent while focused; the foot line carries the mix hint and the key). A caption of two or more words grows the
  demo's **word chips and variant chips** under the well — a one-word caption does not, so the panel matches the
  mock. *Examples* (five of the demo's captions); **Size**; **Seed** + Random + *each time*; **Count**; the
  collapsed **Sampling** section (shape / colour guidance, colour temperature, top-p, the stray-piece filter, the
  row budget, exact caption vectors and the **Silhouette** mode — the mock folds all of it away, so the row the
  old build had for it lives here now); the big **Generate** button (a red-outline **Stop** with an "Esc" hint
  while a run is going, then the progress bar and "batch i/n · step p/q · m ms/step"); the dashed **dropzone**
  (click to pick a file, or drop one on the window: a square 16 / 24 / 32 px sprite or an integer
  nearest-neighbour upscale of one, alpha < 128 = transparent, kind from the file name's words, human if none);
  a rule; then **Library** — "N renders", Rescan, the search field, the All / human / quadruped / bird chips and
  the rows (36 px thumbnail, two-line caption, mono id, kind badge).
* **Centre:** the header («caption», mono meta — while a run draws it says "drawing i of n…"), the ×1…×8 zoom and
  **Save all** (a sheet of the selected run). The gallery shows 176 px cards four per row on the checkerboard; the
  selected card gets the 2 px amber frame and the amber «selected» tag, cards still being drawn say "drawing…" /
  "queued", and a mix's variant is tagged in the card's corner. Right-click a card for **save this sprite (PNG)**
  and **use its shape** (the demo's "shape →"). Under it the **Preview stage**: the 44 px header (step chip,
  "PREVIEW", the action · direction · frames badge, MOCK when it applies, the model note on the right), the big
  canvas at 8× (checkerboard, the frame's ground shadow, the «original» reference card top left, the
  "frame i / n · fps" HUD top right) and the 60 px transport (round play / pause, Restart, the frame strip with
  the amber current frame, the loop switch).
* **Inspector, "2 · Animate":** the selected sprite card (68 px thumbnail in the amber frame, name, size · seed,
  the green kind badge + reason), the **Kind** and **Animator** dropdowns (ONNX carries a green dot), the
  "Build everything from the first walk frame" card, the **Action** chips (26 px, the selected one accent-soft),
  the **Direction** compass (58 × 40 cells, 4 px apart, the sprite in the middle) with the **Frames** / **FPS**
  sliders beside it, and the **Export** block (×1…×8, Sheet PNG, GIF, 8-dir sheet, the saved path).
* **Dock, "Fight scene" (200 px):** the title and "N frames · s", the dataset QC switch, Restart / GIF ×N / Play;
  a seconds ruler; the clip track (input-coloured, radius 10) with one block per clip coloured by its action, its
  duration and the amber playhead (click the track to seek); the walk / run px-per-frame fields and the current
  shot line; the 250 × 168 stage preview in its own card on the right.
* **Status bar (28 px):** "Execution provider" and the selector (opens the menu), the skipped-provider note, the
  active provider line, the latest message, the cuDNN download when it is missing, and on the right the saved
  file plus **Open folder**.

### The "build everything from the first walk frame" option

The checkbox card under the Animator row (**experimental, off by default**, persisted as `ref_from_walk_front` in
`animator_ui.json`). When on, the animation model first draws `walk` (`fly` for birds) at 0° from the sprite as
usual and takes that clip's **first frame** as the reference of every other view, action, direction and the fight
scene — meant to keep the colours from drifting away from the sprite (the anchor of `src/animator/onnx/`; the
preview's note then says `ref walk front`) and exports get a `_refwalk` suffix. When off, every clip is drawn
from the original sprite (`ref original`) and the file names are the default ones. Toggling it draws the
preview and the scene again. It is disabled (with a hint) for the mock and library animators, which do not use a
reference. Clips stay cached per (sprite, size, kind, reference type, action, azimuth, seed, temperature), so
toggling back is instant.

#### Measured: the option is experimental (`selftest --anchor-eval`)

The option was measured off vs on with the same seeds; **it does not reduce colour drift**, so treat it as
experimental. The checkbox is therefore **off by default** in the UI, the same as the animator's own default
(`OnnxSettings::default()`, what the self-tests and every non-UI caller get).

How it is wired (`src/animator/onnx/`): `OnnxSettings.anchor` (`AnchorOpts`: `enabled`, `refk`, `palette`);
`draw_clips`, `draw_views` and `cached` branch on it, so the preview, the 8-direction export, `prefetch` and the
fight scene follow it with no change in `jobs.rs` / `scene/`. The clip of the anchor's own action and direction
(walk 0°) is the first clip itself. A clip's cache key holds a hash of its reference tokens, so anchored clips never
collide with clips drawn from the sprite, and the walk 0° clip is shared by both modes. Kinds without `walk` use
`fly` (birds) or the first action the model has (`anchor_action`).

Open points, decided by measurement (`--variants`, 16 val + 8 app characters at 16 px, 3 + 3 at 32 px):

* **Reference-type token for the anchor: "front render"** (`ANCHOR_REF`). Against "original" at 16 px on the val
  characters: mean-colour shift 15.4 vs 17.1 dE (off: 14.3), histogram distance 0.480 vs 0.495, silhouette IoU the
  same (0.435 vs 0.430). "Original" more often gave thin black or pink silhouettes.
* **Sampling palette: the input sprite's** (`ANCHOR_PALETTE`). Restricting the sampler to the anchor frame's own
  colours changed no metric by more than ~0.1 dE at 16 px, so the safer rule stays: an anchor that lost a colour
  cannot make it unavailable.
* **Strict palette drift is zero either way:** the sampler only draws the reference's colours, so 0.00 % of drawn
  pixels are outside the input sprite's palette in every row. The drift that exists is the model using those colours
  in the wrong places or amounts, which the other metrics measure.

```
target\release\selftest.exe --anchor-eval [--ep cuda|cpu] [--sizes 16,32] [--variants] [--limit N] [--n-app 8]
      [--scenes 2] [--skip-val] [--skip-app] [--resume] [--from-csv [--only-modes off,on-front-inpal]]
      [--anim-mem MB] [--gen-ep cpu] [--seed 1] [--no-media] [--out exports\anchor]
```

Inputs: held-out (split "val") farm characters with their **original v2 sprite** as the input and the farm renders as
ground truth (`anchor_eval\val_chars.json`, made by `anchor_eval\make_val_set.py` from
`pilot3d\train\orig_tokens.npz`: evenly spaced over the sorted val names), and sprites of the app's own
text-to-sprite path (knight, ninja, wizard, cat, dog, parrot, owl, pirate; seed 3). Per character and mode it draws
the views at 45 / 90 / 180 / 270° and, at 90° and 270°, `walk, run, punch` (humans), `walk, run` (quadrupeds),
`fly, glide` (birds); `--scenes N` adds the fight scene for N humans per set and size. Metrics
(`src/anchor_eval.rs`), all against the input sprite's colours: histogram TV (total variation between the frame's
and the input's colour histograms), mean-colour dE (CIE76 between mean colours), clip spread (frame-to-frame RMS of
the body's mean colour inside a clip), and for val characters token agreement and silhouette IoU with the farm
frame. CUDA, seed 1, T 0.8. Values are off / on:

| set | characters | histogram TV | mean-colour dE | clip spread dE | pixels = farm | silhouette IoU |
|---|---|---|---|---|---|---|
| val 16 px | 16 (8 human, 4 quadruped, 4 bird) | 0.428 / 0.480 | 14.25 / 15.41 | 1.96 / 1.96 | 0.796 / 0.798 | 0.533 / 0.435 |
| val 32 px | 12 (4 / 4 / 4) | 0.357 / 0.356 | 13.43 / 12.33 | 0.53 / 1.68 | 0.739 / 0.773 | 0.448 / 0.344 |
| app 16 px | 8 (4 human, 2 quadruped, 2 bird) | 0.467 / 0.538 | 20.43 / 26.23 | 2.65 / 2.54 | – | – |
| app 32 px | 8 (4 / 2 / 2) | 0.554 / 0.599 | 9.11 / 10.45 | 1.51 / 1.71 | – | – |
| fight scene, 16 px | 2 val + 2 app humans | 0.392 / 0.502, 0.394 / 0.484 | 16.05 / 15.72, 11.08 / 15.58 | 2.06 / 2.92, 2.77 / 2.15 | – | – |
| fight scene, 32 px | 2 val + 2 app humans | 0.371 / 0.389, 0.583 / 0.579 | 10.31 / 10.63, 5.90 / 8.12 | 0.27 / 1.89, 1.20 / 1.44 | – | – |

Paired per character (on minus off, mean ± standard error): val 16 px mean-colour dE +1.2 ± 1.2 (6 of 16 better),
IoU −0.098 ± 0.029 (12 of 16 worse); val 32 px mean-colour dE −1.1 ± 1.0 (7 of 12 better), IoU −0.104 ± 0.044
(11 of 12 worse), clip spread +1.14 ± 0.39 (9 of 12 worse); app 16 px mean-colour dE +5.8 ± 1.5 (7 of 8 worse); app
32 px +1.3 ± 2.1. By kind the only consistent gain is quadrupeds on val characters (mean-colour dE 14.9 → 12.4 at
16 px, 13.5 → 10.1 at 32 px); humans lose the most silhouette (IoU 0.590 → 0.469 at 16 px, 0.474 → 0.272 at 32 px).

On the sheets: both modes often fail the turn (dark or black silhouettes for dark inputs). With the option the
side clips keep a lighter, more input-like colour in some cases (the 32 px cat stays yellow / pink where off turns
brown), but shapes lose structure (the 32 px ninja loses its head / body split and eye band; a val robot becomes a
few red fragments where off gives black blobs), because the second-generation reference carries the first clip's
errors. Full tables: `exports\anchor\anchor_table_final.md` (off vs on) and `anchor_table.md` (all variants);
rows in `anchor_metrics.csv`; sheets `<set>_<name>_sheet.png` (rows: [farm], off, on for the views and each action
at 90°), side-by-side GIFs `<set>_<name>_<action>_a090.gif`, scene GIFs `scene_<set>_<name>_<mode>.gif`.

### The fight scene (`src/animator/scene/`)

* **Built from the Animator trait only:** `scene::build` first hands every clip it needs to `prefetch` (the ONNX
  animator draws them as one batch and caches them), then asks the selected animator for `walk` / `punch` /
  `spin_kick` at 270°, `run` at 315°, 0°, 45° (the turn) and 90°, each with its native frame count (`animate`),
  and, where it applies (`applies_dataset_qc`: the library only), for the QC all 8 directions of each action at
  32 px (`animate_directions`). So the library, the mock and the ONNX animator all get the scene with no extra
  code. The frames are in place; the timeline (`scene::timeline`) moves the sprite.
* **Directions:** 270° faces screen-right, 90° screen-left (checked on 120 random farm humans: the punching arm
  reaches right at 270° in 105 of them, left at 90° in 105; the planted foot of the walk slides left at 270°).
* **Availability = the published dataset's rule** (`pilot3d/dataset/pack.py`, `qc`): an action is dropped when more
  than 30 % of its 32 px frames over all 8 directions touch the canvas border, change area outside 0.4–2.5× of
  that direction's first frame, or have a second 4-connected piece > 5 %. Same comparisons as numpy / scipy;
  on 300 farm humans the Rust check and a Python run of `pack.py`'s code kept exactly the same 19. `spin_kick`
  fails for most humans (the leg leaves the canvas), so about 12 % of the library humans qualify (945 of 7 896;
  `selftest --scene-scan` counts them). The QC switch in the panel turns the check off; it is never applied to the mock (the mock only
  moves the reference sprite, which often fills its canvas).
* **Speeds and timing:** one tick = one preview frame (1 / fps); run frames last 70/110 of a tick, as in the farm
  GIFs (`pixelate_all.py`). Walk 1.25 and run 2.0 stage px per frame at 32 px (scaled for 16 / 24): measured on
  qualifying farm humans (~25 px tall, chibi legs, ~5 px steps, so ~1.25 px per walk frame keeps the planted foot
  in place; run strides 1.6–2 px) and consistent with the mocap (the run covers 1.56× the walk's distance per
  frame). Longer-legged characters slide a little. The walk-in is the slow part (~7 s at 8 fps, the whole loop
  ~13 s); raise the fps or the walk speed to taste.

### Kind detection (draft)

From the caption words, using the word lists and rules of `pilot3d/dataset/select_inputs.py` (QUAD, BIRD, SKIP,
ITEMS, NOT_ANIMAL, ROLE):

* a role word makes a **human**, even with an animal word ("cat warrior");
* an object named after an animal ("eagle feather") is **not animatable**;
* body plans the farm skips (fish, snake, penguin, monkey, …) are **not animatable**;
* otherwise bird words give **bird** and four-legged words give **quadruped**;
* no person and no animal word gives **not animatable**.

Russian / Ukrainian words go through a built-in stem dictionary (`classify.rs`, about 300 stems: кот / кіт → cat,
рыцарь / лицар → knight, сова → owl, летучая мышь / кажан → bat, перо → feather, …). Unknown non-English words fall
back to the nearest English list word in the multilingual text encoder's space (CSLS-scored, accepted only at
cosine ≥ 0.93, because single-word embeddings are noisy). The user can always override the kind.

## Stage 1 = the web demo

Every user-facing stage-1 feature of `release/github/docs` (index.html + demo/app.js, engine.js, mix.js, text.js,
v=12) and where it is in the app:

| Demo | App |
|---|---|
| caption (EN / RU / UK), example chips + "no text" with precomputed vectors | caption box; *Examples* (`presets.json`, the demo's own vectors, also used whenever the caption equals an example) |
| word chips: concepts from the lexicon, opacity = probability, attributes in italics, unknown words marked, a tap flips object ↔ attribute; note "mix · N variants" / "one caption" / "only 3 words used" | the same under the caption (unknown = "?"; kinds learned for unknown words are kept for the session, as the demo's page state) |
| variant chips inside / on top / centre / blend / one caption (unfit ones dimmed, at least one on) | the same (unfit ones struck through) |
| mixing (`mix.js` + `Strategy.mix`): variants drawn per sprite from seed ^ 0x5bd1e995, weight maps 1.6 / 0.4, third object 0.6, attributes 0.5–1.0, "one caption" = split guidance on the caption vector | `mix.rs` (parse, plan, drawVariant, variantMaps, V8's `Math.hypot`) + `model::Strategy::mix` in the same sampler |
| the "one caption" vector: exact once the text encoder is loaded, else the normalised sum of the word vectors | the same rule for a fresh demo page; *More settings → mix: exact caption vectors* = the demo after its encoder loaded |
| size 16 / 24 / 32 (24 = window of the 32 canvas) | same |
| Silhouette: new / one shape (source sprite, then seed + 1 with its silhouette) / draw (pad, clear / fill all / invert, right button or Shift erases) | same; "one shape" shows the source first |
| "shape →" on a sprite | *use its shape* (save bar) |
| how many 1–32 (default 8 / 4), seed, rnd, new seed every run (typing a seed unticks it) | same |
| shape / colour guidance, colour temperature, top-p (slider steps 0.25 / 0.05 / 0.01) | same, values snapped to the slider steps (0.8 is the demo's 0.8) |
| redraw sprites with stray 1–2 px pieces (off with a silhouette) | same |
| row budget `rowsFor` (WebGPU 24 / 8, CPU 16 / 4, `?rows=N`) | *Rows per step* (default: WebGPU) |
| per-sprite PNG `<slug>_<size>_s<seed>_<kk>.png` (`_src` for the source), "download sheet" `..._sheet.png` | *selected sprite PNG*, *sheet of its run* / *save sheet*: those names in `exports\`, native size with transparency, `_x<N>` added for ×2 / ×4 / ×8 |
| status line, progress, timing line | status bar progress (pixel, sprites done, seconds); summary + timing per run |
| not ported | the browser plumbing (download cache, WebGPU / crash fallbacks, phone limits), `?strategy=` (a developer switch), the demo's sheet layout (a preview: ×8 / ×5 / ×4 with 8 px gaps; the app's sheet is a plain grid of cells, 8 per row) |

## Architecture

```
src/
  main.rs          egui app shell: `App::new`, the window loop (`eframe::App`), the entry point
  state.rs         the app state types: cards, runs, selections, animation / scene state, the `App` struct
  helpers.rs       textures, small drawing utilities, the PNG picker, demo silhouettes, the placeholder library animator
  ui/              `impl App` per screen area: animate, center (gallery + preview), chrome (top / status bar), demo
                   (scripted demo mode), export_ui, inspector (right panel, settings), left (generate + library),
                   poll (worker / animation-thread messages), scene_ui (fight scene dock)
  worker/          background thread that owns the models: mod (re-exports), handle (the `Worker` the UI holds), messages
                   (`Cmd` / `Msg` and request types), state (load with fallback, encode), generate (mix + draw), classify,
                   presets, headless (no-UI driver for the selftest)
  model/           port of engine.js: tables (input vectors), runners (fixed / growing cache), strategy (plain, split
                   guidance, mix), masks (silhouettes, edge flags, row budget), sampling (temperature, top-p), filter (the
                   empty / detached-pixel filter, 24-in-32 window), engine (`SpriteModel`: load, generate, arena shrinking),
                   params
  mix/             port of mix.js: lexicon, parse (caption -> concepts), plan (variants), weights (weight maps)
  save.rs          saving generated sprites / sheets with the demo's file names
  config.rs        where things live: app folder, model cache, source model files, 3D-farm library roots (animator_ui.json overrides)
  cancel.rs        `Cancelled` (the user stopped a job) and `Failure`, the error sent across thread boundaries
  theme.rs         design tokens of the Figma file (colours, radii, spacing, fonts) and the egui style built from them
  ui_kit/          egui widgets on top of the theme: buttons, cards, chips, controls, misc, text
  anchor_eval.rs   metrics and tables of `selftest --anchor-eval` (colour drift of drawn frames against their input sprite)
  text.rs          MiniLM text encoder (tokenizers crate + ORT CPU), mean pooling + L2 normalisation (text.js)
  ep.rs            ONNX Runtime environment, CUDA / DirectML / CPU sessions (+ CUDA-graph sessions), DLL search
                   path, cuDNN download
  files.rs         model cache: downloads / copies the graphs, float16 -> float32 decoder.data
  fetch.rs         downloads with curl.exe (Hugging Face, PyPI, GitHub)
  classify.rs      kind detection (word lists + RU/UK stems)
  sprite.rs        palette-indexed sprite (-1 = transparent), PNG loading (upscale detection), mulberry32
  export.rs        PNG sheets and GIFs
  animator/
    mod.rs         the Animator trait, actions per kind, azimuths
    mock.rs        MockAnimator (procedural placeholder)
    library.rs     LibraryAnimator + npz / npy reader for the 3D farm
    onnx/          OnnxAnimator: tables (input vectors, sampler), engine + run + runners (prefill and step, fixed / growing
                   cache, CUDA graphs), clips (clip planning + cache), reference (palette mapping, anchor), settings,
                   animator (docs = the contract)
    jobs.rs        the animator thread: preview / 8-direction sheet / scene requests, progress, stop
    scene/         scripted multi-action scenes (the fight scene): script, build (clips via the Animator), qc (dataset QC), timeline
  bin/selftest/    headless verification: main (CLI), stage1, parity, scene, onnx_anim, anchor
parity/
  demo_parity.mjs  the web demo's engine.js / mix.js / text.js run headless under Node (reference sprites)
  encode.mjs       the demo's text.js caption vectors (WebAssembly or native ONNX Runtime)
  cases.json       the parity cases; ref/ref_*.json their reference sprites
```

### The Animator trait

```rust
pub trait Animator: Send + Sync {
    fn name(&self) -> &str;
    fn is_mock(&self) -> bool { false }                       // UI labels mock output
    fn actions(&self, kind: Kind) -> Vec<ActionSpec>;         // name, default frames, loops?
    fn animate(&self, sprite: &Sprite, req: &AnimRequest) -> Result<Animation>;
    fn animate_directions(&self, sprite: &Sprite, req: &AnimRequest, azimuths: &[u16]) -> Result<Vec<Animation>>;
    fn prefetch(&self, sprite: &Sprite, kind: Kind, clips: &[(String, u16)]) -> Result<()> { Ok(()) } // batch + cache
    fn applies_dataset_qc(&self) -> bool { !self.is_mock() }   // the fight scene's pack.py check (library only)
}
pub struct AnimRequest { kind: Kind, action: String, azimuth: u16, n_frames: usize /* 0 = native */, size: usize }
pub struct Animation { frames: Vec<Sprite> /* reference palette */, note: String }
```

* **MockAnimator:** frames are the reference sprite pushed around inside its bounding box. Leg bands swing (side
  views) or lift alternately (front / back views), the upper body bobs, a "front arm" region moves for punch / slash
  / wave, outer columns flap for birds. On top of that: lean, squash / stretch about the feet, rotation (roll,
  death), hops, and colour flashes mapped back to the palette (hurt, cast). 180°–315° are mirrored and back views
  are slightly darker. It has no anatomy; it only makes the loop, the directions and the exports testable.
* **LibraryAnimator:** plays `<S>__anim__<action>__a<az>_f<frame>` from a farm entry's `pix/sprites.npz` (palette
  from `palette`, int16 indices). Actions come from `pix/meta.json` and the kind from `job.json`. The frame count can
  be resampled (nearest). It shows that entry's own frames, not the selected generated sprite.
* **OnnxAnimator:** the animation model (see *The animation model*). `animate` / `animate_directions` / `prefetch`
  go through `draw_clips` (views first, then the frames, all requested azimuths as one batch, cached). On the
  calling thread; the UI wraps it in `OnnxWith` (stop flag, progress, reference type) on the animator thread. The
  planned contract changed in three places: no classifier-free guidance (the model was trained without dropping
  the condition), a separate *view* task draws the turn before the first frame, and the context goes in through
  its own prefill graph instead of token by token.

## Verification (on the dev box: RTX 5070 Ti, i9-9900K, GPU shared with other work)

`selftest.exe [--ep auto|cuda|directml|cpu] [--captions "knight,cat,parrot"] [--sizes 16,32] [--seed 1]`
writes PNG sheets to `exports\selftest*\`.

* **Correct output:** knights, cats and parrots at 16×16, 24×24 and 32×32 look like the web demo's sprites (no
  stripes or noise; the self-test also flags rows that repeat the row above). For "knight", seed 1, 8×16×16, **CPU,
  CUDA and DirectML give pixel-identical sprites**, and the static and growing graphs agree. This needs TF32 off: with
  TF32 on, some CUDA samples drifted.
* **Text encoder:** cosine 0.992–0.997 against the demo's precomputed (Python) caption vectors in `presets.json`.
* **RNG:** `mulberry32` matches the JS `makeRng` bit for bit (unit test with values from Node).

| Provider | 8 sprites 16×16 | 4 sprites 32×32 | per step |
|---|---|---|---|
| CUDA (fixed cache on GPU) | 1.5–1.6 s | 5.6–6.7 s (9.5 s while the GPU was busy) | 5–6 ms |
| DirectML (growing cache on GPU) | 12.5 s | 50.9 s | ~49 ms |
| CPU, 8 threads (growing cache) | 5.5 s | 34.9 s (2 batches of 2) | 17–22 ms |

24×24 (4 sprites) on CUDA: 5.4–5.7 s. Model load: 1–2 s (plus a one-off 153 MB float16 → float32 expansion). CUDA
VRAM: under the 2 GB arena cap.

### Parity with the web demo

`parity/demo_parity.mjs` imports the demo's own `engine.js`, `mix.js` and `text.js` (unchanged, from
`release/github/docs/demo`) under Node 24 and runs the run logic of `app.js` (prepareMix, strategyFor, the modes,
`rowsFor`) copied line by line; the model files are the release's. `selftest --parity <ref.json>` draws the same cases
through the app's worker (the UI's code path) and compares every pixel and every variant label. Setup: `npm install
onnxruntime-web@1.31.1 onnxruntime-node@1.31.1 @huggingface/tokenizers@0.2.0` in `parity\` (or `PARITY_MODULES=<a
node_modules folder>`), then `node demo_parity.mjs --backend web|node [--cache fixed|growing] --out ref\x.json`.

Cases (`parity/cases.json`, 23): mixed "burger skull" seed 3 at 16 / 24 / 32, "potion skull" and "sword carrot" at 16
and 32, "fried burger skull" (an attribute, 4 rows per sprite), 20 sprites (3 chunks), only the "inside" variant,
"frost sword" (attribute + object, an example caption) and with "frost" flipped to an object, exact caption vectors,
the CPU row budget, a Cyrillic mix ("череп бургер", both words through the encoder), a plain caption (guidance 1 / 1),
split guidance 1.5 / 2.5 with T 0.8 and top-p 0.9, "ninja" at 32, 12 sprites, no text, a drawn silhouette at 16 and
24, and "one shape".

Results (17 Nov 2026; `exports\parity\parity_*.txt`):

| Reference (the demo's JavaScript) | App on the CPU | App on CUDA |
|---|---|---|
| `ref_web_fixed.json`: onnxruntime-web, WebAssembly, fixed cache, 1 thread = **the demo's own CPU path**, end to end (11 cases at 16 px + "burger skull" at 32) | **11 / 11 cases, 85 / 85 sprites** | 10 / 11 cases, 84 / 85 sprites |
| `ref_node_growing.json`: onnxruntime-node (native CPU), growing cache, encoder on WebAssembly (all 23 cases) | 22 / 23 cases, 172 / 173 sprites | 22 / 23 cases, 172 / 173 sprites |
| `ref_node_growing_native_encoder.json`: the 5 cases whose vectors come from the encoder, encoder on native ONNX Runtime | **5 / 5 cases, 44 / 44 sprites** | 3 / 5 cases, 42 / 44 sprites |

Every variant label is identical in every case. Among them, pixel-identical everywhere: "burger skull" seed 3 at 16,
24 and 32, "potion skull" and "sword carrot" at 16 and 32, "fried burger skull", 20 sprites in 3 chunks, the CPU row
budget, the plain caption, split guidance with T 0.8 / top-p 0.9, "ninja" at 32, the drawn silhouettes and "one
shape". The few sprites that differ all come from
float noise, never from the logic: (1) the demo's text encoder runs on WebAssembly; the app's encoder is bit-identical
to `text.js` on native ONNX Runtime (`node parity\encode.mjs "..." node`, then `selftest --compare-vectors`: 384 / 384
numbers for five captions, same token ids) but differs from the WebAssembly one by up to 1.3·10⁻⁷; captions made of
lexicon words, the example captions and "no text" never touch the encoder. (2) CUDA's float results differ from the
CPU's (~10⁻⁶ in the logits). Either way a sprite changes only when a random number lands between the two
probabilities: here 2 sprites in ~200 ("череп бургер" sprite 6 on CUDA from pixel 182, one of 12 "knight" sprites
from pixel 235); after that pixel the rest of that sprite differs. The demo's WebGPU backend cannot run headless; it
will show the same kind of rare float flips.

### Stage 1 in the self-test

`selftest.exe` (after the caption sheets) runs the stage-1 checks: a two-word mix (8 variants, deterministic, different
from the plain caption), a three-concept mix with an attribute within the 24-row budget, a word-kind override, a drawn
silhouette at 16 and 24 px (every sprite fills exactly the mask), "draw" without a silhouette refused, "one shape"
(every sprite has the source's silhouette), saving 8 sprites at ×1 and ×4 and the sheet with the demo's names and
loading them back pixel-identical, and mixed vs plain timings. `--skip-stage1` / `--skip-stage1-timing` leave them
out; the overview sheet is `exports\selftest\stage1\stage1_overview_16.png`. Unit tests (`cargo test --release`):
`mix::tests` (the regex, parsing order and kinds, plans, labels, weight maps, V8's `Math.hypot` against Node),
`save::tests` (names, PNG round trips at 16 / 24 / 32 px ×1 / ×2 / ×4, sheet cells).

Timings, "burger skull" seed 11, the default row budget (17 Nov, nothing else running on the box's CPU / GPU):

| | plain caption (split guidance, 2 rows / sprite) | mixed (two words, 3 rows / sprite; "one caption" variants 2) |
|---|---|---|
| CUDA, 8 sprites 16×16 | 1.54 s (1 chunk of 16 rows, 5.6 ms/step) | 1.65 s (1 chunk of 24 rows, 6.0 ms/step) |
| CUDA, 4 sprites 32×32 | 5.6 s (1 chunk of 8 rows) | 11.0 s (2 chunks of ≤ 6 rows) |
| CPU (8 threads), 8 sprites 16×16 | 5.6 s | 7.0 s |
| CPU, 4 sprites 32×32 | 30.9 s | 46.7 s (2 chunks) |

At 16 px a mix costs about the same as a plain caption (the 24-row budget holds 8 mixed sprites in one pass, and a
GPU step costs about the same for 16 or 24 rows); at 24 / 32 px the 8-row budget holds two mixed sprites instead of
four, so a mix takes two passes (as in the demo). 20 mixed sprites at 16 px: 4.9 s on CUDA (3 chunks); 24×24, 4 mixed:
11.4 s. Encoding: 0 ms for lexicon words and the examples, ~5 ms per caption otherwise.

Scripted UI check (no mouse needed): `set ANIMATOR_UI_DEMO=caption=knight;size=16;action=walk;az=90;lib=s32_4900;
libaction=walk;out=ui_demo;export=1;exit=1` and then run the exe. It generates, selects, animates, exports and
saves `exports\ui_demo_1.png` / `_2.png` (egui screenshots).

Stage 1 in the UI: `set ANIMATOR_UI_DEMO=caption=burger skull;size=16;seed=3;count=8;more=1;save=1;out=stage1/
burger_skull_16_s3;exit=1` generates the mix, saves the first sprite and the sheet (`exports\burger_skull_16_s3_00.png`,
`..._sheet.png`) and the window (`exports\stage1\burger_skull_16_s3_1.png`). More stage-1 keys: `mode=same|draw`,
`mask=potion|disc`, `variants=inside,over`, `flip=frost` (click a word chip), `rows=N`, `exact=1`, `scale=4`, `idx=K`
(see `Demo` in main.rs). Screenshots of 17 Nov in `exports\stage1\`: `burger_skull_16_s3_1.png` (with More settings),
`burger_skull_32_s3_1.png`, `draw_potion_16_s5_1.png`, `one_shape_crown_16_s6_1.png`, `frost_sword_flip_16_s0_1.png`.

The animation model in the UI: `set ANIMATOR_UI_DEMO=caption=knight;size=32;animator=onnx;action=walk;frames=8;
az=270;out=onnx/ui_onnx_knight32;export=1;scenet=3.0,7.0,7.8,8.8,10.5;exit=1` generates, picks the first knight,
waits until the model has drawn the preview and the fight scene (`wait=` seconds, default 600; 66 s in all on the
dev box), exports and saves the window and five paused scene moments (`exports\onnx\ui_onnx_knight32_*.png`).
`png=<file>;kind=human` loads a PNG instead of generating (`ui_onnx_png16_*.png`: a held-out 16 px character
saved 2x, 10 s in all).

Fight scene check (CPU only, no generation): `set ANIMATOR_UI_DEMO=gen=0;ep=cpu;lib=s16_13614;libsize=32;
libaction=punch;libaz=270;out=scene_demo;export=1;scenet=2.0,7.2,7.9,9.2,10.6;exit=1` saves `scene_demo_2.png`,
pauses the scene at those seconds for `scene_demo_scene_1..5.png` and exports the scene GIF. Headless:

* `selftest.exe --scene-scan [--roots a;b] [--limit N] [--list file]`: runs the app's availability check on every
  library human. On 16 Nov 2026 (14 208 entries, 7 896 humans): **945 have the scene (12 %)**; the dataset QC
  dropped spin_kick for 6 826, run 1 315, walk 1 054, punch 862 (spin_kick alone for 5 508). In 164 of the 945 a
  shown frame still touches the side / top of the 32 px cell (the kicking foot in spin_kick at 270°, 1–3 of its 8
  frames: the QC allows up to 30 %); 479 more only touch the bottom row (feet on the ground line).
* `selftest.exe --scene-scan --gif s16_13614,s32_70245 [--fps 8]`: scene GIFs at 16 / 24 / 32 px in
  `exports\scene\` without scanning the library.
* Unit tests (`cargo test --release`, `animator::scene::tests`): missing actions, the QC decision (also against
  hand-made frames that mimic pack.py's flags), the timeline order, frame counts, positions (enters from the left
  edge, fights at the centre facing right, turns through 315/0/45, runs off the left edge), total length, the
  looping lookup and the GIF delays; the mock gets a scene without the QC. `animator::onnx::tests`: the per-clip
  seed and the sampler against the Python formulas, the palette mapping both ways (exact, nearest, v2 palette),
  the context length; `sprite::tests`: PNG upscale detection.

### The animation model

`selftest.exe --onnx-anim [--ep cuda|cpu] [--sizes 32,16] [--caption knight] [--skip-parity|-gifs|-timing|-scene]`
(report in `exports\onnx\selftest_onnx_<ep>.txt`, GIFs / sheets next to it; kept from the 17 Nov runs:
`selftest_onnx_cuda.txt` (CUDA graphs, everything), `selftest_onnx_cuda_plain_steps.txt` (before CUDA graphs),
`selftest_onnx_cpu_parity16.txt`, `selftest_onnx_cpu_timing.txt`; UI screenshots `ui_onnx_knight32_*.png`,
`ui_onnx_png16_*.png`). Held-out characters (split "val",
never trained on) from `farm_out_full`: a human, a quadruped and a bird at 32 px, a human and a quadruped at 16 px;
reference = the original sprite (`orig_tokens.npz`, what v2 draws).

* **Export parity (Python, `export_anim_onnx.py`, ONNX Runtime CPU vs AnimGPT in PyTorch, float32, teacher
  forced, a view row + an animation row per batch):** max |logit difference| 1.9–2.9·10⁻⁵, argmax identical on
  100 % of pixels, at 16, 24 (front-render reference) and 32 px, for `step.onnx` and `step_grow.onnx`.
* **App parity (Rust + ONNX Runtime CUDA, with and without CUDA graphs, vs PyTorch with the same sampler and
  random numbers, `parity_ref.json`):** teacher-forced bits per frame within 1·10⁻⁴; **35 of 35 sampled frames
  pixel-identical** (views at 90 / 180 / 270, then 4 walk / fly frames from the sampled 90° view, 5 characters).
* **Against the farm (ground truth):** teacher-forced bits per pixel, views 0.55 (32 px) / 0.68 (16 px), walk / fly
  0.22 / 0.22 (training's val set: 0.60 / 0.72 and 0.31 / 0.33). Sampled (T 0.8): views agree with the farm on
  67–84 % of pixels (silhouette IoU 0.72–0.85; the worst single view is the quadruped's back, IoU 0.41), 8-direction
  walks / flights 77–95 % (IoU 0.70–0.84). Sheets `<char>_<S>_views_ref_farm_model.png` (reference / farm / model)
  and `<char>_<S>_<action>_8dirs_farm_model.png` (farm row, model row per direction) show it.

| CUDA, RTX 5070 Ti (shared) | per step | 1 frame | walk, 1 direction | walk, 8 directions | fight scene |
|---|---|---|---|---|---|
| 16 px, CUDA graphs | 1.6 ms (1 row), 2.2 ms (8 rows) | 0.4–0.6 s | 3.6–4.0 s | 5.8 s | 5.5 s |
| 32 px, CUDA graphs | 2.9 ms (1 row), 4.9 ms (8 rows) | 3.0–5.3 s | 27.5 s | 48.2 s | 44–50 s |
| 16 px, plain steps | 7.4 ms (1 or 8 rows) | 1.9 s | 17.0 s | 17.3 s | 16–20 s |
| 32 px, plain steps | 6.9–7.1 ms | 7.1–7.5 s | 64.2 s | 67.3 s | 57–70 s |
| 16 px, CPU (8 threads, growing cache) | 7.9 ms (1 row), 18.7 ms (8 rows) | 2.2–5.9 s | 19.5 s | 53.3 s | – |
| 32 px, CPU | 12.5 ms (1 row) | 13.9 s | 125 s | – (≈ 4 min est.) | – |

A walk = the view at each direction (one batch) + 8 frames (one batch each). The context prefill is 6 ms (16 px,
1 row) to 240 ms (32 px, 8 rows) per frame. Without CUDA graphs a step costs the same at batch 1 and 8 and at 16
and 32 px: it is kernel-launch overhead (~600 nodes under Windows WDDM), not arithmetic. VRAM: the self-test process
peaks at ≈ 2.9–3.2 GB with 8 rows at 32 px (CUDA context, three sessions' weights, 1.6 GB of caches).

## Known gaps / TODO

* **Animation model quality:** animation of a given view is good (walk / fly chains follow the farm closely); the
  **turn to a new direction is the weak part**, especially for animals: silhouettes and colours drift (a darker
  palette, a wrong back view), and every clip starts from that generated view, so its errors carry into all frames.
  Sprites the model never saw (e.g. the app's own "knight") turn worse than held-out farm characters; spin_kick
  (rare in the dataset after the QC) is the messiest action. 24 px uses the "front render" reference type.
* **Speed:** 32 px is slow (≈ 3–5 s per frame, ≈ 45–50 s for the fight scene) even with CUDA graphs; a fused
  attention / in-place KV cache would remove the per-step full copy of the target cache. DirectML runs one row per
  batch (its device pointers cannot be offset) and was not timed; the CPU is a fallback.
* Mock motion knows no anatomy (side-view birds flap beak and tail; it assumes the sprite faces right at 90°,
  while farm renders face left at 90°).
* Fight scene: one speed for every character (long-legged ones slide a little); the walk-in is long (~7 s at
  8 fps); a spin-kick foot can be cut at the edge of its 32 px cell in 1–3 frames (rendered that way by the farm);
  no fight partner / hit effects; library / mock scenes are built on the UI thread (instant), model scenes on the
  animator thread.
* The kind detector is word-based. Captions without a known subject word give "not animatable" (use the override).
  The RU/UK stem list is hand-made and short.
* Sprite model: CUDA speed is bound by per-step overhead (12 layers, small batches); the CUDA-graph path of the
  animation model would apply to it too. DirectML is slow (~49 ms per step) and its static-graph failure is not
  investigated.
* No persistence of settings (the demo keeps the enabled variant chips in localStorage). The stage-1 controls are
  functional, not designed yet.
* Parity with the web demo: pixel-identical on the demo's CPU paths except where the text encoder is involved
  (WebAssembly vs native float differences, ~10⁻⁷; see *Parity with the web demo*); the demo's WebGPU backend was
  not run.
* Switching provider reloads the model (1–2 s). The CUDA libraries need an app restart after download.
