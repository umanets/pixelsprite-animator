//! ONNX Runtime environment and execution providers: CUDA first, DirectML second, CPU last.
//!
//! The ONNX Runtime build (pyke's `cuda13,tensorrt,nvrtx,directml` package for ONNX Runtime 1.31, downloaded by
//! `ort-sys` at build time) is linked into the executable; its CUDA provider (`onnxruntime_providers_cuda.dll`,
//! copied next to the exe) needs the CUDA 13 runtime (cudart64_13, cublas64_13, cublasLt64_13, cufft64_12) and
//! cuDNN 9 built for CUDA 13 (cudnn64_9.dll) on the DLL search path. A CUDA 13 toolkit puts the former on PATH, but
//! most machines have neither, so [`download_cuda_libs`] fetches NVIDIA's own wheels from PyPI and keeps only their
//! DLLs in `<app>/models/cuda/`. [`setup_dll_paths`] prepends the folders it finds to PATH.
//! `ANIMATOR_UI_IGNORE_SYSTEM_CUDA=1` makes the app look only in its own folder (to test a clean machine).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Result, anyhow, bail};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use crate::config::{Config, models_dir};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ep {
    Cuda,
    DirectMl,
    Cpu,
}

impl Ep {
    pub fn label(self) -> &'static str {
        match self {
            Ep::Cuda => "CUDA",
            Ep::DirectMl => "DirectML",
            Ep::Cpu => "CPU",
        }
    }
    pub fn is_gpu(self) -> bool {
        self != Ep::Cpu
    }
}

/// What the user asked for; `Auto` tries CUDA, then DirectML, then the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpChoice {
    Auto,
    Only(Ep),
}

impl EpChoice {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "cuda" => EpChoice::Only(Ep::Cuda),
            "directml" | "dml" => EpChoice::Only(Ep::DirectMl),
            "cpu" => EpChoice::Only(Ep::Cpu),
            _ => EpChoice::Auto,
        }
    }
    /// Providers to try, best first. A forced GPU provider still falls back to the CPU so the app keeps working.
    pub fn order(self) -> Vec<Ep> {
        match self {
            EpChoice::Auto => vec![Ep::Cuda, Ep::DirectMl, Ep::Cpu],
            EpChoice::Only(Ep::Cpu) => vec![Ep::Cpu],
            EpChoice::Only(ep) => vec![ep, Ep::Cpu],
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            EpChoice::Auto => "Auto (CUDA > DirectML > CPU)",
            EpChoice::Only(ep) => ep.label(),
        }
    }
}

static ENV: OnceLock<Environment> = OnceLock::new();

/// Cached result of the first [`setup_dll_paths`] call; set exactly once per process.
static DLL_REPORT: OnceLock<DllReport> = OnceLock::new();

/// Errors unless `report` has been filled, i.e. [`setup_dll_paths`] has run.
fn require_dll_setup(report: &OnceLock<DllReport>) -> Result<()> {
    if report.get().is_none() {
        bail!("setup_dll_paths must run before the ONNX Runtime environment is created");
    }
    Ok(())
}

/// The one ONNX Runtime environment of the process. Errors if [`setup_dll_paths`] has not run.
pub fn environment() -> Result<&'static Environment> {
    require_dll_setup(&DLL_REPORT)?;
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::init().with_name("animator_ui").with_telemetry(false).build().map_err(|e| anyhow!("ONNX Runtime init: {e}"))?;
    Ok(ENV.get_or_init(|| env))
}

/// Result of [`setup_dll_paths`], shown in the UI and the self-test.
#[derive(Debug, Clone, Default)]
pub struct DllReport {
    pub added: Vec<PathBuf>,
    pub cudnn: Option<PathBuf>,
    pub cudart: Option<PathBuf>,
}

impl DllReport {
    pub fn summary(&self) -> String {
        let f = |p: &Option<PathBuf>| p.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "NOT FOUND".into());
        format!("cudart64_13: {} · cudnn64_9 (CUDA 13): {}", f(&self.cudart), f(&self.cudnn))
    }
}

/// Local folder where [`download_cuda_libs`] puts cuDNN.
pub fn local_cudnn_dir() -> PathBuf {
    models_dir().join("cuda").join("cudnn")
}

/// Local folder where [`download_cuda_libs`] puts the CUDA 13 runtime (cudart, cuBLAS, cuFFT).
pub fn local_cuda_dir() -> PathBuf {
    models_dir().join("cuda").join("runtime")
}

/// An NVIDIA driver is installed (its CUDA entry point is in System32); without one CUDA cannot work at all.
pub fn has_nvidia_driver() -> bool {
    std::env::var_os("SystemRoot").is_some_and(|r| Path::new(&r).join("System32").join("nvcuda.dll").is_file())
}

fn ignore_system_cuda() -> bool {
    std::env::var("ANIMATOR_UI_IGNORE_SYSTEM_CUDA").is_ok_and(|v| v == "1")
}

fn has_cuda_dll(dir: &Path) -> bool {
    ["cudnn64_9.dll", "cudart64_13.dll", "cublas64_13.dll"].iter().any(|n| dir.join(n).is_file())
}

/// Prepends CUDA 13 / cuDNN folders to PATH (affects later LoadLibrary calls of this process only).
/// Idempotent: the PATH is changed by the first call only, later calls return a clone of the first report
/// (their `cfg` is ignored).
pub fn setup_dll_paths(cfg: &Config) -> DllReport {
    DLL_REPORT.get_or_init(|| compute_and_apply_dll_paths(cfg)).clone()
}

fn compute_and_apply_dll_paths(cfg: &Config) -> DllReport {
    let mut cands: Vec<PathBuf> = cfg.cuda_dll_dirs.clone().unwrap_or_default();
    cands.push(local_cudnn_dir());
    cands.push(local_cuda_dir());
    let system = !ignore_system_cuda();
    for var in ["CUDNN_PATH", "CUDNN_HOME"].into_iter().filter(|_| system) {
        if let Ok(p) = std::env::var(var) {
            let p = PathBuf::from(p);
            cands.push(p.join("bin"));
            cands.extend(dirs_with(&p.join("bin"), "cudnn64_9.dll", 2));
        }
    }
    cands.extend(dirs_with(Path::new(r"C:\Program Files\NVIDIA\CUDNN"), "cudnn64_9.dll", 4).into_iter().filter(|d| {
        let s = d.to_string_lossy();
        system && (s.contains("\\13.") || s.contains("cu13"))
    }));
    for var in ["CUDA_PATH", "CUDA_HOME"].into_iter().filter(|_| system) {
        if let Ok(p) = std::env::var(var) {
            let p = PathBuf::from(p);
            cands.push(p.join("bin").join("x64"));
            cands.push(p.join("bin"));
        }
    }
    if let Some(rd) = fs::read_dir(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA").ok().filter(|_| system) {
        let mut v: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("v13"))).collect();
        v.sort();
        for p in v.into_iter().rev() {
            cands.push(p.join("bin").join("x64"));
            cands.push(p.join("bin"));
        }
    }
    let path_now = std::env::var_os("PATH").unwrap_or_default();
    let mut current: Vec<PathBuf> = std::env::split_paths(&path_now).collect();
    if !system {
        current.retain(|p| !has_cuda_dll(p)); // a clean machine has no CUDA folders on PATH
    }
    let mut report = DllReport::default();
    let mut front = Vec::new();
    for c in cands {
        if !c.is_dir() || front.contains(&c) {
            continue;
        }
        if has_cuda_dll(&c) {
            front.push(c);
        }
    }
    for d in front.iter().chain(current.iter()) {
        if report.cudnn.is_none() && d.join("cudnn64_9.dll").is_file() {
            report.cudnn = Some(d.clone());
        }
        if report.cudart.is_none() && d.join("cudart64_13.dll").is_file() {
            report.cudart = Some(d.clone());
        }
    }
    current.retain(|p| !front.contains(p));
    report.added = front.clone();
    front.extend(current);
    if let Ok(joined) = std::env::join_paths(front) {
        // SAFETY: this closure runs at most once per process (it is the initialiser of the `DLL_REPORT` OnceLock),
        // and every call site (`main`, the selftest `main` and its sub-commands) calls `setup_dll_paths` on the
        // main thread right after `Config::load`, before any worker thread is spawned, so no other thread can
        // be reading or writing the process environment concurrently.
        unsafe { std::env::set_var("PATH", joined) };
    }
    report
}

fn dirs_with(root: &Path, file: &str, depth: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if root.join(file).is_file() {
        out.push(root.to_path_buf());
    }
    if depth == 0 {
        return out;
    }
    if let Ok(rd) = fs::read_dir(root) {
        for e in rd.flatten() {
            if e.path().is_dir() {
                out.extend(dirs_with(&e.path(), file, depth - 1));
            }
        }
    }
    out
}

/// One NVIDIA wheel from PyPI (a zip); only the DLLs whose names start with one of `keep` are extracted.
pub struct CudaWheel {
    pub what: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
    pub keep: &'static [&'static str],
    /// `true`: goes to [`local_cudnn_dir`], else to [`local_cuda_dir`].
    pub cudnn: bool,
}

/// What the CUDA provider of ONNX Runtime 1.31 needs next to an NVIDIA driver: the CUDA 13 runtime, cuBLAS, cuFFT
/// and cuDNN 9 built for CUDA 13. All four are NVIDIA's own packages on PyPI.
pub const CUDA_WHEELS: [CudaWheel; 4] = [
    CudaWheel {
        what: "CUDA 13 runtime",
        file: "nvidia_cuda_runtime-13.0.96-py3-none-win_amd64.whl",
        url: "https://files.pythonhosted.org/packages/b7/94/6b867483bec07da24ffa32736c79fabb94ef3a7af4d787a9d4a974868576/nvidia_cuda_runtime-13.0.96-py3-none-win_amd64.whl",
        sha256: "f79298c8a098cec150a597c8eba58ecdab96e3bdc4b9bc4f9983635031740492",
        bytes: 2_927_037,
        keep: &["cudart64_"],
        cudnn: false,
    },
    CudaWheel {
        what: "cuBLAS 13",
        file: "nvidia_cublas-13.1.0.3-py3-none-win_amd64.whl",
        url: "https://files.pythonhosted.org/packages/10/f5/f50bc3f5c2bb57ab8f5b4d78bc1146b57810d42cb8fcb28cbe2e14050376/nvidia_cublas-13.1.0.3-py3-none-win_amd64.whl",
        sha256: "2a3b94a37def342471c59fad7856caee4926809a72dd5270155d6a31b5b277be",
        bytes: 404_355_960,
        keep: &["cublas64_", "cublasLt64_"],
        cudnn: false,
    },
    CudaWheel {
        what: "cuFFT 12",
        file: "nvidia_cufft-12.0.0.61-py3-none-win_amd64.whl",
        url: "https://files.pythonhosted.org/packages/85/b2/f8af21a2ed1beed337a6a02c5a28aeb85441f4d578ec3d529543c775ea4b/nvidia_cufft-12.0.0.61-py3-none-win_amd64.whl",
        sha256: "2abce5b39d2f5ae12730fb7e5db6696533e36c26e2d3e8fd1750bdd2853364eb",
        bytes: 213_342_123,
        keep: &["cufft64_"],
        cudnn: false,
    },
    CudaWheel {
        what: "cuDNN 9.28 for CUDA 13",
        file: "nvidia_cudnn_cu13-9.28.1.4-py3-none-win_amd64.whl",
        url: "https://files.pythonhosted.org/packages/48/9c/26e6bdd31c97c389aa565b49c916be0dec17dd61dad6cd33fee286d1b946/nvidia_cudnn_cu13-9.28.1.4-py3-none-win_amd64.whl",
        sha256: "73d874c6bd8a1c75ca040228f77c4e163efe9bab3b65bd85914e2b7334ac68ee",
        bytes: 407_065_099,
        keep: &["cudnn"],
        cudnn: true,
    },
];

/// The wheels this machine still needs: cuDNN if none was found, the runtime wheels if no CUDA 13 runtime was found.
pub fn missing_cuda_wheels(report: &DllReport) -> Vec<&'static CudaWheel> {
    CUDA_WHEELS.iter().filter(|w| if w.cudnn { report.cudnn.is_none() } else { report.cudart.is_none() }).collect()
}

/// Download size of [`missing_cuda_wheels`] in MB (0 = nothing to download).
pub fn missing_cuda_mb(report: &DllReport) -> u64 {
    missing_cuda_wheels(report).iter().map(|w| w.bytes).sum::<u64>() / 1_000_000
}

/// Downloads the CUDA libraries this machine lacks (see [`missing_cuda_wheels`]; up to about 1 GB) and extracts
/// their DLLs to `<app>/models/cuda/`. The CUDA provider only becomes usable after a restart of the app (DLLs are
/// resolved once per process).
pub fn download_cuda_libs(report: &DllReport, say: &mut dyn FnMut(String)) -> Result<PathBuf> {
    let wheels = missing_cuda_wheels(report);
    for (k, w) in wheels.iter().enumerate() {
        let dest = if w.cudnn { local_cudnn_dir() } else { local_cuda_dir() };
        let n = fetch_wheel_dlls(w, &dest, &format!("{} ({} of {}, {} MB)", w.what, k + 1, wheels.len(), w.bytes / 1_000_000), say)?;
        say(format!("{}: {n} DLLs in {}", w.what, dest.display()));
    }
    let root = models_dir().join("cuda");
    say(format!("CUDA libraries are in {}. Restart the app to use CUDA.", root.display()));
    Ok(root)
}

fn fetch_wheel_dlls(w: &CudaWheel, dest: &Path, label: &str, say: &mut dyn FnMut(String)) -> Result<usize> {
    use sha2::{Digest, Sha256};
    fs::create_dir_all(dest)?;
    let wheel = dest.join(w.file);
    if fs::metadata(&wheel).map(|m| m.len()).unwrap_or(0) != w.bytes {
        crate::fetch::download(w.url, &wheel, label, say)?;
    }
    say(format!("Checking {}…", w.what));
    let mut h = Sha256::new();
    let mut f = fs::File::open(&wheel)?;
    std::io::copy(&mut f, &mut h)?;
    drop(f);
    let got = format!("{:x}", h.finalize());
    if got != w.sha256 {
        let _ = fs::remove_file(&wheel);
        bail!("{}: download hash mismatch ({got})", w.what);
    }
    say(format!("Extracting {}…", w.what));
    let mut zip = zip::ZipArchive::new(fs::File::open(&wheel)?)?;
    let mut n = 0;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let name = entry.name().replace('\\', "/");
        let file = name.rsplit('/').next().unwrap_or_default().to_string();
        if file.to_ascii_lowercase().ends_with(".dll") && w.keep.iter().any(|k| file.starts_with(k)) {
            let mut out = fs::File::create(dest.join(&file))?;
            std::io::copy(&mut entry, &mut out)?;
            n += 1;
        }
    }
    drop(zip);
    let _ = fs::remove_file(&wheel);
    if n == 0 {
        bail!("no DLLs found in {}", w.file);
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_guard_errors_until_setup_has_run() {
        let cell: OnceLock<DllReport> = OnceLock::new();
        let err = require_dll_setup(&cell).unwrap_err().to_string();
        assert!(err.contains("setup_dll_paths must run"), "{err}");
        cell.set(DllReport::default()).unwrap();
        assert!(require_dll_setup(&cell).is_ok());
    }

    #[test]
    fn setup_dll_paths_is_idempotent_and_unlocks_environment_guard() {
        let cfg = Config::default();
        let first = setup_dll_paths(&cfg);
        let path_after_first = std::env::var_os("PATH");
        let second = setup_dll_paths(&cfg);
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
        assert_eq!(path_after_first, std::env::var_os("PATH"), "second call must not touch PATH");
        assert!(require_dll_setup(&DLL_REPORT).is_ok());
        // each added folder appears exactly once in PATH
        let parts: Vec<PathBuf> = std::env::split_paths(&path_after_first.unwrap_or_default()).collect();
        for d in &first.added {
            assert_eq!(parts.iter().filter(|p| *p == d).count(), 1, "{}", d.display());
        }
    }
}

/// Options for one session.
#[derive(Debug, Clone, Copy)]
pub struct SessionOpts {
    pub cuda_mem_limit_mb: usize,
    pub cpu_threads: usize,
}

impl Default for SessionOpts {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        SessionOpts { cuda_mem_limit_mb: 2048, cpu_threads: cores.clamp(1, 8) }
    }
}

/// A CUDA session that replays its whole graph as CUDA graphs (one launch per call instead of hundreds of kernels):
/// the symbolic dimensions are fixed with `dims` (so the shape computations fold away and every node runs on the
/// GPU), and the caller binds every input and output to device buffers that keep their addresses. Calls pick a
/// captured graph with the run option `gpu_graph_id`.
pub fn build_cuda_graph_session(model: &Path, opts: SessionOpts, dims: &[(&str, i64)]) -> Result<Session> {
    use ort::ep as eps;
    let env = environment()?;
    let e = |e: ort::Error<ort::session::builder::SessionBuilder>| anyhow!("{e}");
    let mut b = Session::builder(env).map_err(|e| anyhow!("{e}"))?.with_optimization_level(GraphOptimizationLevel::All).map_err(e)?;
    for (name, v) in dims {
        b = b.with_dimension_override(*name, *v).map_err(e)?;
    }
    b = b
        .with_execution_providers([eps::CUDA::default()
            .with_device_id(0)
            .with_arena_extend_strategy(eps::ArenaExtendStrategy::SameAsRequested)
            .with_memory_limit(opts.cuda_mem_limit_mb << 20)
            .with_tf32(false)
            .with_cuda_graph(true)
            .build()
            .error_on_failure()])
        .map_err(e)?;
    b.commit_from_file(model).map_err(|err| anyhow!("{} (CUDA graph): {err}", model.display()))
}

/// One session on exactly this provider; registration failures are errors (no silent CPU fallback).
pub fn build_session(ep: Ep, model: &Path, opts: SessionOpts) -> Result<Session> {
    use ort::ep as eps;
    let env = environment()?;
    let e = |e: ort::Error<ort::session::builder::SessionBuilder>| anyhow!("{e}");
    let mut b = Session::builder(env)
        .map_err(|e| anyhow!("{e}"))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(e)?;
    b = match ep {
        Ep::Cuda => b
            .with_execution_providers([eps::CUDA::default()
                .with_device_id(0)
                .with_arena_extend_strategy(eps::ArenaExtendStrategy::SameAsRequested)
                .with_memory_limit(opts.cuda_mem_limit_mb << 20)
                // plain fp32 matmuls, like the web demo (TF32 flips more samples away from the demo's sprites)
                .with_tf32(false)
                .build()
                .error_on_failure()])
            .map_err(e)?,
        Ep::DirectMl => b
            .with_memory_pattern(false)
            .map_err(e)?
            .with_parallel_execution(false)
            .map_err(e)?
            .with_execution_providers([eps::DirectML::default().build().error_on_failure()])
            .map_err(e)?,
        Ep::Cpu => b.with_intra_threads(opts.cpu_threads).map_err(e)?,
    };
    b.commit_from_file(model).map_err(|err| anyhow!("{} on {}: {err}", model.display(), ep.label()))
}
