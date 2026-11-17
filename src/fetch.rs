//! Downloads with Windows' own `curl.exe` (part of Windows 10 1803 and later): no extra dependency, the system
//! proxy settings apply, and a release build needs nothing but the files next to the exe.

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, bail};

/// A `Command` that never opens a console window (the app is a GUI-subsystem program).
pub fn quiet_command(program: &str) -> Command {
    let mut c = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    c
}

/// Downloads `url` to `dest` (through `<dest>.part`, so a broken download never looks finished). `say` gets
/// "<label>: 45 MB" lines while the file grows.
pub fn download(url: &str, dest: &Path, label: &str, say: &mut dyn FnMut(String)) -> Result<()> {
    if let Some(dir) = dest.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut part = dest.as_os_str().to_owned();
    part.push(".part");
    let part = Path::new(&part).to_path_buf();
    let _ = fs::remove_file(&part);
    say(format!("Downloading {label}…"));
    let mut child = quiet_command("curl.exe")
        .args(["-L", "--fail", "--silent", "--show-error", "--retry", "3", "--connect-timeout", "30", "-o"])
        .arg(&part)
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("running curl.exe (it is part of Windows 10 1803 and later)")?;
    let mut last = 0;
    let status = loop {
        if let Some(st) = child.try_wait()? {
            break st;
        }
        std::thread::sleep(Duration::from_millis(400));
        let mb = fs::metadata(&part).map(|m| m.len()).unwrap_or(0) / 1_000_000;
        if mb >= last + 5 {
            last = mb;
            say(format!("Downloading {label}: {mb} MB"));
        }
    };
    if !status.success() {
        let mut err = String::new();
        if let Some(mut e) = child.stderr.take() {
            use std::io::Read;
            let _ = e.read_to_string(&mut err);
        }
        let _ = fs::remove_file(&part);
        bail!("downloading {url} failed: {}", err.trim());
    }
    let _ = fs::remove_file(dest);
    fs::rename(&part, dest).with_context(|| format!("moving the download to {}", dest.display()))?;
    Ok(())
}

/// `https://huggingface.co/<repo>/resolve/<revision>/<file>`
pub fn hf_url(repo: &str, revision: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file}")
}

/// Fetches the listed files of a Hugging Face repo into `dest` (keeping their relative paths); files that are
/// already there are skipped.
pub fn hf_files(repo: &str, revision: &str, files: &[&str], dest: &Path, say: &mut dyn FnMut(String)) -> Result<()> {
    for f in files {
        let to = dest.join(f);
        if fs::metadata(&to).map(|m| m.len() > 0).unwrap_or(false) {
            continue;
        }
        download(&hf_url(repo, revision, f), &to, &format!("{repo} · {f}"), say)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::hf_url;

    #[test]
    fn hf_urls() {
        assert_eq!(hf_url("a/b", "abc", "onnx/x.bin"), "https://huggingface.co/a/b/resolve/abc/onnx/x.bin");
    }
}
