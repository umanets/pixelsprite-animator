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
    download_auth(url, dest, label, say, None)
}

/// The Hugging Face access token of the `HF_TOKEN` environment variable (needed only for private or gated repos).
pub fn hf_token() -> Option<String> {
    let t = std::env::var("HF_TOKEN").ok()?;
    let t = t.trim();
    (!t.is_empty() && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')).then(|| t.to_owned())
}

/// [`download`] with an optional bearer token. The token goes to curl through its standard input (`--config -`),
/// never on the command line; curl does not pass it on when the server redirects to another host.
fn download_auth(url: &str, dest: &Path, label: &str, say: &mut dyn FnMut(String), bearer: Option<&str>) -> Result<()> {
    if let Some(dir) = dest.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut part = dest.as_os_str().to_owned();
    part.push(".part");
    let part = Path::new(&part).to_path_buf();
    let _ = fs::remove_file(&part);
    say(format!("Downloading {label}…"));
    let mut cmd = quiet_command("curl.exe");
    if bearer.is_some() {
        cmd.args(["--config", "-"]);
    }
    let mut child = cmd
        .args(["-L", "--fail", "--silent", "--show-error", "--retry", "3", "--connect-timeout", "30", "-o"])
        .arg(&part)
        .arg(url)
        .stdin(if bearer.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("running curl.exe (it is part of Windows 10 1803 and later)")?;
    if let (Some(token), Some(mut stdin)) = (bearer, child.stdin.take()) {
        use std::io::Write;
        let _ = writeln!(stdin, "header = \"Authorization: Bearer {token}\"");
    }
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
        hf_file(repo, revision, f, &to, say)?;
    }
    Ok(())
}

/// Downloads one file of a Hugging Face repo to `dest` (with the `HF_TOKEN` token when it is set).
pub fn hf_file(repo: &str, revision: &str, file: &str, dest: &Path, say: &mut dyn FnMut(String)) -> Result<()> {
    let name = file.rsplit('/').next().unwrap_or(file);
    let token = hf_token();
    download_auth(&hf_url(repo, revision, file), dest, &format!("{repo} · {name}"), say, token.as_deref()).map_err(|e| {
        if token.is_none() { e.context("if the repo is private, set the HF_TOKEN environment variable to a read token") } else { e }
    })
}

#[cfg(test)]
mod tests {
    use super::hf_url;

    #[test]
    fn hf_urls() {
        assert_eq!(hf_url("a/b", "abc", "onnx/x.bin"), "https://huggingface.co/a/b/resolve/abc/onnx/x.bin");
    }
}
