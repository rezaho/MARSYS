// Tauri 2 desktop shell for MARSYS Spren — v0.3 foundation.
//
// Spawns the Python sidecar (`<workspace>/.venv/bin/python -m spren --port 0`),
// reads stdout for the ready signal, captures port + token, opens the webview.
// PyInstaller + canonical sidecar plugin is deferred to Session 10.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use regex::Regex;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::Manager;

const READY_PATTERN: &str = r"^spren-ready: port=(\d+) token=(\S+)";
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Parse the sidecar's ready line. Pure function, unit-testable.
pub fn parse_ready_line(line: &str) -> Option<(u16, String)> {
    let re = Regex::new(READY_PATTERN).ok()?;
    let caps = re.captures(line.trim())?;
    let port: u16 = caps.get(1)?.as_str().parse().ok()?;
    let token = caps.get(2)?.as_str().to_string();
    Some((port, token))
}

/// Resolve the path to the workspace venv's Python (dev mode only).
fn resolve_python() -> PathBuf {
    let venv = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv");
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

fn spawn_sidecar() -> std::io::Result<Child> {
    let python = resolve_python();
    Command::new(python)
        .args(["-m", "spren", "--port", "0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .stdin(Stdio::piped())
        .spawn()
}

/// Owns the sidecar `Child` plus its stdin handle. The window-close handler
/// asks for `request_shutdown(timeout)` first; only on timeout does it fall
/// back to `force_kill()`.
struct SidecarHandle {
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<ChildStdin>>,
}

/// Returned by [`SidecarHandle::request_shutdown`] when the child does not
/// exit within the requested timeout.
#[derive(Debug)]
pub struct ShutdownTimeout;

impl SidecarHandle {
    fn new(mut child: Child) -> Self {
        let stdin = child.stdin.take();
        Self {
            child: Mutex::new(Some(child)),
            stdin: Mutex::new(stdin),
        }
    }

    /// Send `shutdown\n` to the sidecar's stdin (closing the pipe afterwards
    /// to surface EOF to the reader thread on the other side), then poll for
    /// the child to exit. Returns `Ok(())` once the child has reaped, or
    /// `Err(ShutdownTimeout)` if `timeout` elapses first; in the timeout case
    /// the child is restored into the handle so a follow-up `force_kill` can
    /// reach it.
    fn request_shutdown(&self, timeout: Duration) -> Result<(), ShutdownTimeout> {
        if let Some(mut stdin) = self.stdin.lock().expect("stdin mutex").take() {
            let _ = stdin.write_all(b"shutdown\n");
            let _ = stdin.flush();
            // Drop closes the pipe; combined with the explicit `shutdown\n`
            // line, both the line-driven path and the EOF path are covered.
            drop(stdin);
        }

        let Some(mut child) = self.child.lock().expect("child mutex").take() else {
            return Ok(());
        };

        let deadline = Instant::now() + timeout;
        loop {
            match child.try_wait() {
                Ok(Some(_status)) => return Ok(()),
                Ok(None) => {
                    if Instant::now() >= deadline {
                        *self.child.lock().expect("child mutex") = Some(child);
                        return Err(ShutdownTimeout);
                    }
                    std::thread::sleep(POLL_INTERVAL);
                }
                Err(_) => return Ok(()),
            }
        }
    }

    /// Last-resort: SIGKILL the child and reap. Idempotent.
    fn force_kill(&self) {
        if let Some(mut child) = self.child.lock().expect("child mutex").take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn main() {
    env_logger::init();

    let mut child = spawn_sidecar().expect("failed to spawn spren sidecar");
    let stdout = child.stdout.take().expect("sidecar stdout");

    // Read first line, which must be the ready signal.
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .expect("failed to read sidecar ready line");
    let (port, token) = parse_ready_line(&line)
        .unwrap_or_else(|| panic!("sidecar did not emit a ready signal; got: {line:?}"));

    log::info!("sidecar ready on port {port}");

    // Drain remaining stdout in a background thread so the child doesn't block.
    std::thread::spawn(move || {
        for l in reader.lines().map_while(Result::ok) {
            log::debug!("[sidecar] {l}");
        }
    });

    let sidecar = Arc::new(SidecarHandle::new(child));
    let sidecar_for_close = Arc::clone(&sidecar);

    let init_script = format!(
        "window.__SPREN_AUTH__ = {token:?}; window.__SPREN_PORT__ = {port};",
        token = token,
        port = port
    );

    tauri::Builder::default()
        .manage(Arc::clone(&sidecar))
        .setup(move |app| {
            let main_window = app.get_webview_window("main").expect("main window missing");
            main_window.eval(&init_script)?;
            Ok(())
        })
        .on_window_event(move |_window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                match sidecar_for_close.request_shutdown(SHUTDOWN_TIMEOUT) {
                    Ok(()) => log::info!("sidecar exited cleanly"),
                    Err(ShutdownTimeout) => {
                        log::warn!(
                            "sidecar did not exit within {:?}; force-killing",
                            SHUTDOWN_TIMEOUT
                        );
                        sidecar_for_close.force_kill();
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ready_line_happy() {
        let line = "spren-ready: port=8765 token=abc123_xyz data-dir=/tmp/spren";
        let (port, token) = parse_ready_line(line).expect("should parse");
        assert_eq!(port, 8765);
        assert_eq!(token, "abc123_xyz");
    }

    #[test]
    fn parse_ready_line_with_trailing_whitespace() {
        let line = "spren-ready: port=42 token=tok123\n";
        let (port, token) = parse_ready_line(line).expect("should parse");
        assert_eq!(port, 42);
        assert_eq!(token, "tok123");
    }

    #[test]
    fn parse_ready_line_rejects_garbage() {
        assert!(parse_ready_line("hello world").is_none());
        assert!(parse_ready_line("").is_none());
        assert!(parse_ready_line("spren-ready: port= token=").is_none());
        assert!(parse_ready_line("spren-ready: port=NaN token=foo").is_none());
    }

    #[test]
    fn parse_ready_line_strips_to_first_token() {
        let line = "spren-ready: port=8765 token=abc data-dir=/x";
        let (_, token) = parse_ready_line(line).expect("should parse");
        // \S+ in the regex matches up to next whitespace, so token excludes ` data-dir=...`.
        assert_eq!(token, "abc");
    }

    /// Spawn a small subprocess that reads one line from stdin, sleeps a bit,
    /// then exits 0. Used as a controllable stand-in for the real sidecar.
    fn spawn_mock_sidecar(extra_sleep_ms: u64) -> Child {
        let script = format!(
            "import sys, time; sys.stdin.readline(); time.sleep({}); sys.exit(0)",
            extra_sleep_ms as f64 / 1000.0
        );
        Command::new(resolve_python())
            .args(["-c", &script])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn mock sidecar")
    }

    /// Spawn a subprocess that ignores its stdin and sleeps forever. Used to
    /// drive the timeout path of `request_shutdown`.
    fn spawn_unresponsive_sidecar() -> Child {
        let script = "import time; time.sleep(60)";
        Command::new(resolve_python())
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn unresponsive sidecar")
    }

    #[test]
    fn request_shutdown_returns_ok_when_child_exits_promptly() {
        let child = spawn_mock_sidecar(50);
        let handle = SidecarHandle::new(child);
        handle
            .request_shutdown(Duration::from_secs(2))
            .expect("child should exit on shutdown line");
    }

    #[test]
    fn request_shutdown_returns_timeout_when_child_ignores_stdin() {
        let child = spawn_unresponsive_sidecar();
        let handle = SidecarHandle::new(child);
        let err = handle
            .request_shutdown(Duration::from_millis(300))
            .expect_err("unresponsive child should hit the deadline");
        let _ = err; // ShutdownTimeout has no payload.
        handle.force_kill();
    }

    #[test]
    fn force_kill_is_idempotent() {
        let child = spawn_unresponsive_sidecar();
        let handle = SidecarHandle::new(child);
        handle.force_kill();
        handle.force_kill(); // second call is a no-op.
    }
}
