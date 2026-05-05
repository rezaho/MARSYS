// Tauri 2 desktop shell for MARSYS Spren — v0.3 foundation.
//
// Spawns the Python sidecar (`<workspace>/.venv/bin/python -m spren --port 0`),
// reads stdout for the ready signal, captures port + token, opens the webview.
// PyInstaller + canonical sidecar plugin is deferred to Session 10.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use regex::Regex;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::Manager;

const READY_PATTERN: &str = r"^spren-ready: port=(\d+) token=(\S+)";

/// Parse the sidecar's ready line. Pure function — unit-testable.
pub fn parse_ready_line(line: &str) -> Option<(u16, String)> {
    let re = Regex::new(READY_PATTERN).ok()?;
    let caps = re.captures(line.trim())?;
    let port: u16 = caps.get(1)?.as_str().parse().ok()?;
    let token = caps.get(2)?.as_str().to_string();
    Some((port, token))
}

/// Resolve the path to the workspace `.venv/bin/python` (dev mode only).
fn resolve_python() -> PathBuf {
    // `apps/desktop/` → `<workspace>/.venv/bin/python`
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir.join("../../.venv/bin/python")
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

#[derive(Default)]
struct SidecarHandle {
    child: Mutex<Option<Child>>,
}

fn main() {
    env_logger::init();

    // Spawn sidecar.
    let mut child = spawn_sidecar().expect("failed to spawn spren sidecar");
    let stdout = child.stdout.take().expect("sidecar stdout");

    // Read first line — must be the ready signal.
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    reader.read_line(&mut line).expect("failed to read sidecar ready line");
    let (port, token) = parse_ready_line(&line)
        .unwrap_or_else(|| panic!("sidecar did not emit a ready signal; got: {line:?}"));

    log::info!("sidecar ready on port {port}");

    // Drain remaining stdout in a background thread so the child doesn't block.
    std::thread::spawn(move || {
        for line in reader.lines() {
            if let Ok(l) = line {
                log::debug!("[sidecar] {l}");
            }
        }
    });

    let sidecar = Arc::new(SidecarHandle {
        child: Mutex::new(Some(child)),
    });
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
                if let Ok(mut guard) = sidecar_for_close.child.lock() {
                    if let Some(mut child) = guard.take() {
                        let _ = child.kill();
                        let _ = child.wait();
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
}
