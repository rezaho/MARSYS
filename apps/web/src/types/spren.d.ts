// Ambient declarations for the runtime values the Tauri shell injects via
// init_script (and that browser-tab mode reads from the URL fragment).
//
// `__SPREN_AUTH__` holds the per-launch bearer token returned by the FastAPI
// sidecar's ready signal. `__SPREN_PORT__` is the localhost port the sidecar
// is bound to in the current process group; the frontend uses it to resolve
// the API base URL when running under the Tauri shell.

export {};

declare global {
  interface Window {
    __SPREN_AUTH__?: string;
    __SPREN_PORT__?: number;
  }
}
