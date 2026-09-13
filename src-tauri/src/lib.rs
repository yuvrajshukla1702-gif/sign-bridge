use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{
  atomic::{AtomicBool, Ordering},
  Mutex,
};
use std::time::Duration;

use tauri::{
  menu::{Menu, MenuItem},
  tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
  AppHandle, Manager, RunEvent,
};

struct ServerState {
  child: Mutex<Option<Child>>,
  spawned: AtomicBool,
}

fn project_dir() -> Option<PathBuf> {
  let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let candidate = manifest.join("..");
  if candidate.join("run_webapp.py").exists() || candidate.join("backend").exists() {
    return Some(candidate.canonicalize().unwrap_or(candidate));
  }
  if let Ok(exe) = std::env::current_exe() {
    if let Some(dir) = exe.parent() {
      for up in [dir.to_path_buf(), dir.join(".."), dir.join("../..")] {
        if up.join("run_webapp.py").exists() {
          return Some(up.canonicalize().unwrap_or(up));
        }
      }
    }
  }
  None
}

fn which_python(project: &PathBuf) -> PathBuf {
  let venv = project.join(".venv\\Scripts\\python.exe");
  if venv.exists() {
    return venv;
  }
  PathBuf::from("python")
}

fn spawn_api_server() -> Option<Child> {
  let dir = project_dir()?;
  let python = which_python(&dir);

  let mut cmd = Command::new(python);
  cmd.args([
    "-m",
    "uvicorn",
    "backend.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
  ]);
  cmd.current_dir(&dir)
    .stdout(Stdio::null())
    .stderr(Stdio::null());

  #[cfg(windows)]
  {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
  }

  cmd.spawn().ok()
}

fn server_healthy() -> bool {
  let Ok(addr) = "127.0.0.1:8000".parse() else {
    return false;
  };
  let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(300)) else {
    return false;
  };
  if stream
    .write_all(b"GET /api/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
    .is_err()
  {
    return false;
  }
  let mut buf = [0u8; 200];
  let Ok(n) = stream.read(&mut buf) else {
    return false;
  };
  let text = String::from_utf8_lossy(&buf[..n]);
  text.contains("200") || text.contains("HTTP/")
}

#[tauri::command]
async fn ensure_server(state: tauri::State<'_, ServerState>) -> Result<bool, String> {
  if server_healthy() {
    state.spawned.store(true, Ordering::Relaxed);
    return Ok(true);
  }

  if !state.spawned.load(Ordering::Relaxed) {
    let mut slot = state.child.lock().map_err(|e| e.to_string())?;
    if slot.is_none() {
      *slot = spawn_api_server();
      state.spawned.store(true, Ordering::Relaxed);
    }
  }

  for i in 0..60 {
    let wait_ms = if i < 10 {
      100
    } else if i < 25 {
      250
    } else {
      400
    };
    tokio::time::sleep(Duration::from_millis(wait_ms)).await;
    if server_healthy() {
      return Ok(true);
    }
  }
  Err("Sign Bridge API did not become ready on :8000".into())
}

fn show_main_window(app: &AppHandle) {
  if let Some(window) = app.get_webview_window("main") {
    let _ = window.show();
    let _ = window.unminimize();
    let _ = window.set_focus();
  }
}

/// WebView2 silently denies getUserMedia unless PermissionRequested is allowed.
#[cfg(windows)]
fn grant_media_permissions(window: &tauri::WebviewWindow) {
  let _ = window.with_webview(|platform| {
    use webview2_com::{
      Microsoft::Web::WebView2::Win32::{
        COREWEBVIEW2_PERMISSION_KIND, COREWEBVIEW2_PERMISSION_KIND_CAMERA,
        COREWEBVIEW2_PERMISSION_KIND_MICROPHONE, COREWEBVIEW2_PERMISSION_STATE_ALLOW,
      },
      PermissionRequestedEventHandler,
    };

    unsafe {
      let controller = platform.controller();
      let Ok(webview) = controller.CoreWebView2() else {
        return;
      };

      let mut token: i64 = 0;
      let handler = PermissionRequestedEventHandler::create(Box::new(|_sender, args| {
        let Some(args) = args else {
          return Ok(());
        };
        let mut kind = COREWEBVIEW2_PERMISSION_KIND::default();
        args.PermissionKind(&mut kind)?;
        if kind == COREWEBVIEW2_PERMISSION_KIND_CAMERA
          || kind == COREWEBVIEW2_PERMISSION_KIND_MICROPHONE
        {
          args.SetState(COREWEBVIEW2_PERMISSION_STATE_ALLOW)?;
        }
        Ok(())
      }));
      let _ = webview.add_PermissionRequested(&handler, &mut token);
      // Keep the handler alive for the process lifetime.
      std::mem::forget(handler);
    }
  });
}

#[cfg(not(windows))]
fn grant_media_permissions(_window: &tauri::WebviewWindow) {}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .manage(ServerState {
      child: Mutex::new(None),
      spawned: AtomicBool::new(false),
    })
    .invoke_handler(tauri::generate_handler![ensure_server])
    .setup(|app| {
      let show_i = MenuItem::with_id(app, "show", "Show Sign Bridge", true, None::<&str>)?;
      let quit_i = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
      let menu = Menu::with_items(app, &[&show_i, &quit_i])?;

      if let Some(icon) = app.default_window_icon().cloned() {
        let _tray = TrayIconBuilder::new()
          .icon(icon)
          .menu(&menu)
          .tooltip("Sign Bridge")
          .on_menu_event(move |app, event| match event.id.as_ref() {
            "show" => show_main_window(app),
            "quit" => app.exit(0),
            _ => {}
          })
          .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
              button: MouseButton::Left,
              button_state: MouseButtonState::Up,
              ..
            } = event
            {
              show_main_window(tray.app_handle());
            }
          })
          .build(app)?;
      }

      let app_handle = app.handle().clone();
      tauri::async_runtime::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        if let Some(state) = app_handle.try_state::<ServerState>() {
          if !server_healthy() {
            let _ = ensure_server(state).await;
          } else {
            state.spawned.store(true, Ordering::Relaxed);
          }
        }
      });

      if let Some(window) = app.get_webview_window("main") {
        grant_media_permissions(&window);
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
      }

      Ok(())
    })
    .build(tauri::generate_context!())
    .expect("error while building Sign Bridge")
    .run(|app_handle, event| {
      if let RunEvent::Exit = event {
        if let Some(state) = app_handle.try_state::<ServerState>() {
          if let Ok(mut slot) = state.child.lock() {
            if let Some(mut child) = slot.take() {
              let _ = child.kill();
            }
          }
        }
      }
    });
}
