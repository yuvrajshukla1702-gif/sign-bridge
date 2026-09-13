/**
 * Ensure Sign Bridge FastAPI is listening on :8000 before Tauri opens.
 * Used as Tauri beforeDevCommand.
 */
import { spawn } from "node:child_process";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const PORT = 8000;

function healthy() {
  return new Promise((resolve) => {
    const req = http.get(
      { host: "127.0.0.1", port: PORT, path: "/api/health", timeout: 800 },
      (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      }
    );
    req.on("error", () => resolve(false));
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
  });
}

function pythonPath() {
  const venv = path.join(root, ".venv", "Scripts", "python.exe");
  if (fs.existsSync(venv)) return venv;
  return "python";
}

async function main() {
  if (await healthy()) {
    console.log("Sign Bridge API already on :8000");
    return;
  }

  console.log("Starting Sign Bridge API on :8000 ...");
  const child = spawn(
    pythonPath(),
    [
      "-m",
      "uvicorn",
      "backend.main:app",
      "--host",
      "127.0.0.1",
      "--port",
      "8000",
    ],
    {
      cwd: root,
      detached: true,
      stdio: "ignore",
      windowsHide: true,
    }
  );
  child.unref();

  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 500));
    if (await healthy()) {
      console.log("Sign Bridge API ready.");
      return;
    }
  }
  console.error("WARNING: API did not become ready in time");
}

main();
