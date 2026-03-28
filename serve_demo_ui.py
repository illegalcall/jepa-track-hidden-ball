#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import time
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
UPLOAD_ROOT = ROOT / "demo_uploads"
SAMPLE_SHEET = ROOT / "demo_cases" / "case_1" / "sheet.png"
LOCAL_INFERENCE_ROOT = ROOT / "local_inference_assets"
LOCAL_CHECKPOINT = ROOT / "local_model_assets" / "lewm_auxonly_123456_h12_epoch_12_object.ckpt"
REMOTE_VM_NAME = os.environ.get("JEPA_VM_NAME", "bench-run-demo-api-cpu")
REMOTE_VM_ZONE = os.environ.get("JEPA_VM_ZONE", "us-central1-c")
REMOTE_PROJECT = os.environ.get("JEPA_PROJECT", "corethink-bench")
REMOTE_WORKDIR = os.environ.get("JEPA_REMOTE_WORKDIR", "/home/dhruvsharma/jepapoc")
REMOTE_CHECKPOINT = os.environ.get(
    "JEPA_REMOTE_CHECKPOINT",
    "/home/dhruvsharma/.stable_worldmodel/ablation_auxonly_123456_h12/lewm_auxonly_123456_h12_epoch_12_object.ckpt",
)
REMOTE_HISTORY_SIZE = int(os.environ.get("JEPA_HISTORY_SIZE", "12"))
REMOTE_DEVICE = os.environ.get("JEPA_DEVICE", "cpu")


def safe_name(filename: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in filename)
    return cleaned or "upload.png"


def run_logged(cmd, *, cwd=None, timeout=900):
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n>>> {printable}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        bufsize=1,
    )
    lines = []
    deadline = time.time() + timeout if timeout else None

    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            print(line, end="", flush=True)
            lines.append(line)
            continue

        if proc.poll() is not None:
            break

        if deadline and time.time() > deadline:
            proc.kill()
            raise subprocess.TimeoutExpired(cmd, timeout)

        time.sleep(0.1)

    if proc.stdout:
        remainder = proc.stdout.read()
        if remainder:
            print(remainder, end="", flush=True)
            lines.append(remainder)

    output = "".join(lines)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=output)
    return output


def extract_json_blob(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in command output.")
    return json.loads(text[start : end + 1])


def has_local_inference():
    required = [
        LOCAL_CHECKPOINT,
        LOCAL_INFERENCE_ROOT / "jepa.py",
        LOCAL_INFERENCE_ROOT / "module.py",
    ]
    return all(path.exists() for path in required)


def run_local_jepa(job_id: str, frames_dir: Path):
    result_path = UPLOAD_ROOT / job_id / "result.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LOCAL_INFERENCE_ROOT)

    printable = [
        "python3",
        str(ROOT / "demo_jepawm_predict.py"),
        "--checkpoint",
        str(LOCAL_CHECKPOINT),
        "--frames-dir",
        str(frames_dir),
        "--history-size",
        str(REMOTE_HISTORY_SIZE),
        "--device",
        "cpu",
        "--output",
        str(result_path),
    ]
    print("\n>>> local-inference", " ".join(shlex.quote(part) for part in printable), flush=True)
    proc = subprocess.run(
        printable,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=600,
        env=env,
        check=True,
    )
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", flush=True)
    result = json.loads(result_path.read_text())
    result["inferenceMode"] = "local"
    result["device"] = "cpu"
    return result


def ensure_remote_vm():
    describe_cmd = [
        "gcloud",
        "compute",
        "instances",
        "describe",
        REMOTE_VM_NAME,
        "--zone",
        REMOTE_VM_ZONE,
        "--project",
        REMOTE_PROJECT,
        "--format=json",
    ]
    try:
        output = run_logged(describe_cmd, timeout=120)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Remote JEPA VM is unavailable. "
            "If auth expired, run `gcloud auth login`. "
            f"If the instance does not exist, create `{REMOTE_VM_NAME}` in {REMOTE_VM_ZONE}."
        ) from exc

    payload = json.loads(output)
    status = payload.get("status")
    if status == "RUNNING":
        return

    start_cmd = [
        "gcloud",
        "compute",
        "instances",
        "start",
        REMOTE_VM_NAME,
        "--zone",
        REMOTE_VM_ZONE,
        "--project",
        REMOTE_PROJECT,
    ]
    run_logged(start_cmd, timeout=600)


def run_remote_jepa(job_id: str, frames_dir: Path):
    ensure_remote_vm()

    remote_job_root = f"{REMOTE_WORKDIR}/demo_api_jobs/{job_id}"
    remote_frames_dir = f"{remote_job_root}/frames"
    remote_result_path = f"{remote_job_root}/result.json"

    run_logged(
        [
            "gcloud",
            "compute",
            "ssh",
            REMOTE_VM_NAME,
            "--zone",
            REMOTE_VM_ZONE,
            "--project",
            REMOTE_PROJECT,
            "--command",
            f"mkdir -p {shlex.quote(remote_frames_dir)} {shlex.quote(REMOTE_WORKDIR)}",
        ],
        timeout=180,
    )

    run_logged(
        [
            "gcloud",
            "compute",
            "scp",
            str(ROOT / "demo_jepawm_predict.py"),
            f"{REMOTE_VM_NAME}:{REMOTE_WORKDIR}/demo_jepawm_predict.py",
            "--zone",
            REMOTE_VM_ZONE,
            "--project",
            REMOTE_PROJECT,
        ],
        timeout=180,
    )

    run_logged(
        [
            "gcloud",
            "compute",
            "scp",
            "--recurse",
            str(frames_dir),
            f"{REMOTE_VM_NAME}:{remote_job_root}",
            "--zone",
            REMOTE_VM_ZONE,
            "--project",
            REMOTE_PROJECT,
        ],
        timeout=300,
    )

    command = " && ".join(
        [
            f"cd {shlex.quote(REMOTE_WORKDIR)}",
            (
                "python3 demo_jepawm_predict.py "
                f"--checkpoint {shlex.quote(REMOTE_CHECKPOINT)} "
                f"--frames-dir {shlex.quote(remote_frames_dir)} "
                f"--history-size {REMOTE_HISTORY_SIZE} "
                f"--device {shlex.quote(REMOTE_DEVICE)} "
                f"--output {shlex.quote(remote_result_path)} "
                "> /dev/null"
            ),
            f"cat {shlex.quote(remote_result_path)}",
        ]
    )
    result_text = run_logged(
        [
            "gcloud",
            "compute",
            "ssh",
            REMOTE_VM_NAME,
            "--zone",
            REMOTE_VM_ZONE,
            "--project",
            REMOTE_PROJECT,
            "--command",
            command,
        ],
        timeout=600,
    )

    try:
        return extract_json_blob(result_text)
    finally:
        try:
            run_logged(
                [
                    "gcloud",
                    "compute",
                    "ssh",
                    REMOTE_VM_NAME,
                    "--zone",
                    REMOTE_VM_ZONE,
                    "--project",
                    REMOTE_PROJECT,
                    "--command",
                    f"rm -rf {shlex.quote(remote_job_root)}",
                ],
                timeout=120,
            )
        except Exception:
            pass


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in {"/", ""}:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/demo_ui/")
            self.end_headers()
            return

        if self.path == "/api/status":
            mode = "local" if has_local_inference() else "remote"
            return self._json(
                {
                    "sampleSheetPath": "/demo_cases/case_1/sheet.png",
                    "mode": mode,
                    "remoteVmName": REMOTE_VM_NAME,
                    "remoteVmZone": REMOTE_VM_ZONE,
                    "remoteCheckpoint": REMOTE_CHECKPOINT,
                    "localCheckpoint": str(LOCAL_CHECKPOINT),
                    "historySize": REMOTE_HISTORY_SIZE,
                    "device": "cpu" if mode == "local" else REMOTE_DEVICE,
                }
            )

        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/run-jepa-sheet":
            return self._json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return self._json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)

        try:
            image_b64 = payload["imageBase64"]
        except KeyError:
            return self._json({"error": "Missing imageBase64"}, status=HTTPStatus.BAD_REQUEST)

        filename = safe_name(payload.get("filename", "upload.png"))
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return self._json({"error": "Invalid base64 image data"}, status=HTTPStatus.BAD_REQUEST)

        job_id = time.strftime("%Y%m%d-%H%M%S")
        job_dir = UPLOAD_ROOT / job_id
        frames_dir = job_dir / "frames"
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / filename
        input_path.write_bytes(image_bytes)

        print(f"\n=== JEPA demo job {job_id} ===", flush=True)
        print(f"Input image: {input_path}", flush=True)

        try:
            manifest_text = run_logged(
                [
                    "python3",
                    str(ROOT / "demo_sheet_to_frames.py"),
                    "--sheet",
                    str(input_path),
                    "--output-dir",
                    str(frames_dir),
                ],
                cwd=ROOT,
                timeout=120,
            )
            manifest = extract_json_blob(manifest_text)
            if has_local_inference():
                result = run_local_jepa(job_id, frames_dir)
            else:
                result = run_remote_jepa(job_id, frames_dir)
            result["jobId"] = job_id
            result["sheet"] = manifest
            result["inputImage"] = str(input_path.resolve())
            return self._json(result)
        except subprocess.TimeoutExpired:
            return self._json({"error": "JEPA run timed out"}, status=HTTPStatus.GATEWAY_TIMEOUT)
        except RuntimeError as exc:
            return self._json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
        except subprocess.CalledProcessError as exc:
            return self._json(
                {
                    "error": "JEPA run failed",
                    "details": exc.output,
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main():
    parser = argparse.ArgumentParser(description="Serve the shell-game demo UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    url = f"http://{args.host}:{args.port}/demo_ui/"
    print(f"Serving demo UI at {url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
