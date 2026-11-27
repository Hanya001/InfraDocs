import argparse
import hashlib
import importlib.metadata
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PACKAGES = ("torch", "torch-npu", "triton-ascend", "vllm", "vllm-ascend")
ENVIRONMENT = (
    "ASCEND_RT_VISIBLE_DEVICES", "TORCHINDUCTOR_NPU_BACKEND", "VLLM_USE_V2_MODEL_RUNNER",
    "VLLM_USE_BREAKABLE_CUDAGRAPH", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR", "TORCH_COMPILE_DEBUG_DIR", "VLLM_WORKER_MULTIPROC_METHOD",
)
FATAL_PATTERNS = ("EngineCore failed to start",)


def inspect(args, parser):
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    errors = []
    config_path = args.model / "config.json"
    config = {}
    if not config_path.is_file():
        errors.append("config.json is missing")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            errors.append(f"config.json is invalid: {error}")
    shards = list(args.model.glob("*.safetensors"))
    if not shards:
        errors.append("no safetensors files found")
    indexes = list(args.model.glob("*.safetensors.index.json"))
    if len(shards) > 1 and not indexes:
        errors.append("multiple shards found without a safetensors index")
    missing_shards = []
    for index in indexes:
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            names = set(data.get("weight_map", {}).values())
            missing_shards.extend(name for name in sorted(names) if not (args.model / name).is_file())
        except (json.JSONDecodeError, TypeError) as error:
            errors.append(f"invalid index {index.name}: {error}")
    if missing_shards:
        errors.append(f"index references {len(missing_shards)} missing shards")
    if args.expected_name and args.model.name != args.expected_name:
        errors.append(f"directory name {args.model.name!r} does not match {args.expected_name!r}")
    report = {"model": str(args.model), "expected_name": args.expected_name,
              "valid": not errors, "errors": errors, "architectures": config.get("architectures"),
              "model_type": config.get("model_type"),
              "quantization_fields": {key: value for key, value in config.items() if "quant" in key.lower()},
              "safetensors_files": len(shards), "index_files": [path.name for path in indexes],
              "missing_shards": missing_shards}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WEIGHTS " + json.dumps(report, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


def git_info(path):
    def git(*arguments):
        return subprocess.run(["git", "-C", str(path), *arguments], check=False,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace")
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    return {"path": str(path), "head": head.stdout.strip() if head.returncode == 0 else None,
            "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            "error": (head.stderr or dirty.stderr).strip() or None}


def file_hash(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(args, parser):
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    repositories = []
    for name in (".", "pytorch", "vllm-ascend", "triton-ascend"):
        path = args.workspace / name
        if (path / ".git").exists():
            repositories.append(git_info(path))
    report = {"time": datetime.now().astimezone().isoformat(), "python": sys.executable,
              "python_version": sys.version, "workspace": str(args.workspace), "packages": packages,
              "repositories": repositories, "environment": {name: os.environ.get(name) for name in ENVIRONMENT},
              "env_sha256": file_hash(args.workspace / "env.sh")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ENVIRONMENT_RECORDED {args.out}", flush=True)


def evidence(args, parser):
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    log_texts = {str(path): path.read_text(encoding="utf-8", errors="replace") for path in args.log}
    combined = "\n".join(log_texts.values())
    backend_in_config = False
    actual_config = None
    if args.run_config:
        config = json.loads(args.run_config.read_text(encoding="utf-8"))
        actual_config = config.get("actual_compilation_config")
        requested = config.get("requested_compilation_config")
        if requested is not None:
            backend_in_config = requested.get("backend") == "inductor"
        else:
            engine = config.get("engine_kwargs", config)
            backend_in_config = engine.get("compilation_config", {}).get("backend") == "inductor"
    artifacts = []
    for root in [*args.cache_dir, *args.artifact_dir]:
        for pattern in ("output_code.py", "*.npubin", "*.ttir", "*.ttadapter"):
            artifacts.extend(str(path) for path in root.rglob(pattern) if path.is_file())
    result = {
        "backend_in_config": backend_in_config,
        "actual_compilation_config": actual_config,
        "backend_in_log": bool(re.search(r"backend[=:\" ]+inductor", combined, re.IGNORECASE)),
        "artifacts": artifacts,
        "cudagraph_modes": sorted(set(re.findall(r"cudagraph_mode[=:\" ]+([A-Z_]+)", combined))),
        "capture_log_lines": len(re.findall(r"(?i)cudagraph.*captur|captur.*cudagraph", combined)),
        "replay_log_lines": len(re.findall(r"(?i)cudagraph.*replay|replay.*cudagraph", combined)),
    }
    actual_backend = actual_config.get("backend") == "inductor" if actual_config else False
    result["has_inductor_evidence"] = bool(actual_backend or result["backend_in_log"] or artifacts)
    report = {"comparison_type": "inductor_evidence", "logs": list(log_texts),
              "cache_dirs": [str(path) for path in args.cache_dir],
              "artifact_dirs": [str(path) for path in args.artifact_dir],
              "run_config": str(args.run_config) if args.run_config else None, "evidence": result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EVIDENCE " + json.dumps(result, ensure_ascii=False), flush=True)
    if args.require_inductor and not result["has_inductor_evidence"]:
        raise SystemExit(1)


def process_group_exists(process):
    pgid = process.pid
    if sys.platform.startswith("linux") and os.path.isdir("/proc"):
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                stat = Path(f"/proc/{name}/stat").read_text(encoding="utf-8")
                fields = stat[stat.rfind(")") + 2:].split()
                if len(fields) >= 3 and int(fields[2]) == pgid and fields[0] != "Z":
                    return True
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
                continue
        return False
    if os.name == "nt":
        return process.poll() is None
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_process(process, grace_s=30):
    if not process_group_exists(process):
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and process_group_exists(process):
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass
    if process_group_exists(process):
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def read_output(stream, events):
    try:
        for line in iter(stream.readline, ""):
            events.put(("line", line))
    finally:
        events.put(("eof", ""))


def execute(command, timeout_s, environment):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1,
                               env=environment, start_new_session=os.name != "nt")
    parent_pid = os.getppid()
    events = queue.Queue()
    threading.Thread(target=read_output, args=(process.stdout, events), daemon=True).start()
    received_signal = []

    def on_signal(signum, frame):
        received_signal.append(signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    deadline = time.monotonic() + timeout_s
    return_code = None
    reason = None
    grace_s = 30
    main_exit_at = None
    eof = False
    while return_code is None:
        if received_signal:
            return_code, reason, grace_s = 128 + received_signal[0], f"received signal {received_signal[0]}", 5
            break
        if os.name != "nt" and os.getppid() != parent_pid:
            return_code, reason, grace_s = 125, "parent process exited", 5
            break
        if time.monotonic() >= deadline:
            return_code, reason = 124, f"total timeout after {timeout_s}s"
            break
        child_return_code = process.poll()
        if child_return_code is not None:
            if main_exit_at is None:
                main_exit_at = time.monotonic()
            if eof or time.monotonic() - main_exit_at >= 2:
                return_code = child_return_code
                if process_group_exists(process):
                    reason, grace_s = "main process exited with residual children", 5
                    break
                return return_code
        try:
            kind, line = events.get(timeout=0.05 if main_exit_at is not None else 0.2)
        except queue.Empty:
            continue
        if kind == "eof":
            eof = True
            continue
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except BrokenPipeError:
            return_code, reason, grace_s = 125, "output pipe closed", 5
            break
        matched = next((pattern for pattern in FATAL_PATTERNS if pattern in line), None)
        if matched:
            return_code, reason, grace_s = 1, f"fatal engine event: {matched}", 10
            break
    print(f"SUPERVISOR_STOP reason={reason} pgid={process.pid}", flush=True)
    stop_process(process, grace_s=grace_s)
    if reason.startswith("fatal engine event") and process.returncode not in (None, 0):
        return process.returncode if process.returncode > 0 else 128 - process.returncode
    return return_code


def parse_devices(value, parser):
    if not value:
        return []
    try:
        devices = sorted({int(item) for item in value.split(",")})
    except ValueError:
        parser.error("--devices must be a comma-separated list of integers")
    if not devices or any(device < 0 for device in devices):
        parser.error("--devices must contain non-negative device IDs")
    return devices


def busy_devices(output, devices, max_used_mb):
    busy = set()
    selected = {str(device) for device in devices}
    current_card = None
    for line in output.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 2:
            continue
        first = fields[1].split()
        if first and first[0].isdigit():
            current_card = first[0]
        if current_card not in selected:
            continue
        if len(fields) >= 5 and fields[2].isdigit() and fields[3].isdigit():
            busy.add(int(current_card))
        for field in fields:
            match = re.fullmatch(r"(\d+)\s*/\s*\d+(?:\s*MB)?", field)
            if match and int(match.group(1)) > max_used_mb:
                busy.add(int(current_card))
    return sorted(busy)


def check_devices(args, parser):
    devices = parse_devices(args.devices, parser)
    if not devices:
        parser.error("--devices cannot be empty")
    if args.max_used_mb < 0:
        parser.error("--max-used-mb cannot be negative")
    try:
        completed = subprocess.run(["npu-smi", "info"], check=False,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
    except OSError as error:
        raise SystemExit(f"failed to execute npu-smi info: {error}")
    if completed.returncode != 0:
        raise SystemExit(f"npu-smi info failed: {completed.stdout.strip()}")
    busy = busy_devices(completed.stdout, devices, args.max_used_mb)
    if busy:
        raise SystemExit("NPU devices unavailable or not idle: " + ",".join(map(str, busy)))
    print("DEVICES_IDLE " + ",".join(map(str, devices)), flush=True)


def execute_with_locks(args, command, environment, devices):
    locks = []
    if devices:
        try:
            import fcntl
        except ImportError:
            raise SystemExit("device locking requires Linux fcntl")
        args.lock_dir.mkdir(parents=True, exist_ok=True)
    try:
        for device in devices:
            lock = (args.lock_dir / f"npu_{device}.lock").open("w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                raise SystemExit(f"NPU device {device} is already locked")
            locks.append(lock)
        if devices:
            environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(device) for device in devices)
            print(f"DEVICES_LOCKED {environment['ASCEND_RT_VISIBLE_DEVICES']}", flush=True)
        return execute(command, args.timeout, environment)
    finally:
        for lock in locks:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


def exec_command(args, parser):
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.cache_dir.exists() and not args.cache_dir.is_dir():
        parser.error(f"cache path is not a directory: {args.cache_dir}")
    devices = parse_devices(args.devices, parser)
    if devices and not args.lock_dir:
        parser.error("--lock-dir is required with --devices")
    cache_state = "warm" if args.cache_dir.exists() and any(args.cache_dir.iterdir()) else "cold"
    cache_dir = args.cache_dir.resolve()
    paths = {"VLLM_CACHE_ROOT": cache_dir / "vllm_cache",
             "TORCHINDUCTOR_CACHE_DIR": cache_dir / "inductor",
             "TRITON_CACHE_DIR": cache_dir / "triton"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    if args.compile_debug_dir:
        paths["TORCH_COMPILE_DEBUG_DIR"] = args.compile_debug_dir.resolve()
        paths["TORCH_COMPILE_DEBUG_DIR"].mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({name: str(path) for name, path in paths.items()})
    run_command = command
    if args.workspace:
        workspace = args.workspace.resolve()
        # A workspace env.sh may export cache variables unconditionally, so re-apply
        # the per-run cache paths after sourcing it; otherwise runs share caches.
        exports = " ".join(f"export {name}={shlex.quote(str(path))}" for name, path in paths.items())
        run_command = ["bash", "-lc",
                       f"cd {shlex.quote(str(workspace))} && source ./env.sh && {exports} "
                       f"&& exec {shlex.join(command)}"]
    state = {"cache_dir": str(cache_dir), "cache_state": cache_state,
             "cache_paths": {name: str(path) for name, path in paths.items()},
             "workspace": str(args.workspace.resolve()) if args.workspace else None,
             "devices": devices, "timeout_s": args.timeout, "command": command, "return_code": None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUNTIME " + json.dumps(state, ensure_ascii=False), flush=True)
    try:
        return_code = execute_with_locks(args, run_command, environment, devices)
    except BaseException:
        state["return_code"] = 130
        args.out.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    state["return_code"] = return_code
    args.out.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(return_code)


def main():
    parser = argparse.ArgumentParser(description="Prepare runtime context or execute one supervised command.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    device_parser = subparsers.add_parser("check-devices", help="Check that selected NPUs have no processes or significant HBM use.")
    device_parser.add_argument("--devices", required=True)
    device_parser.add_argument("--max-used-mb", type=int, default=8192)
    device_parser.set_defaults(handler=check_devices)
    exec_parser = subparsers.add_parser("exec", help="Run a command with caches, optional device locks and supervision.")
    exec_parser.add_argument("--cache-dir", type=Path, required=True)
    exec_parser.add_argument("--out", type=Path, required=True)
    exec_parser.add_argument("--workspace", type=Path)
    exec_parser.add_argument("--compile-debug-dir", type=Path)
    exec_parser.add_argument("--devices")
    exec_parser.add_argument("--lock-dir", type=Path)
    exec_parser.add_argument("--timeout", type=int, default=3600)
    exec_parser.add_argument("command", nargs=argparse.REMAINDER)
    exec_parser.set_defaults(handler=exec_command)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect model weights and quantization fields.")
    inspect_parser.add_argument("--model", type=Path, required=True)
    inspect_parser.add_argument("--expected-name")
    inspect_parser.add_argument("--out", type=Path, required=True)
    inspect_parser.set_defaults(handler=inspect)
    snapshot_parser = subparsers.add_parser("snapshot", help="Record software and runtime environment.")
    snapshot_parser.add_argument("--workspace", type=Path, required=True)
    snapshot_parser.add_argument("--out", type=Path, required=True)
    snapshot_parser.set_defaults(handler=snapshot)
    evidence_parser = subparsers.add_parser("evidence", help="Collect Inductor path evidence.")
    evidence_parser.add_argument("--log", type=Path, action="append", default=[])
    evidence_parser.add_argument("--cache-dir", type=Path, action="append", default=[])
    evidence_parser.add_argument("--artifact-dir", type=Path, action="append", default=[])
    evidence_parser.add_argument("--run-config", type=Path)
    evidence_parser.add_argument("--out", type=Path, required=True)
    evidence_parser.add_argument("--require-inductor", action="store_true")
    evidence_parser.set_defaults(handler=evidence)
    args = parser.parse_args()
    args.handler(args, parser)


if __name__ == "__main__":
    main()
