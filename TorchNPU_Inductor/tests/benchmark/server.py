import argparse
import json
import os
import shlex
import socket
import sys
import time
import urllib.request
from pathlib import Path


def config_value(value):
    if value is None:
        return None
    if hasattr(value, "name"):
        return value.name
    if hasattr(value, "value"):
        return value.value
    return value


def compilation_config_dict(vllm_config):
    config = vllm_config.compilation_config
    return {"backend": config_value(config.backend),
            "cudagraph_mode": config_value(config.cudagraph_mode),
            "level": config_value(getattr(config, "level", getattr(vllm_config, "optimization_level", None))),
            "mode": config_value(getattr(config, "mode", getattr(vllm_config, "performance_mode", None)))}


def record_server_config(command, metadata, requested):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    engine_parser = FlexibleArgumentParser(add_help=False)
    EngineArgs.add_cli_args(engine_parser)
    parsed, _ = engine_parser.parse_known_args(command[3:])
    engine_args = EngineArgs.from_cli_args(parsed)
    vllm_config = engine_args.create_engine_config()
    actual = compilation_config_dict(vllm_config)
    data = json.loads(metadata.read_text(encoding="utf-8"))
    data["requested_compilation_config"] = requested
    data["actual_compilation_config"] = actual
    metadata.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args, parser):
    if not 0 <= args.port <= 65535 or args.tp < 1:
        parser.error("--port must be in 0..65535 and --tp must be positive")
    if args.cudagraph_mode and args.track != "inductor":
        parser.error("--cudagraph-mode is only valid with --track inductor")
    if args.mtp and args.mtp_tokens < 1:
        parser.error("--mtp-tokens must be positive")
    port = args.port
    if port == 0:
        with socket.socket() as sock:
            sock.bind((args.host, 0))
            port = sock.getsockname()[1]
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", args.model, "--host", args.host, "--port", str(port),
               "--tensor-parallel-size", str(args.tp), "--max-model-len", str(args.max_model_len),
               "--max-num-seqs", str(args.max_num_seqs),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    requested_compilation = None
    if args.track == "eager":
        command.append("--enforce-eager")
    elif args.track == "inductor":
        compilation = {"backend": "inductor"}
        requested_compilation = compilation
        if args.cudagraph_mode:
            compilation["cudagraph_mode"] = args.cudagraph_mode
        command.extend(["--compilation-config", json.dumps(compilation, separators=(",", ":"))])
    if args.enable_expert_parallel:
        command.append("--enable-expert-parallel")
    if args.quantization:
        command.extend(["--quantization", args.quantization])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.tokenizer_mode:
        command.extend(["--tokenizer-mode", args.tokenizer_mode])
    if args.kv_cache_dtype:
        command.extend(["--kv-cache-dtype", args.kv_cache_dtype])
    if args.enable_prefix_caching is not None:
        command.append("--enable-prefix-caching" if args.enable_prefix_caching else "--no-enable-prefix-caching")
    if args.mtp:
        command.extend(["--speculative-config", json.dumps({
            "method": "mtp", "num_speculative_tokens": args.mtp_tokens}, separators=(",", ":"))])
    extra = list(args.vllm_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    command.extend(extra)
    if args.dry_run:
        print(json.dumps({"host": args.host, "port": port, "command": command}, ensure_ascii=False))
        return
    if args.metadata:
        if args.metadata.exists():
            parser.error(f"metadata file already exists: {args.metadata}")
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps({
            "model": args.model, "track": args.track, "host": args.host, "port": port,
            "pid": os.getpid(), "command": command}, ensure_ascii=False, indent=2), encoding="utf-8")
        record_server_config(command, args.metadata, requested_compilation)
    print("RUN " + shlex.join(command), flush=True)
    os.execv(sys.executable, command)


def process_running(pid):
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        fields = stat.read_text(encoding="utf-8").rsplit(")", 1)[-1].split()
        return bool(fields) and fields[0] != "Z"
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait(args, parser):
    if args.timeout <= 0 or args.interval <= 0:
        parser.error("--timeout and --interval must be positive")
    url = args.base_url.rstrip("/") + "/v1/models"
    started = time.monotonic()
    deadline = started + args.timeout
    while time.monotonic() < deadline:
        if args.server_pid and not process_running(args.server_pid):
            raise SystemExit(f"server process {args.server_pid} exited before becoming ready")
        try:
            with urllib.request.urlopen(url, timeout=min(5, args.interval)) as response:
                if response.status == 200:
                    print(json.dumps({"ready": True, "url": url,
                                      "elapsed_s": time.monotonic() - started}), flush=True)
                    return
        except Exception:
            pass
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
    raise SystemExit(f"server readiness timeout after {args.timeout}s: {url}")


def main():
    parser = argparse.ArgumentParser(description="Run or wait for an OpenAI-compatible vLLM server.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run one foreground server.")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--track", choices=("eager", "default", "inductor"), required=True)
    run_parser.add_argument("--host", default="127.0.0.1")
    run_parser.add_argument("--port", type=int, required=True)
    run_parser.add_argument("--tp", type=int, default=1)
    run_parser.add_argument("--max-model-len", type=int, default=40960)
    run_parser.add_argument("--max-num-seqs", type=int, default=32)
    run_parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    run_parser.add_argument("--cudagraph-mode")
    run_parser.add_argument("--quantization")
    run_parser.add_argument("--trust-remote-code", action="store_true")
    run_parser.add_argument("--tokenizer-mode")
    run_parser.add_argument("--kv-cache-dtype")
    run_parser.add_argument("--enable-expert-parallel", action="store_true")
    run_parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--mtp", action="store_true")
    run_parser.add_argument("--mtp-tokens", type=int, default=1)
    run_parser.add_argument("--metadata", type=Path)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("vllm_args", nargs=argparse.REMAINDER,
                            help="Additional vLLM arguments after --.")
    run_parser.set_defaults(handler=run)
    wait_parser = subparsers.add_parser("wait", help="Wait for HTTP readiness without inspecting logs.")
    wait_parser.add_argument("--base-url", required=True)
    wait_parser.add_argument("--timeout", type=float, default=1800)
    wait_parser.add_argument("--interval", type=float, default=5)
    wait_parser.add_argument("--server-pid", type=int)
    wait_parser.set_defaults(handler=wait)
    args = parser.parse_args()
    args.handler(args, parser)


if __name__ == "__main__":
    main()
