import argparse
import huggingface_hub.constants
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from process_supervisor import stop_process_group

CASES = {
    "mmlu": {
        "case_type": "accuracy",
        "dataset_path": "vllm-ascend/mmlu",
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "mmlu/mmlu_200_custom",
        "max_out_len": 2048,
        "batch_size": 32,
        "temperature": 0,
    },
    "gsm8k": {
        "case_type": "accuracy",
        "dataset_path": "vllm-ascend/gsm8k-lite",
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_chat_prompt",
        "max_out_len": 2048,
        "batch_size": 32,
        "temperature": 0,
    },
}


def wait_ready(port, process, timeout=1800):
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise TimeoutError("vLLM server readiness timeout")


def stop_server(process):
    stop_process_group(process, grace_s=30)


def stop_on_signal(signum, frame):
    raise SystemExit(128 + signum)


def main():
    signal.signal(signal.SIGTERM, stop_on_signal)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--track", choices=("eager", "inductor"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--tokenizer-mode")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--tasks", default="mmlu,gsm8k")
    args = parser.parse_args()
    if args.port == 0:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            args.port = sock.getsockname()[1]

    args.out.mkdir(parents=True, exist_ok=False)
    os.chdir(args.out)
    workspace = os.environ["WORKSPACE"]
    os.environ["BENCHMARK_HOME"] = str(Path(workspace) / "accuracy" / "benchmark")
    sys.path.insert(0, os.environ.get(
        "VLLM_ASCEND_SOURCE", str(Path(workspace) / "pin" / "inductor-meta-worktree" / "vllm-ascend")
    ))
    from tools.aisbench import AisbenchRunner

    config_dir = Path(os.environ["BENCHMARK_HOME"]) / "ais_bench" / "benchmark" / "configs" / "datasets" / "mmlu"
    custom_config = config_dir / "mmlu_200_custom.py"
    source_config = (config_dir / "mmlu_gen_0_shot_cot_chat_prompt.py").read_text(encoding="utf-8")
    source_config += "\nmmlu_datasets = [d for d in mmlu_datasets if d['name'] == 'miscellaneous']\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=config_dir, delete=False) as config:
        config.write(source_config)
        temporary_config = config.name
    os.replace(temporary_config, custom_config)

    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", str(args.tp), "--max-model-len", "40960",
               "--max-num-seqs", "32", "--gpu-memory-utilization", "0.80"]
    if args.enable_expert_parallel:
        command.append("--enable-expert-parallel")
    if args.tokenizer_mode:
        command.extend(["--tokenizer-mode", args.tokenizer_mode])
    if args.track == "eager":
        command.append("--enforce-eager")
    else:
        command.extend(["--compilation-config", '{"backend":"inductor"}'])

    results = {}
    started = time.monotonic()
    with open(args.out / "server.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=args.out, start_new_session=True)
        try:
            wait_ready(args.port, process)
            results["server_ready_s"] = time.monotonic() - started
            for name in args.tasks.split(","):
                case_started = time.monotonic()
                case = dict(CASES[name], model_path=args.model, num_prompts=args.num_prompts)
                if name == "mmlu":
                    case["dataset_path_local"] = os.environ.get(
                        "MMLU_DATASET_PATH", str(Path(workspace) / "accuracy" / "datasets" / "mmlu" / "mmlu"))
                elif name == "gsm8k":
                    case["dataset_path_local"] = os.environ.get(
                        "GSM8K_DATASET_PATH", str(Path(workspace) / "accuracy" / "datasets" / "gsm8k" / "gsm8k"))
                with AisbenchRunner(model=args.model, port=args.port,
                                    aisbench_config=case, verify=False) as bench:
                    if not hasattr(bench, "result"):
                        bench._get_result_accuracy()
                    result = bench.result
                results[name] = {"score": result, "num_prompts": args.num_prompts,
                                 "max_out_len": case["max_out_len"],
                                 "elapsed_s": time.monotonic() - case_started}
                print(f"ACCURACY {args.track} {name} {json.dumps(results[name])}", flush=True)
        finally:
            stop_server(process)
    results["track"] = args.track
    results["model"] = args.model
    results["total_elapsed_s"] = time.monotonic() - started
    (args.out / "accuracy_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
