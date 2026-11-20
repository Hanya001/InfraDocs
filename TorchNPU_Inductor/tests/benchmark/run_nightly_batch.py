import argparse
import concurrent.futures
import json
import os
import posixpath
import shlex
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import dataclass

os.environ.setdefault("MSYS_NO_PATHCONV", "1")
os.environ.setdefault("MSYS2_ARG_CONV_EXCL", "*")
from datetime import datetime
from pathlib import Path

HOST = os.environ.get("BENCH_HOST", "a5-25")
CONTAINER = os.environ.get("BENCH_CONTAINER", "inductor_a51112_pin_z00896543_20261116_195508")
BASE = os.environ.get("BENCH_BASE", "/home/z00896543/validation/a51112_20261116_195508")
WORKSPACE = os.environ.get("BENCH_WORKSPACE", f"{BASE}/pin/inductor-meta-worktree")
SCRIPTS = os.environ.get("BENCH_SCRIPTS", f"{BASE}/scripts/bench213_matrix")


@dataclass(frozen=True)
class Model:
    name: str
    weights: str
    tp: int = 1
    nightly: bool = False
    long_context: bool = False
    kind: str = "text"
    tokenizer_mode: str = ""
    enable_ep: bool = False


MODELS = [
    Model("Qwen3-0.6B", "/mnt/weight/Qwen/Qwen3-0.6B"),
    Model("Qwen3-0.6B-W8A8", "/mnt/weight/Qwen3-0.6B-W8A8", nightly=True),
    Model("Qwen3-Embedding-0.6B", "/mnt/weight/Qwen3-Embedding-0.6B", kind="embedding"),
    Model("Qwen3.5-0.8B", "/mnt/weight/Qwen3.5-0.8B", nightly=True),
    Model("Qwen2.5-VL-7B-Instruct", "/mnt/weight/Qwen2.5-VL-7B-Instruct", kind="multimodal"),
    Model("Qwen3-8B-W8A8", "/mnt/weight/Qwen3-8B-W8A8", nightly=True),
    Model("DeepSeek-V2-Lite-W8A8", "/mnt/weight/DeepSeek-V2-Lite-W8A8"),
    Model("DeepSeek-V2-Lite", "/mnt/weight/DeepSeek-V2-Lite"),
    Model("Qwen3.8-27B-w8a8", "/mnt/weight/Qwen3.8-27B-w8a8", tp=2, nightly=True, long_context=True),
    Model("Qwen3.8-27B-w8a8-mxfp8", "/mnt/weight/weights/Qwen3.8-27B-w8a8-mxfp8", tp=2),
    Model("Qwen3.8-27B-FP8", "/mnt/weight/Qwen3.8-27B-FP8", tp=2, nightly=True),
    Model("Qwen3-30B-A3B-W8A8", "/mnt/weight/Qwen3-30B-A3B-W8A8", tp=2),
    Model("Qwen3.8-27B", "/mnt/weight/Qwen3.8-27B", tp=2, nightly=True, long_context=True),
    Model("Qwen3.6-35B-A3B-w4a8-mxfp", "/mnt/weight/Qwen3.6-35B-A3B-w4a8-mxfp", tp=2, nightly=True, long_context=True),
    Model("Qwen3.6-35B-A3B-w8a8", "/mnt/weight/Qwen3.6-35B-A3B-w8a8", tp=2, nightly=True, long_context=True),
    Model("Qwen3.6-35B-A3B-w8a8-mxfp8", "/mnt/weight/Qwen3.6-35B-A3B-w8a8-mxfp8", tp=2, nightly=True, long_context=True),
    Model("Qwen3.6-35B-A3B", "/mnt/weight/Qwen3.6-35B-A3B", tp=2, nightly=True, long_context=True),
    Model("Kimi-K3-expert16-top8-100B-CountBenchQA", "/mnt/weight/Kimi-K3-expert16-top8-100B-CountBenchQA", tp=4, kind="multimodal"),
    Model("Qwen3.8-Flash-Next", "/mnt/weight/Qwen3.8-Flash-Next", tp=8, kind="multimodal"),
    Model("GLM-5.3-Flash-w8a8", "/mnt/weight/GLM-5.3-Flash-w8a8", tp=8, kind="multimodal"),
    Model("GLM-5.2-w4a8c8", "/mnt/weight/GLM-5.2-w4a8c8", tp=8),
    Model("DeepSeek-V4.1-Flash", "/mnt/weight/DeepSeek-V4.1-Flash-1017", tp=8, tokenizer_mode="deepseek_v4", enable_ep=True),
]


def quote(value):
    return shlex.quote(str(value))


class Runner:
    def __init__(self, args):
        self.args = args
        self.deadline = datetime.fromisoformat(args.deadline).timestamp()
        self.correctness_deadline = datetime.fromisoformat(args.correctness_deadline).timestamp()
        self.remote_root = f"{BASE}/results/{args.run_id}"
        self.local_root = args.local_root
        self.local_root.mkdir(parents=True, exist_ok=True)
        (self.local_root / "_driver").mkdir(exist_ok=True)
        (self.local_root / "status").mkdir(exist_ok=True)
        self.device_condition = threading.Condition()
        self.free_devices = {int(value) for value in args.devices.split(",")}
        self.thread_devices = threading.local()

    def remaining(self):
        return max(0, int(self.deadline - time.time()))

    def devices(self, model):
        return getattr(self.thread_devices, "value", None) or ",".join(
            str(index) for index in range(model.tp))

    def acquire_devices(self, model, deadline):
        with self.device_condition:
            while len(self.free_devices) < model.tp:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                self.device_condition.wait(min(remaining, 5))
            selected = sorted(self.free_devices)[:model.tp]
            self.free_devices.difference_update(selected)
            self.thread_devices.value = ",".join(str(index) for index in selected)
            print(f"DEVICES_ACQUIRED {model.name} devices={self.thread_devices.value}", flush=True)
            return True

    def release_devices(self, model):
        with self.device_condition:
            selected = {int(value) for value in self.thread_devices.value.split(",")}
            self.free_devices.update(selected)
            print(f"DEVICES_RELEASED {model.name} devices={self.thread_devices.value}", flush=True)
            self.thread_devices.value = None
            self.device_condition.notify_all()

    def remote(self, command, timeout, log_name, deadline=None):
        seconds = min(self.remaining(), timeout)
        if deadline is not None:
            seconds = min(seconds, max(0, int(deadline - time.time())))
        if seconds < 30:
            return 124
        wrapped = f"timeout --signal=TERM --kill-after=60s {seconds}s bash -lc {quote(command)}"
        ssh_command = f"docker exec {CONTAINER} bash -lc {quote(wrapped)}"
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=3", HOST, ssh_command]
        log_path = self.local_root / "_driver" / log_name
        print(f"REMOTE_START {log_name} timeout_s={seconds}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(command + "\n")
            log.flush()
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                if any(marker in line for marker in (
                        "MODEL_EVENT", "BENCH_EVENT", "BENCH_HEARTBEAT", "ACCURACY ",
                        "REMOTE_ERROR", "Traceback", "RuntimeError", "RUN vllm")):
                    print(line, end="", flush=True)
            return_code = process.wait()
        print(f"REMOTE_END {log_name} rc={return_code} remaining_s={self.remaining()}", flush=True)
        return return_code

    def host(self, command, timeout=300):
        return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, command],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout)

    def weights_ready(self, model):
        code = r'''import glob, json, os, sys
path = sys.argv[1]
files = glob.glob(os.path.join(path, "*.safetensors"))
if not os.path.isfile(os.path.join(path, "config.json")) or not files:
    raise SystemExit(1)
indexes = glob.glob(os.path.join(path, "*.safetensors.index.json"))
if os.path.basename(path) == "Kimi-K3-expert16-top8-100B-CountBenchQA":
    indexes = [os.path.join(path, "model.safetensors.index.json")]
if len(files) > 1 and not indexes:
    raise SystemExit(2)
for index in indexes:
    data = json.load(open(index, encoding="utf-8"))
    missing = [name for name in set(data.get("weight_map", {}).values())
               if not os.path.isfile(os.path.join(path, name))]
    if missing:
        print("missing shards:", missing[:10], flush=True)
        raise SystemExit(3)
'''
        result = self.host(f"docker exec {CONTAINER} python3 -c {quote(code)} {quote(model.weights)}")
        if result.returncode != 0:
            print(f"WEIGHTS_NOT_READY {model.name} rc={result.returncode} {result.stdout.strip()}", flush=True)
        return result.returncode == 0

    def correctness_dir(self, model):
        if model.name == "Qwen3.5-0.8B":
            return f"{self.remote_root}/correctness/Qwen3.5-0.8B/Qwen3.5-0.8B"
        return f"{self.remote_root}/correctness/{model.name}"

    def environment_prefix(self, model, track, out, cache):
        parent = posixpath.dirname(out)
        pieces = [
            f"cd {quote(WORKSPACE)}", "source env.sh", f"export WORKSPACE={quote(BASE)}",
            f"export VLLM_ASCEND_SOURCE={quote(WORKSPACE + '/vllm-ascend')}",
            "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
            "export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCHINDUCTOR_AUTOGRAD_CACHE=0",
            "export VLLM_USE_V2_MODEL_RUNNER=1 PYTHONUNBUFFERED=1",
            f"export ASCEND_RT_VISIBLE_DEVICES={self.devices(model)}",
            f"export VLLM_CACHE_ROOT={cache}/vllm_cache",
            f"export TORCHINDUCTOR_CACHE_DIR={cache}/inductor",
            f"export TRITON_CACHE_DIR={cache}/triton",
            f"export TORCH_COMPILE_DEBUG_DIR={out}.compile_debug",
            "export TORCH_COMPILE_DEBUG=0",
            f"mkdir -p {quote(parent)} \"$VLLM_CACHE_ROOT\" \"$TORCHINDUCTOR_CACHE_DIR\" \"$TRITON_CACHE_DIR\"",
        ]
        if model.name in ("DeepSeek-V4.1-Flash", "Kimi-K3-expert16-top8-100B-CountBenchQA"):
            pieces.append("export VLLM_USE_BREAKABLE_CUDAGRAPH=0")
        manifest = (
            f"{{ date -Is; printf 'model={model.name}\\ntrack={track}\\ntp={model.tp}\\n"
            f"devices={self.devices(model)}\\nmodel_runner=MRV2\\ncache={cache}\\n'; "
            "git rev-parse HEAD; git -C pytorch rev-parse HEAD; git -C vllm-ascend rev-parse HEAD; "
            "git -C triton-ascend rev-parse HEAD; "
            "\"$PY\" -I -c 'import importlib.metadata as m; print({p:m.version(p) for p in [\"torch\",\"torch-npu\",\"triton-ascend\",\"vllm\",\"vllm-ascend\"]})'; "
            f"}} > {quote(out + '.environment.txt')} 2>&1"
        )
        pieces.append(manifest)
        return " && ".join(pieces)

    def run_correctness(self, model):
        if model.name == "Qwen3.5-0.8B":
            command = f"test -f {quote(self.correctness_dir(model) + '/inductor/correctness_complete.json')}"
            result = self.host(f"docker exec {CONTAINER} bash -lc {quote(command)}")
            if result.returncode == 0:
                print("CORRECTNESS_REUSE Qwen3.5-0.8B", flush=True)
                return 0
        tracks = "eager default inductor" if model.kind == "embedding" else "eager inductor"
        root = f"{self.remote_root}/correctness"
        if model.name == "Qwen3.5-0.8B":
            root = f"{self.remote_root}/correctness/Qwen3.5-0.8B"
        extra = "VLLM_USE_BREAKABLE_CUDAGRAPH=0" if model.name in (
            "DeepSeek-V4.1-Flash", "Kimi-K3-expert16-top8-100B-CountBenchQA") else ""
        command = (f"cd {quote(SCRIPTS)} && BENCH_BASE={quote(BASE)} BENCH_WORKSPACE={quote(WORKSPACE)} "
                   f"MODEL_SET=20261118 BENCH_DEVICES={self.devices(model)} "
                   f"VLLM_USE_V2_MODEL_RUNNER=1 {extra} TRACKS={quote(tracks)} "
                   f"bash run_a5.sh {quote(root)} {quote(model.name)}")
        return self.remote(command, 5400, f"{model.name}.correctness.log", self.correctness_deadline)

    def run_accuracy(self, model, track):
        out = f"{self.remote_root}/accuracy/{model.name}/{track}"
        cache = f"{self.correctness_dir(model)}/{track}/runtime/log"
        command = self.environment_prefix(model, track, out, cache)
        command += (f" && \"$PY\" -u {quote(SCRIPTS + '/run_standard_accuracy.py')} --model {quote(model.weights)}"
                    f" --track {track} --out {quote(out)} --port 0 --tp {model.tp} --num-prompts 200")
        if model.enable_ep:
            command += " --enable-expert-parallel"
        if model.tokenizer_mode:
            command += f" --tokenizer-mode {quote(model.tokenizer_mode)}"
        return self.remote(command, 5400, f"{model.name}.accuracy.{track}.log", self.correctness_deadline)

    def run_performance(self, model, track):
        out = f"{self.remote_root}/performance/{model.name}/{track}"
        if track == "inductor":
            cache = f"{self.correctness_dir(model)}/inductor/runtime/log"
        else:
            cache = f"{self.remote_root}/cache/{model.name}/default"
        command = self.environment_prefix(model, track, out, cache)
        command += (f" && \"$PY\" -u {quote(SCRIPTS + '/run_official_performance.py')} --model {quote(model.weights)}"
                    f" --track {track} --out {quote(out)} --port 0 --tp {model.tp} --repeats 3")
        if model.long_context:
            command += " --long-context"
        if model.enable_ep:
            command += " --enable-expert-parallel"
        if model.tokenizer_mode:
            command += f" --tokenizer-mode {quote(model.tokenizer_mode)}"
        return self.remote(command, 7200, f"{model.name}.performance.{track}.log")

    def save_status(self, model, status):
        path = self.local_root / "status" / f"{model.name}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def print_result(self, model):
        correctness = self.local_root / "correctness" / model.name
        if model.name == "Qwen3.5-0.8B":
            correctness = correctness / "Qwen3.5-0.8B"
        result = {"model": model.name}
        comparison = correctness / "compare.json"
        if comparison.is_file():
            data = json.loads(comparison.read_text(encoding="utf-8"))
            if "token_identical" in data:
                result["token_identical"] = f"{data['token_identical']}/{data['n_prompts']}"
                result["text_identical"] = f"{data['text_identical']}/{data['n_prompts']}"
            if "embedding_metrics" in data:
                result["embedding_max_abs_diff"] = max(x["max_abs_diff"] for x in data["embedding_metrics"])
                result["embedding_min_cosine"] = min(x["cosine"] for x in data["embedding_metrics"])
        for track in ("eager", "inductor"):
            path = self.local_root / "accuracy" / model.name / track / "accuracy_results.json"
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                for task in ("mmlu", "gsm8k"):
                    if task in data:
                        result[f"{task}_{track}"] = data[task]["score"]
        for rate in ("1", "4", "16", "inf"):
            values = {}
            for track in ("default", "inductor"):
                directory = self.local_root / "performance" / model.name / track
                measurements = []
                for path in directory.glob(f"sharegpt_qps_{rate}_repeat*.json"):
                    data = json.loads(path.read_text(encoding="utf-8"))
                    measurements.append(data["output_throughput"])
                if measurements:
                    values[track] = statistics.median(measurements)
                    result[f"qps{rate}_{track}_tok_s"] = round(values[track], 2)
            if len(values) == 2 and values["default"]:
                result[f"qps{rate}_speedup"] = round(values["inductor"] / values["default"], 4)
        print("MODEL_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)

    def sync(self, force=False):
        if self.args.sync_mode == "none" or (self.args.sync_mode == "final" and not force):
            return
        print(f"SYNC_START {self.local_root}", flush=True)
        for directory in ("correctness", "accuracy", "performance"):
            self.host(f"mkdir -p {self.remote_root}/{directory}")
        archive = f"{BASE}/results/{self.args.run_id}.sync.tar.gz"
        command = (f"tar -czf {archive} -C {self.remote_root} "
                   "--exclude='*/runtime' --exclude='*.pyc' correctness accuracy performance")
        result = self.host(command, timeout=600)
        if result.returncode != 0:
            print(f"SYNC_ERROR tar rc={result.returncode} {result.stdout.strip()}", flush=True)
            return
        local_archive = self.local_root / ".sync.tar.gz"
        result = subprocess.run(["scp", "-q", f"{HOST}:{archive}", str(local_archive)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", timeout=600)
        if result.returncode != 0:
            print(f"SYNC_ERROR scp rc={result.returncode} {result.stdout.strip()}", flush=True)
            return
        with tarfile.open(local_archive, "r:gz") as bundle:
            bundle.extractall(self.local_root, filter="data")
        local_archive.unlink()
        self.host(f"rm -f {archive}")
        subprocess.run([sys.executable, str(Path(__file__).with_name("summarize_nightly.py")),
                        str(self.local_root)], check=False)
        print(f"SYNC_COMPLETE {self.local_root}", flush=True)

    def load_status(self, model):
        path = self.local_root / "status" / f"{model.name}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"model": model.name, "weights": model.weights, "tp": model.tp,
                "model_runner": "MRV2", "optimization_level": "O2", "stages": {}}

    def run_model_correctness(self, model):
        status = self.load_status(model)
        self.save_status(model, status)
        print(f"CORRECTNESS_MODEL_START {model.name} tp={model.tp} remaining_s={self.remaining()}", flush=True)
        if min(self.remaining(), int(self.correctness_deadline - time.time())) < 30:
            status["stages"]["correctness"] = "skipped_correctness_deadline"
            self.save_status(model, status)
            return
        if not self.weights_ready(model):
            status["stages"]["weights"] = "skipped_missing_or_incomplete"
            self.save_status(model, status)
            return
        if not self.acquire_devices(model, self.correctness_deadline):
            status["stages"]["correctness"] = "skipped_correctness_deadline"
            self.save_status(model, status)
            return
        try:
            status["stages"]["correctness"] = self.run_correctness(model)
            self.save_status(model, status)
            if self.args.fail_fast and status["stages"]["correctness"] != 0:
                raise RuntimeError(f"Correctness failed for {model.name}: rc={status['stages']['correctness']}")
            if min(self.remaining(), int(self.correctness_deadline - time.time())) < 30:
                return
            if model.nightly and model.kind == "text":
                for track in ("eager", "inductor"):
                    status["stages"][f"accuracy_{track}"] = self.run_accuracy(model, track)
                    self.save_status(model, status)
                    if self.args.fail_fast and status["stages"][f"accuracy_{track}"] != 0:
                        raise RuntimeError(f"Accuracy {track} failed for {model.name}")
                    if min(self.remaining(), int(self.correctness_deadline - time.time())) < 30:
                        return
            print(f"CORRECTNESS_MODEL_END {model.name}", flush=True)
        finally:
            self.release_devices(model)

    def run_model_performance(self, model):
        if model.kind == "embedding":
            return
        status = self.load_status(model)
        if status["stages"].get("weights") == "skipped_missing_or_incomplete":
            return
        if self.remaining() < 30:
            status["stages"]["performance"] = "skipped_deadline"
            self.save_status(model, status)
            return
        if not self.acquire_devices(model, self.deadline):
            status["stages"]["performance"] = "skipped_deadline"
            self.save_status(model, status)
            return
        try:
            print(f"PERFORMANCE_MODEL_START {model.name} tp={model.tp} devices={self.devices(model)} remaining_s={self.remaining()}", flush=True)
            for track in ("default", "inductor"):
                status["stages"][f"performance_{track}"] = self.run_performance(model, track)
                self.save_status(model, status)
                if self.args.fail_fast and status["stages"][f"performance_{track}"] != 0:
                    raise RuntimeError(f"Performance {track} failed for {model.name}")
                if self.remaining() < 30:
                    return
            print(f"PERFORMANCE_MODEL_END {model.name}", flush=True)
        finally:
            self.release_devices(model)

    def run(self):
        for required in (f"{WORKSPACE}/env.sh", f"{SCRIPTS}/run_a5.sh"):
            result = self.host(f"docker exec {CONTAINER} test -f {required}")
            if result.returncode != 0:
                raise RuntimeError(f"Remote preflight failed for {CONTAINER}:{required}: {result.stdout.strip()}")
        print(f"PREFLIGHT_OK host={HOST} container={CONTAINER} workspace={WORKSPACE}", flush=True)
        selected = set(self.args.models.split(",")) if self.args.models else None
        models = [model for model in MODELS if not selected or model.name in selected]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.max_workers) as pool:
            futures = {pool.submit(self.run_model_correctness, model): model for model in models}
            for future in concurrent.futures.as_completed(futures):
                model = futures[future]
                future.result()
                self.sync()
                self.print_result(model)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.performance_workers) as pool:
            futures = {pool.submit(self.run_model_performance, model): model for model in models}
            for future in concurrent.futures.as_completed(futures):
                model = futures[future]
                future.result()
                self.sync()
                self.print_result(model)
        self.sync(force=True)
        print("BATCH_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20261118_nightly_all_01")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--deadline", default="2026-11-19T09:00:00+08:00")
    parser.add_argument("--correctness-deadline", default="2026-11-19T03:00:00+08:00")
    parser.add_argument("--models")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--performance-workers", type=int, default=1)
    parser.add_argument("--sync-mode", choices=("on_change", "final", "none"), default="final")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        for model in MODELS:
            print(model.name, model.weights, model.tp, "nightly" if model.nightly else "extended")
        return
    Runner(args).run()


if __name__ == "__main__":
    main()
