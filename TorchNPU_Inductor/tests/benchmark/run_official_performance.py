import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from run_standard_accuracy import stop_on_signal, stop_server, wait_ready


def run_bench(vllm, args, common, name, log):
    for repeat in range(1, args.repeats + 1):
        filename = f"{name}_repeat{repeat}.json"
        command = [vllm, "bench", "serve", *common,
                   "--result-filename", filename]
        print("RUN", " ".join(command), flush=True)
        with open(log, "a", encoding="utf-8") as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                           cwd=args.out, check=True)


def main():
    signal.signal(signal.SIGTERM, stop_on_signal)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--track", choices=("default", "inductor"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--tokenizer-mode")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--qps", default="1,4,16,inf")
    parser.add_argument("--long-context", action="store_true")
    args = parser.parse_args()
    if args.port == 0:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            args.port = sock.getsockname()[1]
    args.out.mkdir(parents=True, exist_ok=False)

    from huggingface_hub import hf_hub_download
    sharegpt = hf_hub_download(
        repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
        repo_type="dataset",
        filename="ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    vllm = shutil.which("vllm")
    if not vllm:
        raise RuntimeError("vllm executable not found")

    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", str(args.tp), "--max-model-len", "40960",
               "--max-num-seqs", "32", "--gpu-memory-utilization", "0.80"]
    if args.enable_expert_parallel:
        command.append("--enable-expert-parallel")
    if args.tokenizer_mode:
        command.extend(["--tokenizer-mode", args.tokenizer_mode])
    if args.track == "inductor":
        command.extend(["--compilation-config", '{"backend":"inductor"}'])

    started = time.monotonic()
    produced = []
    with open(args.out / "server.log", "w", encoding="utf-8") as server_log:
        process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT,
                                   cwd=args.out, start_new_session=True)
        try:
            wait_ready(args.port, process)
            common = ["--model", args.model, "--backend", "vllm",
                      "--endpoint", "/v1/completions", "--host", "127.0.0.1", "--port", str(args.port),
                      "--percentile-metrics", "ttft,tpot,itl,e2el",
                      "--metric-percentiles", "50,95,99", "--seed", "0",
                      "--save-result", "--save-detailed", "--result-dir", str(args.out)]
            for qps in args.qps.split(","):
                name = f"sharegpt_qps_{qps}"
                run_bench(vllm, args, common + ["--dataset-name", "sharegpt", "--dataset-path", sharegpt,
                          "--num-prompts", "200", "--num-warmups", "10", "--request-rate", qps],
                          name, args.out / "bench.log")
                produced.append(name)
            if args.long_context:
                for input_len, prompts in ((8192, 32), (32768, 16)):
                    name = f"random_{input_len}_1024"
                    run_bench(vllm, args, common + ["--dataset-name", "random",
                              "--random-input-len", str(input_len), "--random-output-len", "1024",
                              "--random-range-ratio", "0", "--random-prefix-len", "0",
                              "--num-prompts", str(prompts), "--num-warmups", "4",
                              "--max-concurrency", "8", "--request-rate", "inf", "--ignore-eos"],
                              name, args.out / "bench.log")
                    produced.append(name)
        finally:
            stop_server(process)

    summary = {"model": args.model, "track": args.track, "loads": produced,
               "qps": args.qps, "repeats": args.repeats, "total_elapsed_s": time.monotonic() - started}
    (args.out / "performance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
