import argparse
import json
import shlex
import shutil
import subprocess
import time
import urllib.parse
from pathlib import Path


def run(args, parser):
    endpoint = urllib.parse.urlparse(args.base_url)
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname or not endpoint.port:
        parser.error("--base-url must include an http(s) host and port")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if args.dataset_name == "sharegpt" and not args.dataset_path:
        parser.error("--dataset-path is required for the sharegpt dataset")
    if args.dataset_name == "random" and not args.random_input_len:
        parser.error("--random-input-len is required for the random dataset")
    if args.num_prompts is not None and args.num_prompts < 1:
        parser.error("--num-prompts must be positive")
    if args.num_warmups is not None and args.num_warmups < 0:
        parser.error("--num-warmups cannot be negative")
    if args.max_concurrency is not None and args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    if args.random_input_len is not None and args.random_input_len < 1:
        parser.error("--random-input-len must be positive")
    if args.random_output_len < 1:
        parser.error("--random-output-len must be positive")
    if args.sharegpt_output_len < 1:
        parser.error("--sharegpt-output-len must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vllm = shutil.which("vllm")
    if not vllm:
        raise SystemExit("vllm executable not found")
    is_random = args.dataset_name == "random"
    max_concurrency = args.max_concurrency or (8 if is_random else None)
    if args.num_prompts is not None:
        num_prompts = args.num_prompts
    else:
        num_prompts = 4 * max_concurrency if is_random else 200
    if args.num_warmups is not None:
        num_warmups = args.num_warmups
    else:
        num_warmups = max_concurrency if is_random else 10
    command = [vllm, "bench", "serve", "--model", args.model,
               "--backend", "vllm", "--endpoint", args.endpoint,
               "--host", endpoint.hostname, "--port", str(endpoint.port),
               "--dataset-name", args.dataset_name, "--num-prompts", str(num_prompts),
               "--num-warmups", str(num_warmups), "--request-rate", args.request_rate,
               "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
               "--seed", str(args.seed), "--temperature", str(args.temperature),
               "--save-result", "--save-detailed",
               "--result-dir", str(args.out.parent), "--result-filename", args.out.name]
    if args.dataset_path:
        command.extend(["--dataset-path", str(args.dataset_path)])
    if max_concurrency:
        command.extend(["--max-concurrency", str(max_concurrency)])
    if args.dataset_name == "sharegpt":
        command.extend(["--sharegpt-output-len", str(args.sharegpt_output_len)])
    if is_random:
        command.extend(["--random-input-len", str(args.random_input_len),
                        "--random-output-len", str(args.random_output_len),
                        "--random-range-ratio", str(args.random_range_ratio),
                        "--random-prefix-len", str(args.random_prefix_len)])
    if args.ignore_eos:
        command.append("--ignore-eos")
    extra = list(args.bench_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    command.extend(extra)
    print("RUN " + shlex.join(command), flush=True)
    started = time.monotonic()
    subprocess.run(command, check=True)
    if not args.out.is_file():
        raise SystemExit(f"vllm bench serve completed without producing {args.out}")
    result = json.loads(args.out.read_text(encoding="utf-8"))
    if result.get("failed") or result.get("completed") != num_prompts:
        raise SystemExit(f"benchmark requests failed or incomplete: completed={result.get('completed')}, "
                         f"failed={result.get('failed')}, expected={num_prompts}")
    print(json.dumps({"result": str(args.out), "elapsed_s": time.monotonic() - started}), flush=True)


def metric(data, name):
    value = data.get(name)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"missing or invalid metric: {name}")
    return value


def compare(args, parser):
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        reference_throughput = metric(reference, "output_throughput")
        candidate_throughput = metric(candidate, "output_throughput")
        reference_tpot, candidate_tpot = metric(reference, "mean_tpot_ms"), metric(candidate, "mean_tpot_ms")
        reference_ttft, candidate_ttft = metric(reference, "mean_ttft_ms"), metric(candidate, "mean_ttft_ms")
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    throughput_speedup = candidate_throughput / reference_throughput
    tpot_speedup = reference_tpot / candidate_tpot
    ttft_speedup = reference_ttft / candidate_ttft
    report = {
        "comparison_type": "performance", "reference": str(args.reference), "candidate": str(args.candidate),
        "reference_label": args.reference_label, "candidate_label": args.candidate_label,
        "reference_metrics": {"output_throughput": reference_throughput,
                              "mean_tpot_ms": reference_tpot, "mean_ttft_ms": reference_ttft},
        "candidate_metrics": {"output_throughput": candidate_throughput,
                              "mean_tpot_ms": candidate_tpot, "mean_ttft_ms": candidate_ttft},
        "output_throughput_speedup": throughput_speedup,
        "tpot_speedup": tpot_speedup, "ttft_speedup": ttft_speedup,
        "thresholds": {"output_throughput_speedup_min": args.min_throughput_speedup,
                       "tpot_speedup_min": args.min_tpot_speedup,
                       "ttft_speedup_min": args.min_ttft_speedup},
        "within_thresholds": (throughput_speedup >= args.min_throughput_speedup
                              and tpot_speedup >= args.min_tpot_speedup
                              and ttft_speedup >= args.min_ttft_speedup)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPARE " + json.dumps(report, ensure_ascii=False), flush=True)
    if args.require_within_thresholds and not report["within_thresholds"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Run and compare official vllm bench serve results.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run one load against an existing server.")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--base-url", required=True)
    run_parser.add_argument("--endpoint", default="/v1/completions")
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--dataset-name", required=True)
    run_parser.add_argument("--dataset-path", type=Path)
    run_parser.add_argument("--num-prompts", type=int)
    run_parser.add_argument("--num-warmups", type=int)
    run_parser.add_argument("--request-rate", required=True)
    run_parser.add_argument("--max-concurrency", type=int)
    run_parser.add_argument("--random-input-len", type=int)
    run_parser.add_argument("--random-output-len", type=int, default=1024)
    run_parser.add_argument("--random-range-ratio", type=float, default=0)
    run_parser.add_argument("--random-prefix-len", type=int, default=0)
    run_parser.add_argument("--sharegpt-output-len", type=int, default=256)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("bench_args", nargs=argparse.REMAINDER,
                            help="Additional vllm bench serve arguments after --.")
    run_parser.set_defaults(handler=run)
    compare_parser = subparsers.add_parser("compare", help="Compare two existing official result JSON files.")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--out", type=Path, required=True)
    compare_parser.add_argument("--reference-label", default="default")
    compare_parser.add_argument("--candidate-label", default="inductor")
    compare_parser.add_argument("--min-throughput-speedup", type=float, default=0.95)
    compare_parser.add_argument("--min-tpot-speedup", type=float, default=1 / 1.05)
    compare_parser.add_argument("--min-ttft-speedup", type=float, default=1 / 1.10)
    compare_parser.add_argument("--require-within-thresholds", action="store_true")
    compare_parser.set_defaults(handler=compare)
    args = parser.parse_args()
    args.handler(args, parser)


if __name__ == "__main__":
    main()
