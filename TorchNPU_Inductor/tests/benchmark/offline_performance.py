import argparse
import json
import shlex
import shutil
import statistics
import subprocess
import time
from pathlib import Path


def metric(data, name):
    value = data.get(name)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"missing or invalid metric: {name}")
    return value


def run(args, parser):
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if args.tp < 1:
        parser.error("--tp must be positive")
    if args.benchmark == "throughput" and not args.dataset_path:
        parser.error("--dataset-path is required for throughput")
    if args.benchmark == "throughput" and min(args.num_prompts, args.num_warmups) < 0:
        parser.error("--num-prompts and --num-warmups must be non-negative")
    if args.benchmark == "throughput" and args.num_prompts < 1:
        parser.error("--num-prompts must be positive")
    if args.benchmark == "latency" and min(args.input_len, args.output_len, args.batch_size,
                                          args.num_iters_warmup, args.num_iters) < 1:
        parser.error("latency sizes and iteration counts must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vllm = shutil.which("vllm")
    if not vllm:
        raise SystemExit("vllm executable not found")
    command = [vllm, "bench", args.benchmark, "--model", args.model,
               "--tensor-parallel-size", str(args.tp), "--seed", str(args.seed)]
    if args.benchmark == "latency":
        command.extend(["--input-len", str(args.input_len), "--output-len", str(args.output_len),
                        "--batch-size", str(args.batch_size),
                        "--num-iters-warmup", str(args.num_iters_warmup),
                        "--num-iters", str(args.num_iters), "--output-json", str(args.out)])
    else:
        command.extend(["--dataset-name", "sharegpt", "--dataset-path", str(args.dataset_path),
                        "--num-prompts", str(args.num_prompts),
                        "--num-warmups", str(args.num_warmups),
                        "--backend", "vllm", "--output-json", str(args.out)])
    if args.track == "inductor":
        command.extend(["--compilation-config", json.dumps({"backend": "inductor"}, separators=(",", ":"))])
    extra = list(args.vllm_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    command.extend(extra)
    print("RUN " + shlex.join(command), flush=True)
    started = time.monotonic()
    subprocess.run(command, check=True)
    if not args.out.is_file():
        raise SystemExit(f"benchmark completed without producing {args.out}")
    result = json.loads(args.out.read_text(encoding="utf-8"))
    if args.benchmark == "latency":
        if len(result.get("latencies", [])) != args.num_iters:
            raise SystemExit("latency result does not contain the expected number of iterations")
        metric(result, "avg_latency")
    else:
        metric(result, "requests_per_second")
        metric(result, "tokens_per_second")
    print(json.dumps({"result": str(args.out), "elapsed_s": time.monotonic() - started}), flush=True)


def compare(args, parser):
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        if args.benchmark == "latency":
            names = ("avg_latency", "p99_latency")
            values = {
                "avg_latency": (metric(reference, "avg_latency"), metric(candidate, "avg_latency")),
                "p99_latency": (metric(reference["percentiles"], "99"),
                                metric(candidate["percentiles"], "99")),
            }
        else:
            names = ("requests_per_second", "tokens_per_second")
            values = {name: (metric(reference, name), metric(candidate, name)) for name in names}
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    metrics = {}
    for name in names:
        reference_value, candidate_value = values[name]
        speedup = (reference_value / candidate_value if args.benchmark == "latency"
                   else candidate_value / reference_value)
        metrics[name] = {"reference": reference_value, "candidate": candidate_value, "speedup": speedup}
    report = {"comparison_type": "offline_performance", "benchmark": args.benchmark,
              "reference": str(args.reference), "candidate": str(args.candidate),
              "reference_label": args.reference_label, "candidate_label": args.candidate_label,
              "metrics": metrics}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPARE " + json.dumps(report, ensure_ascii=False), flush=True)


def summary(args, parser):
    groups = {}
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("comparison_type") != "offline_performance":
            raise ValueError(f"not an offline performance comparison: {path}")
        for name, values in data["metrics"].items():
            groups.setdefault((data["benchmark"], name), []).append(values)
    units = {"avg_latency": "s", "p99_latency": "s",
             "requests_per_second": "requests/s", "tokens_per_second": "tokens/s"}
    lines = ["# 离线性能测试汇总", "", "加速比大于 1 表示 Inductor 更快。", "",
             "| 测试 | 指标 | 轮数 | Default 中位数 | Inductor 中位数 | 加速比中位数 | 加速比最小值 | 加速比最大值 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (benchmark, name), rows in sorted(groups.items()):
        references = statistics.median(row["reference"] for row in rows)
        candidates = statistics.median(row["candidate"] for row in rows)
        speedups = [row["speedup"] for row in rows]
        lines.append(f"| {benchmark} | {name} ({units[name]}) | {len(rows)} "
                     f"| {references:.6g} | {candidates:.6g} | {statistics.median(speedups):.3f} "
                     f"| {min(speedups):.3f} | {max(speedups):.3f} |")
    output = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output, encoding="utf-8")
    print(output, end="")


def main():
    parser = argparse.ArgumentParser(description="Run and compare offline vLLM latency and throughput benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run one offline benchmark.")
    run_parser.add_argument("--benchmark", choices=("latency", "throughput"), required=True)
    run_parser.add_argument("--track", choices=("default", "inductor"), required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--tp", type=int, default=1)
    run_parser.add_argument("--dataset-path", type=Path)
    run_parser.add_argument("--input-len", type=int, default=32)
    run_parser.add_argument("--output-len", type=int, default=128)
    run_parser.add_argument("--batch-size", type=int, default=8)
    run_parser.add_argument("--num-iters-warmup", type=int, default=5)
    run_parser.add_argument("--num-iters", type=int, default=15)
    run_parser.add_argument("--num-prompts", type=int, default=200)
    run_parser.add_argument("--num-warmups", type=int, default=0)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("vllm_args", nargs=argparse.REMAINDER,
                            help="Additional vLLM arguments after --.")
    run_parser.set_defaults(handler=run)
    compare_parser = subparsers.add_parser("compare", help="Compare two offline benchmark results.")
    compare_parser.add_argument("--benchmark", choices=("latency", "throughput"), required=True)
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--out", type=Path, required=True)
    compare_parser.add_argument("--reference-label", default="default")
    compare_parser.add_argument("--candidate-label", default="inductor")
    compare_parser.set_defaults(handler=compare)
    summary_parser = subparsers.add_parser("summary", help="Summarize repeated offline comparisons.")
    summary_parser.add_argument("inputs", type=Path, nargs="+")
    summary_parser.add_argument("--out", type=Path, required=True)
    summary_parser.set_defaults(handler=summary)
    args = parser.parse_args()
    args.handler(args, parser)


if __name__ == "__main__":
    main()
