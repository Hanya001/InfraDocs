import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

CASES = {
    "mmlu": {
        "case_type": "accuracy", "dataset_path": "vllm-ascend/mmlu",
        "request_conf": "vllm_api_general_chat", "dataset_conf": "mmlu/mmlu_fixed_custom",
        "max_out_len": 2048, "batch_size": 32, "temperature": 0,
    },
    "gsm8k": {
        "case_type": "accuracy", "dataset_path": "vllm-ascend/gsm8k-lite",
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_chat_prompt",
        "max_out_len": 2048, "batch_size": 32, "temperature": 0,
    },
    "gpqa": {
        "case_type": "accuracy", "dataset_path": "vllm-ascend/gpqa",
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "gpqa/gpqa_gen_0_shot_cot_chat_prompt",
        "max_out_len": 32768, "batch_size": 32, "temperature": 0,
    },
}
DATASETS = {"mmlu": "mmlu_200_custom", "gsm8k": "gsm8k_gen_0_shot_cot_chat_prompt",
            "gpqa": "gpqa_gen_0_shot_cot_chat_prompt"}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_manifest(directory, task, source_files, samples, seed):
    selection_sha256 = hashlib.sha256(json.dumps(samples, ensure_ascii=False).encode()).hexdigest()
    manifest = {"task": task, "seed": seed, "num_samples": len(samples),
                "selection_sha256": selection_sha256, "samples": samples,
                "source_files": {name: file_sha256(path) for name, path in source_files.items()}}
    path = directory.parent / f"{task}_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest, path


def prepare_mmlu_data(source, destination, count, seed):
    source_files = {path.name: path for path in sorted((source / "test").glob("*_test.csv"))}
    if not source_files:
        raise ValueError(f"no MMLU test CSV files found under {source / 'test'}")
    rows = {}
    available = {}
    for name, path in source_files.items():
        subject = name.removesuffix("_test.csv")
        with path.open(encoding="utf-8", newline="") as source_csv:
            rows[subject] = list(csv.reader(source_csv))
        if any(len(row) != 6 for row in rows[subject]):
            raise ValueError(f"invalid MMLU row in {path}: expected 6 columns")
        available[subject] = list(range(len(rows[subject])))
        random.Random(f"{seed}:{subject}").shuffle(available[subject])
    if sum(map(len, available.values())) < count:
        raise ValueError("MMLU sample count exceeds the available dataset")
    selected_rows = []
    samples = []
    while len(samples) < count:
        for subject in sorted(rows):
            if not available[subject]:
                continue
            row_index = available[subject].pop()
            selected_rows.append(rows[subject][row_index])
            samples.append({"sample_id": f"{subject}:{row_index}", "subject": subject,
                            "source_row": row_index})
            if len(samples) == count:
                break
    destination.mkdir(parents=True)
    (destination / "dev").mkdir()
    (destination / "test").mkdir()
    with (destination / "dev/fixed200_dev.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        for path in (source / "dev").glob("*_dev.csv"):
            with path.open(encoding="utf-8", newline="") as source_csv:
                writer.writerows(csv.reader(source_csv))
    with (destination / "test/fixed200_test.csv").open("w", encoding="utf-8", newline="") as target:
        csv.writer(target).writerows(selected_rows)
    return save_manifest(destination, "mmlu", source_files, samples, seed)


def prepare_gsm8k_data(source, destination, count, seed):
    test_path = source / "test.jsonl"
    with test_path.open(encoding="utf-8") as source_jsonl:
        rows = [json.loads(line) for line in source_jsonl if line.strip()]
    if len(rows) < count:
        raise ValueError("GSM8K sample count exceeds the available dataset")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    indices = indices[:count]
    destination.mkdir(parents=True)
    with (destination / "test.jsonl").open("w", encoding="utf-8") as target:
        for index in indices:
            target.write(json.dumps(rows[index], ensure_ascii=False) + "\n")
    if (source / "train.jsonl").is_file():
        shutil.copy2(source / "train.jsonl", destination / "train.jsonl")
    samples = [{"sample_id": f"test:{index}", "source_row": index} for index in indices]
    return save_manifest(destination, "gsm8k", {"test.jsonl": test_path}, samples, seed)


def prepare_gpqa_data(source, destination, count):
    source_path = source / "gpqa_diamond.csv"
    with source_path.open(encoding="utf-8", newline="") as source_csv:
        rows = [row for row in csv.reader(source_csv) if row and row[7] != "Question"]
    if any(len(row) < 12 for row in rows):
        raise ValueError(f"invalid GPQA Diamond CSV: {source_path}")
    if count != len(rows):
        raise ValueError(f"GPQA Diamond requires all {len(rows)} samples, got {count}")
    destination.mkdir(parents=True)
    shutil.copy2(source_path, destination / source_path.name)
    samples = [{"sample_id": f"diamond:{index}", "source_row": index}
               for index in range(len(rows))]
    return save_manifest(destination, "gpqa", {source_path.name: source_path}, samples, 0)


def task_num_prompts(name, args):
    override = getattr(args, f"{name}_num_prompts")
    return args.num_prompts if override is None else override


def prepare_task_data(name, args):
    num_prompts = task_num_prompts(name, args)
    if args.fixed_data_root and name in ("mmlu", "gsm8k"):
        manifest_path = args.fixed_data_root / f"{name}_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("num_samples") != num_prompts or len(manifest.get("samples", [])) != num_prompts:
            raise ValueError(f"fixed {name} sample count does not match its --num-prompts setting")
        selection_sha256 = hashlib.sha256(
            json.dumps(manifest["samples"], ensure_ascii=False).encode()
        ).hexdigest()
        if selection_sha256 != manifest.get("selection_sha256"):
            raise ValueError(f"fixed {name} manifest hash mismatch")
        dataset_name = "mmlu" if name == "mmlu" else "gsm8k-lite"
        dataset = args.out / "input_data" / dataset_name
        shutil.copytree(args.fixed_data_root / dataset_name, dataset)
        shutil.copy2(manifest_path, dataset.parent / f"{name}_manifest.json")
        return dataset, {"sample_manifest": str((dataset.parent / f"{name}_manifest.json").relative_to(args.out)),
                         "selection_sha256": selection_sha256,
                         "subjects": sorted({row["subject"] for row in manifest["samples"]}) if name == "mmlu" else [],
                         "num_samples": num_prompts}
    if name == "mmlu":
        if not args.mmlu_dataset:
            raise ValueError("--mmlu-dataset is required for fixed MMLU samples")
        dataset = args.out / "input_data" / "mmlu"
        manifest, path = prepare_mmlu_data(args.mmlu_dataset, dataset, num_prompts, args.sample_seed)
    elif name == "gsm8k":
        if not args.gsm8k_dataset:
            raise ValueError("--gsm8k-dataset is required for fixed GSM8K samples")
        dataset = args.out / "input_data" / "gsm8k-lite"
        manifest, path = prepare_gsm8k_data(args.gsm8k_dataset, dataset, num_prompts, args.sample_seed)
    else:
        if not args.gpqa_dataset:
            raise ValueError("--gpqa-dataset is required for GPQA Diamond")
        dataset = args.out / "input_data" / "gpqa"
        manifest, path = prepare_gpqa_data(args.gpqa_dataset, dataset, num_prompts)
    return dataset, {"sample_manifest": str(path.relative_to(args.out)),
                     "selection_sha256": manifest["selection_sha256"],
                     "subjects": sorted({row["subject"] for row in manifest["samples"]}) if name == "mmlu" else [],
                     "num_samples": manifest["num_samples"]}


def prepare_mmlu_config(benchmark_home, dataset, subjects, selection_sha256):
    config_dir = benchmark_home / "ais_bench/benchmark/configs/datasets/mmlu"
    source_config = (config_dir / "mmlu_gen_0_shot_cot_chat_prompt.py").read_text(encoding="utf-8")
    source_config += f"\nmmlu_datasets = [d for d in mmlu_datasets if d['name'] in {subjects!r}]\n"
    prompt = ("There is a single choice question. Answer the question by replying A, B, C or D. "
              "The last line of your response should be of the form Answer: $ANSWER (without quotes) "
              "where $ANSWER is the answer to the question.\nQuestion: {input}\nA. {A}\nB. {B}\n"
              "C. {C}\nD. {D}\nLet's think step by step.")
    test_files = list((dataset / "test").glob("*_test.csv"))
    if len(test_files) != 1:
        raise ValueError(f"expected one fixed MMLU test CSV under {dataset / 'test'}")
    data_name = test_files[0].name.removesuffix("_test.csv")
    source_config += "base = mmlu_datasets[0]\n"
    source_config += f"base['abbr'] = 'mmlu_{data_name}'\nbase['name'] = {data_name!r}\n"
    source_config += f"base['path'] = {str(dataset)!r}\n"
    source_config += f"base['infer_cfg']['prompt_template']['template']['round'][0]['prompt'] = {prompt!r}\n"
    source_config += "mmlu_datasets = [base]\n"
    config_name = f"mmlu_fixed_{selection_sha256[:12]}"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=config_dir, delete=False) as config:
        config.write(source_config)
        temporary_config = config.name
    os.replace(temporary_config, config_dir / f"{config_name}.py")
    return config_name


def prepare_gsm8k_config(benchmark_home, dataset, selection_sha256):
    config_dir = benchmark_home / "ais_bench/benchmark/configs/datasets/gsm8k"
    source_config = (config_dir / "gsm8k_gen_0_shot_cot_chat_prompt.py").read_text(encoding="utf-8")
    source_config += f"\nfor d in gsm8k_datasets: d['path'] = {str(dataset)!r}\n"
    config_name = f"gsm8k_fixed_{selection_sha256[:12]}"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=config_dir, delete=False) as config:
        config.write(source_config)
        temporary_config = config.name
    os.replace(temporary_config, config_dir / f"{config_name}.py")
    return config_name


def prepare_gpqa_config(benchmark_home, dataset, selection_sha256):
    config_dir = benchmark_home / "ais_bench/benchmark/configs/datasets/gpqa"
    source_config = (config_dir / "gpqa_gen_0_shot_cot_chat_prompt.py").read_text(encoding="utf-8")
    source_config += f"\nfor d in gpqa_datasets: d['path'] = {str(dataset)!r}\n"
    config_name = f"gpqa_fixed_{selection_sha256[:12]}"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=config_dir, delete=False) as config:
        config.write(source_config)
        temporary_config = config.name
    os.replace(temporary_config, config_dir / f"{config_name}.py")
    return config_name


def get_accuracy_result(bench, allow_response_anomalies):
    try:
        bench._get_result_accuracy()
        return bench.result, None
    except AssertionError as error:
        if not allow_response_anomalies or not str(error).startswith("Response anomaly verification failed"):
            raise
    summary = re.search(r"write csv to (.*)", bench.result_line).group(1)
    with open(summary, encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    anomaly_rows = [row for row in rows if row["metric"].startswith("response_anomaly_")]
    accuracy_rows = [row for row in rows if not row["metric"].startswith("response_anomaly_")]
    value_column = list(rows[0])[-1]
    score = float(accuracy_rows[0][value_column])
    anomalies = {"summary": summary, "metrics": {row["metric"]: row[value_column] for row in anomaly_rows}}
    return score, anomalies


def run(args, parser):
    tasks = args.tasks.split(",")
    unknown = set(tasks) - set(CASES)
    if unknown or not all(tasks):
        parser.error(f"unknown tasks: {','.join(sorted(unknown)) or args.tasks}")
    if args.num_prompts < 1:
        parser.error("--num-prompts must be positive")
    for name in tasks:
        if task_num_prompts(name, args) < 1:
            parser.error(f"--{name}-num-prompts must be positive")
    endpoint = urllib.parse.urlparse(args.base_url)
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname or not endpoint.port:
        parser.error("--base-url must include an http(s) host and port")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out = args.out.resolve()
    args.benchmark_home = args.benchmark_home.resolve()
    args.vllm_ascend_source = args.vllm_ascend_source.resolve()
    if args.fixed_data_root:
        args.fixed_data_root = args.fixed_data_root.resolve()
    if args.mmlu_dataset:
        args.mmlu_dataset = args.mmlu_dataset.resolve()
    if args.gsm8k_dataset:
        args.gsm8k_dataset = args.gsm8k_dataset.resolve()
    if args.gpqa_dataset:
        args.gpqa_dataset = args.gpqa_dataset.resolve()
    args.out.mkdir(parents=True)
    os.chdir(args.out)
    try:
        import zstandard  # noqa: F401
    except ImportError:
        raise SystemExit(
            "missing dependency zstandard: ais_bench prints a non-fatal ERROR without it and "
            "tools/aisbench.py treats any ERROR line as fatal. Install zstandard and pin xgrammar back."
        )
    if "mmlu" in tasks:
        source_config = (args.benchmark_home / "ais_bench/benchmark/configs/datasets/mmlu" /
                         "mmlu_gen_0_shot_cot_chat_prompt.py")
        if not source_config.is_file():
            parser.error(f"missing AISBench MMLU config: {source_config}")
    os.environ["BENCHMARK_HOME"] = str(args.benchmark_home)
    sys.path.insert(0, str(args.vllm_ascend_source))
    from tools.aisbench import AisbenchRunner

    results = {"label": args.label, "model": args.model, "base_url": args.base_url,
               "evaluation": {"baseline_kind": "paired_eager_comparison",
                              "official_recipe_comparison": False,
                              "source_benchmark_home": str(args.benchmark_home),
                              "sample_seed": args.sample_seed}}
    started = time.monotonic()
    for name in tasks:
        case_started = time.monotonic()
        num_prompts = task_num_prompts(name, args)
        case = dict(CASES[name], model_path=args.model, num_prompts=num_prompts)
        dataset, sample_info = prepare_task_data(name, args)
        case["dataset_path_local"] = str(dataset)
        if name == "mmlu":
            config_name = prepare_mmlu_config(args.benchmark_home, dataset, sample_info["subjects"],
                                              sample_info["selection_sha256"])
            case["dataset_conf"] = f"mmlu/{config_name}"
            sample_info["dataset_conf"] = case["dataset_conf"]
        elif name == "gsm8k":
            config_name = prepare_gsm8k_config(args.benchmark_home, dataset,
                                               sample_info["selection_sha256"])
            case["dataset_conf"] = f"gsm8k/{config_name}"
            sample_info["dataset_conf"] = case["dataset_conf"]
        else:
            config_name = prepare_gpqa_config(args.benchmark_home, dataset,
                                              sample_info["selection_sha256"])
            case["dataset_conf"] = f"gpqa/{config_name}"
            sample_info["dataset_conf"] = case["dataset_conf"]
        with AisbenchRunner(model=args.model, port=endpoint.port, host_ip=endpoint.hostname,
                            aisbench_config=case, verify=False) as bench:
            if not hasattr(bench, "result"):
                result, anomalies = get_accuracy_result(bench, args.allow_response_anomalies)
            else:
                result, anomalies = bench.result, None
        entry = {"label": args.label, "num_prompts": num_prompts, "max_out_len": case["max_out_len"],
                 "elapsed_s": time.monotonic() - case_started, **sample_info}
        if anomalies:
            entry["response_anomalies"] = anomalies
        if isinstance(result, dict):
            entry.update(result)
        else:
            entry["score"] = result
        entry["detail_files"] = task_detail_files(args.out)
        results[name] = entry
        print(f"ACCURACY {args.label} {name} {json.dumps(results[name])}", flush=True)
    results["total_elapsed_s"] = time.monotonic() - started
    (args.out / "accuracy_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def rescore_dataset(run_dir, task):
    result_path = run_dir / "accuracy_results.json"
    if result_path.is_file():
        results = json.loads(result_path.read_text(encoding="utf-8"))
        config = results.get(task, {}).get("dataset_conf")
        if config:
            return config.split("/")[-1]
    return DATASETS[task]


def rescore_benchmark_home(run_dir):
    result_path = run_dir / "accuracy_results.json"
    if result_path.is_file():
        results = json.loads(result_path.read_text(encoding="utf-8"))
        return results.get("evaluation", {}).get("source_benchmark_home", str(run_dir / "benchmark_home"))
    return str(run_dir / "benchmark_home")


def rescore(args, parser):
    if args.num_prompts < 1:
        parser.error("--num-prompts must be positive")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    candidates = sorted((args.run_dir / "outputs/default").glob("*/predictions"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    if args.timestamp:
        predictions = args.run_dir / "outputs/default" / args.timestamp / "predictions"
        if not predictions.is_dir():
            parser.error(f"predictions directory not found: {predictions}")
    elif candidates:
        predictions = candidates[0]
    else:
        parser.error(f"no predictions found under {args.run_dir}")
    count = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in predictions.rglob("*.jsonl"))
    timestamp = predictions.parent.name
    command = ["ais_bench", "--models", "vllm_api_general_chat_custom",
               "--datasets", rescore_dataset(args.run_dir, args.task), "--mode", "eval", "--reuse", timestamp,
               "--num-prompts", str(args.num_prompts)]
    print("RUN " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["BENCHMARK_HOME"] = rescore_benchmark_home(args.run_dir)
    subprocess.run(command, cwd=args.run_dir, env=environment, check=True)
    summaries = sorted((predictions.parent / "summary").glob("summary_*.csv"),
                       key=lambda path: path.stat().st_mtime)
    if not summaries:
        raise SystemExit("AISBench completed but no summary CSV was produced")
    with summaries[-1].open(encoding="utf-8", newline="") as source:
        rows = list(csv.reader(source))
    result = {"task": args.task, "score": float(rows[1][-1]), "reuse_timestamp": timestamp,
              "predictions_count": count, "predictions_expected": args.num_prompts,
              "predictions_complete": count == args.num_prompts, "summary": str(summaries[-1])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESCORE " + json.dumps(result, ensure_ascii=False), flush=True)


def task_detail_files(out_dir):
    output_dirs = [path for path in (out_dir / "outputs/default").glob("*") if path.is_dir()]
    if not output_dirs:
        return []
    latest = max(output_dirs, key=lambda path: path.stat().st_mtime)
    files = []
    for path in latest.glob("results/**/*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("details"), list):
            files.append(str(path.relative_to(out_dir)))
    return sorted(files)


def load_scored_samples(result_path, entry):
    samples = {}
    for relative in entry.get("detail_files", []):
        path = result_path.parent / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        for index, row in enumerate(data["details"]):
            local_id = row.get("id", index)
            sample_id = f"{path.stem}:{local_id}"
            if sample_id in samples:
                raise ValueError(f"duplicate sample ID in {entry['label']}: {sample_id}")
            if not isinstance(row.get("correct"), bool):
                raise ValueError(f"missing boolean correct field in {path}: {sample_id}")
            samples[sample_id] = {"correct": row["correct"], "prediction": row.get("pred"),
                                  "answer": row.get("answer")}
    return samples


def invalid_prediction(sample):
    value = sample["prediction"]
    return value is None or (isinstance(value, str) and value.strip().lower() in ("", "null", "none"))


def compare_samples(reference_path, reference_entry, candidate_path, candidate_entry):
    if not reference_entry.get("detail_files") or not candidate_entry.get("detail_files"):
        return {"available": False}
    reference = load_scored_samples(reference_path, reference_entry)
    candidate = load_scored_samples(candidate_path, candidate_entry)
    if not reference or not candidate:
        return {"available": False}
    common = sorted(set(reference) & set(candidate))
    transitions = {"correct_to_wrong": 0, "wrong_to_correct": 0,
                   "both_wrong_answer_changed": 0}
    changed = []
    for sample_id in common:
        a, b = reference[sample_id], candidate[sample_id]
        if a["correct"] and not b["correct"]:
            transition = "correct_to_wrong"
        elif not a["correct"] and b["correct"]:
            transition = "wrong_to_correct"
        elif not a["correct"] and not b["correct"] and a["prediction"] != b["prediction"]:
            transition = "both_wrong_answer_changed"
        else:
            continue
        transitions[transition] += 1
        changed.append({"sample_id": sample_id, "transition": transition,
                        "reference_prediction": a["prediction"], "candidate_prediction": b["prediction"]})
    return {"available": True, "common_samples": len(common),
            "missing_in_reference": sorted(set(candidate) - set(reference)),
            "missing_in_candidate": sorted(set(reference) - set(candidate)),
            "reference_invalid_predictions": sum(invalid_prediction(reference[key]) for key in common),
            "candidate_invalid_predictions": sum(invalid_prediction(candidate[key]) for key in common),
            **transitions, "changed_samples": changed}


def numeric_score(data, task):
    value = data[task]["score"]
    if isinstance(value, dict):
        value = value.get("accuracy", value.get("score"))
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric score for task {task}")
    return value


def compare(args, parser):
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        tasks = {}
        for task in args.tasks.split(","):
            reference_score, candidate_score = numeric_score(reference, task), numeric_score(candidate, task)
            tasks[task] = {"reference_score": reference_score, "candidate_score": candidate_score,
                           "difference_percentage_points": candidate_score - reference_score,
                           "sample_comparison": compare_samples(
                               args.reference, reference[task], args.candidate, candidate[task])}
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    report = {"comparison_type": "task_accuracy", "reference": str(args.reference),
              "candidate": str(args.candidate), "reference_label": args.reference_label,
              "candidate_label": args.candidate_label, "tasks": tasks}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPARE " + json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run, rescore and compare AISBench task accuracy.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run tasks against an existing server.")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--base-url", required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--label", default="candidate")
    run_parser.add_argument("--vllm-ascend-source", type=Path, required=True)
    run_parser.add_argument("--benchmark-home", type=Path, required=True)
    run_parser.add_argument("--mmlu-dataset", type=Path)
    run_parser.add_argument("--gsm8k-dataset", type=Path)
    run_parser.add_argument("--gpqa-dataset", type=Path)
    run_parser.add_argument("--num-prompts", type=int, default=200)
    run_parser.add_argument("--fixed-data-root", type=Path,
                            help="Reuse a fixed200/fixed1000_500 directory without sampling")
    run_parser.add_argument("--allow-response-anomalies", action="store_true",
                            help="Record nonzero response anomaly metrics instead of aborting the remaining tasks")
    run_parser.add_argument("--mmlu-num-prompts", type=int)
    run_parser.add_argument("--gsm8k-num-prompts", type=int)
    run_parser.add_argument("--gpqa-num-prompts", type=int)
    run_parser.add_argument("--sample-seed", type=int, default=0)
    run_parser.add_argument("--tasks", default="mmlu,gsm8k")
    run_parser.set_defaults(handler=run)
    rescore_parser = subparsers.add_parser("rescore", help="Score existing predictions without inference.")
    rescore_parser.add_argument("--run-dir", type=Path, required=True)
    rescore_parser.add_argument("--task", choices=tuple(DATASETS), required=True)
    rescore_parser.add_argument("--num-prompts", type=int, required=True)
    rescore_parser.add_argument("--timestamp")
    rescore_parser.add_argument("--out", type=Path, required=True)
    rescore_parser.set_defaults(handler=rescore)
    compare_parser = subparsers.add_parser("compare", help="Compare two accuracy_results.json files.")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--out", type=Path, required=True)
    compare_parser.add_argument("--tasks", default="mmlu,gsm8k")
    compare_parser.add_argument("--reference-label", default="eager")
    compare_parser.add_argument("--candidate-label", default="inductor")
    compare_parser.set_defaults(handler=compare)
    args = parser.parse_args()
    args.handler(args, parser)


if __name__ == "__main__":
    main()
