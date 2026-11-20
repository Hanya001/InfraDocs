import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def value(data, *keys):
    for key in keys:
        if data is None:
            return None
        data = data.get(key)
    return data


def fmt(data, digits=2):
    return "—" if data is None else f"{data:.{digits}f}"


def ratio(default, inductor, latency=False):
    if not default or not inductor:
        return "—"
    value = default / inductor if latency else inductor / default
    return f"{value * 100:.1f}%" if not latency else f"{value:.3f}×"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print("| 模型 | token 一致 | 单请求吞吐 default → Inductor | 16 路吞吐 default → Inductor | 单请求吞吐比 | 16 路吞吐比 | 状态 |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for model in sorted(p for p in args.root.iterdir() if p.is_dir() and not p.name.startswith("_")
                        and not p.name.endswith(("_script_error", "_output_api_error"))):
        comparison = read_json(model / "compare.json")
        accuracy = "—"
        if comparison and "token_identical" in comparison:
            accuracy = f"{comparison['token_identical']}/{comparison['n_prompts']}"
        default = read_json(model / "default" / "perf_metrics.json")
        inductor = read_json(model / "inductor" / "perf_metrics.json")
        b1d = value(default, "perf_batch1", "tok_per_s_e2e")
        b1i = value(inductor, "perf_batch1", "tok_per_s_e2e")
        b16d = value(default, "perf_batch16", "tok_per_s_e2e")
        b16i = value(inductor, "perf_batch16", "tok_per_s_e2e")
        statuses = []
        for track in ("eager", "default", "inductor"):
            code = model / track / "exit_code.txt"
            resume_code = model / track / "exit_code_perf_resume.txt"
            selected = resume_code if resume_code.is_file() else code
            if selected.is_file() and selected.read_text().strip() != "0":
                statuses.append(f"{track}=exit {selected.read_text().strip()}")
        if comparison and "token_identical" in comparison and comparison["token_identical"] != comparison["n_prompts"]:
            statuses.append("精度待后续测试集审查")
        if not statuses:
            statuses.append("完成")
        print(f"| {model.name} | {accuracy} | {fmt(b1d)} → {fmt(b1i)} | {fmt(b16d)} → {fmt(b16i)} | {ratio(b1d, b1i)} | {ratio(b16d, b16i)} | {'；'.join(statuses)} |")


if __name__ == "__main__":
    main()
