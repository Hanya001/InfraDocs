import json
import os
import sys

base = sys.argv[1]


def load(track):
    with open(os.path.join(base, track, "acc_track.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def candidate_map(row, position):
    positions = row.get("logprobs") or []
    if position >= len(positions):
        return {}
    return {item["token_id"]: item["logprob"] for item in positions[position]["candidates"]}


def logprob_metrics(x, y, first_diff):
    if not x.get("logprobs") or not y.get("logprobs"):
        return None
    differences = []
    overlaps = []
    for position in range(first_diff):
        a = candidate_map(x, position)
        b = candidate_map(y, position)
        if not a or not b:
            continue
        common = set(a) & set(b)
        differences.extend(abs(a[token] - b[token]) for token in common)
        overlaps.append(len(common) / min(len(a), len(b)))
    return {
        "positions_before_first_diff": first_diff,
        "common_candidate_values": len(differences),
        "mean_abs_logprob_diff": sum(differences) / len(differences) if differences else None,
        "max_abs_logprob_diff": max(differences) if differences else None,
        "mean_topk_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
    }


a, b = load("eager"), load("inductor")
assert len(a) == len(b) and [x["prompt"] for x in a] == [x["prompt"] for x in b]
text_eq = sum(x["text"] == y["text"] for x, y in zip(a, b))
tok_eq = sum(x["token_ids"] == y["token_ids"] for x, y in zip(a, b))
diffs = []
logprob_rows = []
for x, y in zip(a, b):
    first = first_difference(x["token_ids"], y["token_ids"])
    metrics = logprob_metrics(x, y, first)
    if metrics is not None:
        metrics["idx"] = x["idx"]
        logprob_rows.append(metrics)
    if x["token_ids"] != y["token_ids"]:
        diffs.append({"idx": x["idx"], "first_diff_pos": first,
                      "len_eager": len(x["token_ids"]), "len_inductor": len(y["token_ids"])})

common_values = [row["common_candidate_values"] for row in logprob_rows]
mean_diffs = [row["mean_abs_logprob_diff"] for row in logprob_rows if row["mean_abs_logprob_diff"] is not None]
max_diffs = [row["max_abs_logprob_diff"] for row in logprob_rows if row["max_abs_logprob_diff"] is not None]
overlaps = [row["mean_topk_overlap"] for row in logprob_rows if row["mean_topk_overlap"] is not None]
report = {
    "model_dir": base,
    "reference_track": "eager",
    "n_prompts": len(a),
    "text_identical": text_eq,
    "token_identical": tok_eq,
    "diff_details": diffs,
    "logprob_summary": {
        "prompts": len(logprob_rows),
        "common_candidate_values": sum(common_values),
        "mean_abs_logprob_diff": sum(mean_diffs) / len(mean_diffs) if mean_diffs else None,
        "max_abs_logprob_diff": max(max_diffs) if max_diffs else None,
        "mean_topk_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
    },
    "logprob_by_prompt": logprob_rows,
}
with open(os.path.join(base, "compare.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
print("COMPARE:", json.dumps(report, ensure_ascii=False))
