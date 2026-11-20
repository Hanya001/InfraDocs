import json
import os
import time
from pathlib import Path

import torch

PROMPTS = [
    "What is the capital of China?",
    "Explain gravity",
    "深度学习是机器学习的一个分支。",
    "A" * 256,
    "请比较 TCP 与 UDP，并说明各自适用的场景。" * 8,
    "Short",
    "Explain why the sky is blue during the day.",
    "写一句话介绍长城。",
]


def main():
    track = os.environ["BENCH_TRACK"]
    model = os.environ["BENCH_MODEL"]
    out = Path(os.environ["BENCH_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    from vllm import LLM

    kw = {}
    if track == "inductor":
        kw["compilation_config"] = {"backend": "inductor"}
        if os.environ.get("BENCH_CUDAGRAPH_MODE"):
            kw["compilation_config"]["cudagraph_mode"] = os.environ["BENCH_CUDAGRAPH_MODE"]
    elif track == "eager":
        kw["enforce_eager"] = True
    elif track != "default":
        raise ValueError(track)

    started = time.monotonic()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=8192,
              tensor_parallel_size=int(os.environ.get("BENCH_TP", "1")),
              seed=0, gpu_memory_utilization=0.80, enable_prefix_caching=False,
              disable_log_stats=False, **kw)
    init_s = time.monotonic() - started
    vectors = None
    timings = []
    for _ in range(3):
        started = time.monotonic()
        outputs = llm.embed(PROMPTS)
        timings.append(time.monotonic() - started)
        current = [torch.tensor(o.outputs.embedding, dtype=torch.float32) for o in outputs]
        if vectors is None:
            vectors = current
        else:
            for a, b in zip(vectors, current):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    (out / "embeddings.json").write_text(json.dumps(
        [v.tolist() for v in vectors], ensure_ascii=False), encoding="utf-8")
    report = {"model": model, "track": track, "engine_init_s": init_s,
              "encode_seconds": timings, "batch_size": len(PROMPTS),
              "embedding_dim": vectors[0].numel()}
    (out / "perf_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    eager_path = out.parent / "eager" / "embeddings.json"
    if track == "inductor" and eager_path.is_file():
        eager = [torch.tensor(v) for v in json.loads(eager_path.read_text(encoding="utf-8"))]
        rows = []
        for a, b in zip(eager, vectors):
            rows.append({"max_abs_diff": (a - b).abs().max().item(),
                         "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item()})
        (out.parent / "compare.json").write_text(json.dumps(
            {"reference_track": "eager", "n_prompts": len(rows), "embedding_metrics": rows}, indent=2),
            encoding="utf-8")
    print(f"EMBEDDING_{track.upper()}_PASS", flush=True)


if __name__ == "__main__":
    main()
