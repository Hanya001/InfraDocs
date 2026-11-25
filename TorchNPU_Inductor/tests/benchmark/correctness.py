import argparse
import json
import math
import threading
import time
from pathlib import Path


PREFILL_LOGPROB_ATOL = 0.0689
DECODE_LOGPROB_ATOL = 2 * PREFILL_LOGPROB_ATOL
DECODE_TOPK = 20


def config_value(value):
    if value is None:
        return None
    if hasattr(value, "name"):
        return value.name
    if hasattr(value, "value"):
        return value.value
    return value


def actual_compilation_config(llm):
    vllm_config = llm.llm_engine.vllm_config
    config = vllm_config.compilation_config
    return {"backend": config_value(config.backend),
            "cudagraph_mode": config_value(config.cudagraph_mode),
            "level": config_value(getattr(config, "level", getattr(vllm_config, "optimization_level", None))),
            "mode": config_value(getattr(config, "mode", getattr(vllm_config, "performance_mode", None)))}

PROMPTS_SHORT_ACC = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
ACC_LONG_BASE = [
    ("medicine_zh", [
        "疫苗通过模拟病原体的部分特征，帮助免疫系统在真正感染前形成识别和记忆能力。它通常不会引起完整疾病，但会触发针对特定抗原的免疫反应。",
        "接种疫苗后，抗原呈递细胞处理并展示相关成分，随后激活淋巴细胞。B 细胞产生抗体，部分细胞形成免疫记忆，使身体再次遇到病原体时能够更快响应。",
        "疫苗的作用不仅是保护个人。较高的人群接种率可以减少传播机会，从而降低老人、免疫功能较弱者和无法接种者接触病原体的风险。",
        "不同疫苗的技术路线包括灭活疫苗、减毒疫苗、重组蛋白疫苗和核酸疫苗。它们在抗原表达方式、生产流程、储存条件和接种次数方面存在差异。",
        "常见接种反应包括注射部位疼痛、疲劳和低热，通常持续时间较短。严重不良反应非常少见，但接种机构仍需具备识别和处置过敏等紧急情况的能力。",
        "接种前应核对适用年龄、既往过敏史、当前疾病状态以及疫苗说明书中的禁忌。慢性疾病患者或特殊人群应根据专业医务人员的评估决定是否接种。",
        "疫苗并不能保证完全不感染，也不能替代其他公共卫生措施。接种后仍应结合实际风险采取适当的防护、检测和就医措施。",
        "评价疫苗价值需要综合考虑有效性、安全性、疾病严重程度和人群传播风险，而不是只关注某一次轻微反应或个别感染案例。",
    ], "请根据上文，用三句话说明疫苗的主要作用以及接种时需要注意的事项。"),
    ("law_zh", [
        "个人信息处理应当具有明确、合理的目的，并与处理目的直接相关。组织不能仅因为技术上能够收集数据，就无限制地收集与业务无关的信息。",
        "知情同意要求处理者以清晰方式说明处理目的、方式、信息种类和保存期限。对敏感个人信息或向第三方提供信息等情形，还需要履行更严格的告知和同意义务。",
        "最小必要原则要求收集的信息限于实现业务目的所必需的范围。应用不得通过一次笼统授权持续扩展数据用途，也不应以无关权限作为提供基本功能的条件。",
        "处理者需要采取访问控制、加密、审计和备份等安全措施，防止未经授权的访问、泄露、篡改或丢失。发生安全事件时，应及时处置并履行通知义务。",
        "保存个人信息应有明确期限，超过实现目的所需时间后应删除或匿名化。无期限保存历史数据会增加泄露风险，也不符合目的限制要求。",
        "个人通常享有查询、复制、更正、删除和撤回同意等权利。处理者应提供可操作的请求渠道，并在规定时间内核实和回应。",
        "委托第三方处理或共享数据时，应明确双方责任并进行必要监督。合同不能替代安全管理，也不能用来掩盖超出原始目的的数据使用。",
        "合规工作需要贯穿产品设计、开发、运营和删除的整个生命周期，而不是只在隐私政策中增加一段文字。",
    ], "请根据上文，用两句话概括处理个人信息必须遵守的核心原则。"),
    ("climate_en", [
        "The greenhouse effect is a natural process in which atmospheric gases absorb and re-emit infrared radiation. Without this process, the average surface temperature of Earth would be much lower than it is today.",
        "Human activities increase the concentrations of carbon dioxide, methane, and other greenhouse gases. Fossil fuel combustion, deforestation, agriculture, and industrial processes are major contributors to this increase.",
        "Observed warming is supported by surface measurements, ocean heat content, glacier retreat, sea-level rise, and changes in ecosystems. Multiple independent records show a consistent long-term trend.",
        "Attribution studies compare observed changes with simulations of natural and human influences. The recent warming pattern cannot be explained by solar variation or volcanic activity alone.",
        "Climate change affects heat waves, heavy rainfall, drought, coastal flooding, and food production. The impacts differ by region and depend on both physical hazards and social vulnerability.",
        "Mitigation aims to reduce emissions or remove greenhouse gases from the atmosphere. Renewable energy, efficiency, electrification, forest protection, and carbon storage are common mitigation approaches.",
        "Adaptation aims to reduce damage from changes that are already occurring. Examples include heat-response plans, resilient infrastructure, water management, and improved coastal protection.",
        "Effective climate policy requires both mitigation and adaptation. Mitigation limits future risks, while adaptation addresses the impacts that communities already face.",
    ], "Based on the passage, explain the main human causes of climate change and describe how mitigation and adaptation respond to them. Use three sentences."),
    ("history_en", [
        "Before the printing press, European books were copied largely by hand. Producing a manuscript required substantial labor, and variations introduced by copyists could create different versions of the same text.",
        "Movable-type printing allowed pages to be produced repeatedly from reusable components. This reduced the time and labor needed for each copy and made larger print runs practical.",
        "As books became less expensive and more available, access to written knowledge expanded beyond monasteries and small elite groups. Literacy and commercial publishing grew together in many urban centers.",
        "Universities and scholars benefited from more consistent copies of legal, scientific, and philosophical works. Comparing texts and citing specific editions became easier when many readers could access the same printed version.",
        "Printing also changed religious and political communication. Pamphlets and translated texts could reach wider audiences, helping new ideas circulate while creating new conflicts over authority and censorship.",
        "The technology supported the spread of scientific observations, maps, and technical diagrams. Printers and scholars developed conventions for correction, citation, and the organization of knowledge.",
        "The effects were not immediate or uniform across Europe. Costs, literacy, local institutions, and political control influenced how quickly printing changed each community.",
        "Over time, printing helped create larger reading publics and more durable networks of learning. Its importance lies not only in producing more books, but also in changing who could participate in intellectual life.",
    ], "Based on the passage, explain how the printing press changed European society. Use three sentences."),
]
PROMPTS_LONG_ACC = []
for _tag, _paras, _question in ACC_LONG_BASE:
    _heading = "第{}部分" if _tag.endswith("_zh") else "Section {}"
    _sections = [_heading.format(index + 1) + "\n" + paragraph
                 for index, paragraph in enumerate(_paras)]
    PROMPTS_LONG_ACC.append("\n\n".join(_sections) + "\n\n" + _question)
PROMPTS_ACC = PROMPTS_SHORT_ACC + PROMPTS_LONG_ACC

EMBEDDING_PROMPTS = [
    "What is the capital of China?",
    "Explain gravity",
    "深度学习是机器学习的一个分支。",
    "The Great Wall of China is a series of fortifications built across the historical "
    "northern borders of ancient Chinese states, constructed over many centuries to "
    "protect against invasions, and today it stands as one of the most recognizable "
    "symbols of Chinese history and engineering.",
    "机器学习是人工智能的一个分支，它通过数据训练模型，使计算机能够在没有明确编程的情况下"
    "对未知情况做出预测或决策，近年来已经深入到搜索、推荐、医疗影像和自动驾驶等众多领域。",
    "Short",
    "Explain why the sky is blue during the day.",
    "写一句话介绍长城。",
]


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_output(out):
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)


def engine_kwargs(args):
    kwargs = {"data_parallel_size": args.dp}
    if args.track == "inductor":
        kwargs["compilation_config"] = {"backend": "inductor"}
        if args.cudagraph_mode:
            kwargs["compilation_config"]["cudagraph_mode"] = args.cudagraph_mode
    if args.track == "eager" or args.enforce_eager:
        kwargs["enforce_eager"] = True
    if getattr(args, "enable_expert_parallel", False):
        kwargs["enable_expert_parallel"] = True
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if args.tokenizer_mode:
        kwargs["tokenizer_mode"] = args.tokenizer_mode
    if getattr(args, "mtp", False):
        kwargs["speculative_config"] = {"method": "mtp", "num_speculative_tokens": args.mtp_tokens}
    return kwargs


def validate_engine_args(args, parser):
    if args.tp < 1 or args.dp < 1 or args.repeats < 1:
        parser.error("--tp, --dp and --repeats must be positive")
    if args.cudagraph_mode and args.track != "inductor":
        parser.error("--cudagraph-mode is only valid with --track inductor")
    if getattr(args, "mtp", False) and args.mtp_tokens < 1:
        parser.error("--mtp-tokens must be positive")
    try:
        prepare_output(args.out)
    except ValueError as error:
        parser.error(str(error))


def serialize_logprobs(output):
    positions = []
    for position, candidates in enumerate(output.logprobs or []):
        values = [{"token_id": token_id, "token": item.decoded_token,
                   "logprob": item.logprob, "rank": item.rank}
                  for token_id, item in candidates.items()]
        values.sort(key=lambda item: (item["rank"] is None, item["rank"] or 0, -item["logprob"]))
        positions.append({"position": position, "candidates": values})
    return positions


def generate(args, parser):
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    validate_engine_args(args, parser)
    started = time.monotonic()
    state = {"stage": "starting"}

    def progress(stage, **fields):
        state["stage"] = stage
        print("BENCH_EVENT " + json.dumps({
            "model": args.model, "track": args.track, "stage": stage,
            "elapsed_s": round(time.monotonic() - started), **fields}, ensure_ascii=False), flush=True)

    def heartbeat():
        while True:
            time.sleep(60)
            print("BENCH_HEARTBEAT " + json.dumps({
                "model": args.model, "track": args.track, "stage": state["stage"],
                "elapsed_s": round(time.monotonic() - started)}, ensure_ascii=False), flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    progress("engine_initializing")
    from vllm import LLM, SamplingParams

    config = dict(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len,
                  max_num_seqs=args.max_num_seqs, tensor_parallel_size=args.tp, seed=args.seed,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  enable_prefix_caching=args.enable_prefix_caching,
                  disable_log_stats=False, **engine_kwargs(args))
    if args.kv_cache_dtype:
        config["kv_cache_dtype"] = args.kv_cache_dtype
    init_started = time.monotonic()
    llm = LLM(**config)
    engine_init_s = time.monotonic() - init_started
    save_json(args.out / "run_config.json", {
        "model": args.model, "track": args.track, "tp": args.tp, "dp": args.dp,
        "engine_init_s": engine_init_s, "repeats": args.repeats,
        "sampling": {"max_tokens": args.max_tokens, "temperature": 0, "top_p": 1.0, "top_k": 0,
                     "seed": args.seed, "logprobs": args.logprobs, "ignore_eos": args.ignore_eos},
        "engine_kwargs": config,
        "requested_compilation_config": config.get("compilation_config"),
        "actual_compilation_config": actual_compilation_config(llm)})
    progress("engine_ready", engine_init_s=round(engine_init_s, 2))
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, top_k=0,
                              seed=args.seed, logprobs=args.logprobs, ignore_eos=args.ignore_eos)
    runs = []
    timings = []
    for repeat in range(1, args.repeats + 1):
        stage_started = time.monotonic()
        progress("generating", repeat=repeat)
        outputs = llm.generate(PROMPTS_ACC, sampling, use_tqdm=False)
        rows = [{"idx": i, "prompt": PROMPTS_ACC[i], "text": result.outputs[0].text,
                 "token_ids": list(result.outputs[0].token_ids),
                 "input_tokens": len(result.prompt_token_ids),
                 "output_tokens": len(result.outputs[0].token_ids),
                 "logprobs": serialize_logprobs(result.outputs[0])}
                for i, result in enumerate(outputs)]
        name = "acc_track.jsonl" if repeat == 1 else f"acc_repeat{repeat}.jsonl"
        with (args.out / name).open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        runs.append(rows)
        timings.append({"repeat": repeat, "elapsed_s": time.monotonic() - stage_started})
    stable = all([row["token_ids"] for row in rows] == [row["token_ids"] for row in runs[0]]
                 for rows in runs[1:])
    save_json(args.out / "acc_self_compare.json", {"repeats": args.repeats, "token_stable": stable})
    failed = [row["idx"] for row in runs[0]
              if not row["text"].strip() or "!" * 32 in row["text"] or "\ufffd" * 8 in row["text"]]
    if failed:
        save_json(args.out / "correctness_failed.json",
                  {"reason": "empty or obviously garbled output", "indices": failed})
        progress("correctness_failed", reason="empty or obviously garbled output")
        raise SystemExit(1)
    save_json(args.out / "correctness_complete.json", {
        "model": args.model, "track": args.track, "cudagraph_mode": args.cudagraph_mode or "default",
        "token_stable": stable, "generation_timings": timings})
    progress("generation_complete")


def embedding(args, parser):
    validate_engine_args(args, parser)
    from vllm import LLM

    config = dict(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len,
                  tensor_parallel_size=args.tp, seed=args.seed,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  enable_prefix_caching=args.enable_prefix_caching,
                  disable_log_stats=False, **engine_kwargs(args))
    started = time.monotonic()
    llm = LLM(**config)
    save_json(args.out / "run_config.json", {
        "model": args.model, "track": args.track, "engine_init_s": time.monotonic() - started,
        "repeats": args.repeats, "engine_kwargs": config,
        "requested_compilation_config": config.get("compilation_config"),
        "actual_compilation_config": actual_compilation_config(llm)})
    vectors = None
    timings = []
    stable = True
    for _ in range(args.repeats):
        started = time.monotonic()
        outputs = llm.embed(EMBEDDING_PROMPTS)
        timings.append(time.monotonic() - started)
        current = [[float(value) for value in output.outputs.embedding] for output in outputs]
        if vectors is None:
            vectors = current
        else:
            stable = stable and current == vectors
    save_json(args.out / "embeddings.json", vectors)
    save_json(args.out / "embedding_complete.json", {
        "model": args.model, "track": args.track, "batch_size": len(EMBEDDING_PROMPTS),
        "embedding_dim": len(vectors[0]), "encode_seconds": timings, "exact_repeat_match": stable})
    print(f"EMBEDDING_GENERATED track={args.track} exact_repeat_match={stable}", flush=True)


def load_rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def candidate_map(row, position, topk=None):
    positions = row.get("logprobs") or []
    if position >= len(positions):
        return {}
    return {
        item["token_id"]: item["logprob"]
        for item in positions[position]["candidates"]
        if topk is None or item.get("rank") is None or item["rank"] <= topk
    }


def divergence_side(row, position, token_id, other_row):
    candidates = candidate_map(row, position)
    other_candidates = candidate_map(other_row, position)
    ordered = sorted(candidates.values(), reverse=True)
    gap = ordered[0] - ordered[1] if len(ordered) >= 2 else None
    cross_logprob = other_candidates.get(token_id) if token_id is not None else None
    return {"token_id": token_id, "logprob": candidates.get(token_id),
            "top1_top2_gap": gap,
            "token_in_other_topk": token_id in other_candidates if token_id is not None else None,
            "logprob_in_other_topk": cross_logprob}


def pr_rule_failure(reference, candidate, prefill_tolerance, decode_tolerance):
    if not reference.get("logprobs") or not candidate.get("logprobs"):
        return {"reason": "missing_logprobs"}
    reference_ids = reference["token_ids"]
    candidate_ids = candidate["token_ids"]
    for position, (reference_token, candidate_token) in enumerate(zip(reference_ids, candidate_ids)):
        reference_candidates = candidate_map(reference, position)
        candidate_candidates = candidate_map(candidate, position)
        if reference_token not in reference_candidates or candidate_token not in candidate_candidates:
            return {"reason": "missing_chosen_token_logprob", "position": position,
                    "reference_token": reference_token, "candidate_token": candidate_token}
        tolerance = prefill_tolerance if position == 0 else decode_tolerance
        if position == 0 and reference_token != candidate_token:
            return {"reason": "prefill_token_mismatch", "position": position,
                    "reference_token": reference_token, "candidate_token": candidate_token}
        if reference_token == candidate_token:
            compared_logprob = candidate_candidates[candidate_token]
        else:
            candidate_topk = candidate_map(candidate, position, DECODE_TOPK)
            if reference_token not in candidate_topk:
                return {"reason": "decode_token_not_in_candidate_topk", "position": position,
                        "reference_token": reference_token, "candidate_token": candidate_token,
                        "decode_topk": DECODE_TOPK}
            compared_logprob = candidate_topk[reference_token]
        difference = abs(reference_candidates[reference_token] - compared_logprob)
        if difference > tolerance:
            return {"reason": "prefill_logprob_mismatch" if position == 0 else "decode_logprob_mismatch",
                    "position": position, "reference_token": reference_token,
                    "candidate_token": candidate_token, "logprob_diff": difference,
                    "tolerance": tolerance}
    if len(reference_ids) != len(candidate_ids):
        return {"reason": "token_length_mismatch", "len_reference": len(reference_ids),
                "len_candidate": len(candidate_ids)}
    return None


def compare_generations(reference, candidate, prefill_tolerance=PREFILL_LOGPROB_ATOL,
                        decode_tolerance=DECODE_LOGPROB_ATOL):
    if len(reference) != len(candidate):
        raise ValueError(f"row counts differ: reference={len(reference)} candidate={len(candidate)}")
    if not reference:
        raise ValueError("input files contain no rows")
    for index, (x, y) in enumerate(zip(reference, candidate)):
        if x["prompt"] != y["prompt"]:
            raise ValueError(f"prompt mismatch at row={index}: inputs or script versions differ")
    diffs = []
    logprob_rows = []
    for x, y in zip(reference, candidate):
        first = first_difference(x["token_ids"], y["token_ids"])
        if x["token_ids"] != y["token_ids"]:
            reference_token = x["token_ids"][first] if first < len(x["token_ids"]) else None
            candidate_token = y["token_ids"][first] if first < len(y["token_ids"]) else None
            diffs.append({"idx": x["idx"], "first_diff_pos": first,
                          "len_reference": len(x["token_ids"]), "len_candidate": len(y["token_ids"]),
                          "divergence": {
                              "reference": divergence_side(x, first, reference_token, y),
                              "candidate": divergence_side(y, first, candidate_token, x)}})
        pr_failure = pr_rule_failure(x, y, prefill_tolerance, decode_tolerance)
        differences = []
        overlaps = []
        if x.get("logprobs") and y.get("logprobs"):
            for position in range(first):
                a, b = candidate_map(x, position), candidate_map(y, position)
                if not a or not b:
                    continue
                common = set(a) & set(b)
                differences.extend(abs(a[token] - b[token]) for token in common)
                overlaps.append(len(common) / min(len(a), len(b)))
        logprob_rows.append({
            "idx": x["idx"], "positions_before_first_diff": first,
            "common_candidate_values": len(differences),
            "mean_abs_logprob_diff": sum(differences) / len(differences) if differences else None,
            "max_abs_logprob_diff": max(differences) if differences else None,
            "mean_topk_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
            "pr_failure": pr_failure})
    mean_diffs = [row["mean_abs_logprob_diff"] for row in logprob_rows
                  if row["mean_abs_logprob_diff"] is not None]
    max_diffs = [row["max_abs_logprob_diff"] for row in logprob_rows
                 if row["max_abs_logprob_diff"] is not None]
    overlaps = [row["mean_topk_overlap"] for row in logprob_rows if row["mean_topk_overlap"] is not None]
    summary = {"prompts": len(logprob_rows),
               "common_candidate_values": sum(row["common_candidate_values"] for row in logprob_rows),
               "mean_abs_logprob_diff": sum(mean_diffs) / len(mean_diffs) if mean_diffs else None,
               "max_abs_logprob_diff": max(max_diffs) if max_diffs else None,
               "mean_topk_overlap": sum(overlaps) / len(overlaps) if overlaps else None}
    within = all(row["pr_failure"] is None for row in logprob_rows)
    return {"comparison_type": "generation", "n_prompts": len(reference),
            "text_identical": sum(x["text"] == y["text"] for x, y in zip(reference, candidate)),
            "token_identical": sum(x["token_ids"] == y["token_ids"] for x, y in zip(reference, candidate)),
            "diff_details": diffs, "logprob_summary": summary,
            "logprob_tolerance": {"prefill_abs_logprob_diff_max": prefill_tolerance,
                                  "decode_abs_logprob_diff_max": decode_tolerance,
                                  "decode_topk": DECODE_TOPK},
            "logprob_within_tolerance": within, "logprob_by_prompt": logprob_rows}


def compare_embeddings(reference, candidate, cosine_min, abs_diff_max):
    if len(reference) != len(candidate) or not reference:
        raise ValueError("embedding counts differ or inputs are empty")
    rows = []
    for index, (a, b) in enumerate(zip(reference, candidate)):
        if len(a) != len(b) or not a:
            raise ValueError(f"embedding dimensions differ or are empty at index {index}")
        max_abs_diff = max(abs(x - y) for x, y in zip(a, b))
        norm_a, norm_b = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
        cosine = sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b) if norm_a and norm_b else 0.0
        rows.append({"idx": index, "max_abs_diff": max_abs_diff, "cosine": cosine,
                     "within_tolerance": cosine >= cosine_min and max_abs_diff <= abs_diff_max})
    return {"comparison_type": "embedding", "n_prompts": len(rows),
            "tolerance": {"cosine_min": cosine_min, "abs_diff_max": abs_diff_max},
            "within_tolerance": all(row["within_tolerance"] for row in rows),
            "embedding_metrics": rows}


def compare(args, parser):
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    try:
        if args.type == "generation":
            report = compare_generations(load_rows(args.reference), load_rows(args.candidate),
                                         args.prefill_logprob_tolerance, args.decode_logprob_tolerance)
        else:
            reference = json.loads(args.reference.read_text(encoding="utf-8"))
            candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
            report = compare_embeddings(reference, candidate, args.cosine_min, args.abs_diff_max)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    report.update({"reference": str(args.reference), "candidate": str(args.candidate),
                   "reference_label": args.reference_label, "candidate_label": args.candidate_label})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.out, report)
    print("COMPARE " + json.dumps(report, ensure_ascii=False), flush=True)
    if args.require_token_match and (args.type != "generation" or report["token_identical"] != report["n_prompts"]):
        raise SystemExit(1)
    if args.require_logprob_tolerance and (args.type != "generation" or report["logprob_within_tolerance"] is not True):
        raise SystemExit(1)
    if args.require_within_tolerance and (args.type != "embedding" or not report["within_tolerance"]):
        raise SystemExit(1)


def add_engine_args(parser, embedding_mode=False):
    parser.add_argument("--model", required=True)
    parser.add_argument("--track", choices=("eager", "default", "inductor"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3 if embedding_mode else 2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cudagraph-mode")
    parser.add_argument("--quantization")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tokenizer-mode")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    if embedding_mode:
        return
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--logprobs", type=int, default=DECODE_TOPK)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--mtp-tokens", type=int, default=1)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=False)


def main():
    parser = argparse.ArgumentParser(description="Independent fixed-input generation and correctness comparison tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate", help="Generate fixed-input text results once.")
    add_engine_args(generate_parser)
    generate_parser.set_defaults(handler=generate)
    embedding_parser = subparsers.add_parser("embedding", help="Generate fixed-input embeddings once.")
    add_engine_args(embedding_parser, embedding_mode=True)
    embedding_parser.set_defaults(handler=embedding)
    compare_parser = subparsers.add_parser("compare", help="Compare two existing result files.")
    compare_parser.add_argument("--type", choices=("generation", "embedding"), required=True)
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--out", type=Path, required=True)
    compare_parser.add_argument("--reference-label", default="reference")
    compare_parser.add_argument("--candidate-label", default="candidate")
    compare_parser.add_argument("--prefill-logprob-tolerance", "--mean-logprob-tolerance",
                                dest="prefill_logprob_tolerance", type=float, default=PREFILL_LOGPROB_ATOL)
    compare_parser.add_argument("--decode-logprob-tolerance", "--max-logprob-tolerance",
                                dest="decode_logprob_tolerance", type=float, default=DECODE_LOGPROB_ATOL)
    compare_parser.add_argument("--cosine-min", type=float, default=0.999)
    compare_parser.add_argument("--abs-diff-max", type=float, default=1e-2)
    compare_parser.add_argument("--require-token-match", action="store_true")
    compare_parser.add_argument("--require-logprob-tolerance", action="store_true")
    compare_parser.add_argument("--require-within-tolerance", action="store_true")
    compare_parser.set_defaults(handler=compare)
    args = parser.parse_args()
    args.handler(args, parser)


if __name__ == "__main__":
    main()
