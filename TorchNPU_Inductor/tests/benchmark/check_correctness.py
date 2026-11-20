"""Correctness checks for one model and one track.
Env vars: BENCH_MODEL, BENCH_TRACK(eager|default|inductor), BENCH_TP, BENCH_OUT,
BENCH_VERIFY_REPEATS (default 1).
Outputs: acc_track.jsonl, acc_self_compare.json and correctness_complete.json.
"""
import json, os, time, subprocess, sys, threading
from pathlib import Path

VERIFY_REPEATS = int(os.environ.get("BENCH_VERIFY_REPEATS", "1"))
STAGE = "starting"
STARTED = time.monotonic()


def progress(stage, **fields):
    global STAGE
    STAGE = stage
    event = dict(model=os.environ.get("BENCH_MODEL", ""),
                 track=os.environ.get("BENCH_TRACK", ""), stage=stage,
                 elapsed_s=round(time.monotonic() - STARTED), **fields)
    print("BENCH_EVENT " + json.dumps(event, ensure_ascii=False), flush=True)


def heartbeat():
    while True:
        time.sleep(60)
        print("BENCH_HEARTBEAT " + json.dumps(dict(
            model=MODEL, track=TRACK, stage=STAGE,
            elapsed_s=round(time.monotonic() - STARTED)), ensure_ascii=False), flush=True)

TRACK = os.environ["BENCH_TRACK"]
MODEL = os.environ["BENCH_MODEL"]
TP = int(os.environ.get("BENCH_TP", "1"))
OUT = os.environ["BENCH_OUT"]
os.makedirs(OUT, exist_ok=True)

PROMPTS_SHORT_ACC = [
    "The capital of France is",
    "请比较光合作用与细胞呼吸的能量转换过程，并说明二者之间的联系。",
    "请解方程组：3x + 2y = 16，x - y = 2。写出完整计算过程。",
    "Write a Python function that reverses a string.",
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
    _sections = [_heading.format(index + 1) + "\n" + "\n".join(_paras) for index in range(8)]
    PROMPTS_LONG_ACC.append("\n\n".join(_sections) + "\n\n" + _question)
PROMPTS_ACC = PROMPTS_SHORT_ACC + PROMPTS_LONG_ACC


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def obviously_garbled(text):
    stripped = text.strip()
    return not stripped or "!" * 32 in stripped or "\ufffd" * 8 in stripped


def serialize_logprobs(output):
    positions = []
    for position, candidates in enumerate(output.logprobs or []):
        values = []
        for token_id, item in candidates.items():
            values.append({"token_id": token_id, "token": item.decoded_token,
                           "logprob": item.logprob, "rank": item.rank})
        values.sort(key=lambda item: (item["rank"] is None, item["rank"] or 0, -item["logprob"]))
        positions.append({"position": position, "candidates": values})
    return positions


def accuracy(llm, SamplingParams):
    runs = []
    for repeat in range(1, VERIFY_REPEATS + 1):
        progress("accuracy", repeat=repeat)
        outs = llm.generate(PROMPTS_ACC, SamplingParams(
            max_tokens=1024, temperature=0.0, seed=0, logprobs=5), use_tqdm=False)
        rows = [{"idx": i, "prompt": PROMPTS_ACC[i], "text": r.outputs[0].text,
                 "token_ids": list(r.outputs[0].token_ids),
                 "input_tokens": len(r.prompt_token_ids),
                 "output_tokens": len(r.outputs[0].token_ids),
                 "logprobs": serialize_logprobs(r.outputs[0])} for i, r in enumerate(outs)]
        name = "acc_track.jsonl" if repeat == 1 else f"acc_repeat{repeat}.jsonl"
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        runs.append(rows)
    stable = all([r["token_ids"] for r in rows] == [r["token_ids"] for r in runs[0]]
                 for rows in runs[1:])
    save_json(os.path.join(OUT, "acc_self_compare.json"),
              {"repeats": VERIFY_REPEATS, "token_stable": stable})
    cross_equal = True
    if TRACK == "inductor" and (Path(OUT).parent / "eager/acc_track.jsonl").is_file():
        subprocess.run([sys.executable, str(Path(__file__).with_name("compare_tracks.py")),
                        str(Path(OUT).parent)], check=True)
        comparison = json.loads((Path(OUT).parent / "compare.json").read_text())
        cross_equal = comparison["token_identical"] == len(PROMPTS_ACC)
        eager_self = json.loads((Path(OUT).parent / "eager/acc_self_compare.json").read_text())
        stable = stable and eager_self["token_stable"]
        progress("accuracy_compared", token_identical=comparison["token_identical"],
                 n_prompts=comparison["n_prompts"], token_stable=stable)
    if any(obviously_garbled(row["text"]) for row in runs[0]):
        failure = {"reason": "empty or obviously garbled output"}
        save_json(Path(OUT) / "correctness_failed.json", failure)
        progress("correctness_failed", reason=failure["reason"])
        return True
    if not stable or not cross_equal:
        review = {"action": "post_run_review",
                  "reason": "token mismatch or unstable output; record for post-run review",
                  "token_stable": stable, "cross_equal": cross_equal}
        save_json(Path(OUT) / "review_required.json", review)
        progress("accuracy_review_required", token_stable=stable, cross_equal=cross_equal)
    return True


def main():
    if TRACK not in ("eager", "default", "inductor") or VERIFY_REPEATS < 1:
        raise ValueError("Invalid track or repeat count")
    threading.Thread(target=heartbeat, daemon=True).start()
    progress("engine_initializing")
    from vllm import LLM, SamplingParams
    kw = dict(compilation_config={"backend": "inductor"}) if TRACK == "inductor" else {}
    if TRACK == "inductor" and os.environ.get("BENCH_CUDAGRAPH_MODE"):
        kw["compilation_config"]["cudagraph_mode"] = os.environ["BENCH_CUDAGRAPH_MODE"]
    if TRACK == "eager":
        kw["enforce_eager"] = True
    if os.environ.get("BENCH_ENABLE_EXPERT_PARALLEL") == "1":
        kw["enable_expert_parallel"] = True
    if os.environ.get("BENCH_TOKENIZER_MODE"):
        kw["tokenizer_mode"] = os.environ["BENCH_TOKENIZER_MODE"]
    if os.environ.get("BENCH_MTP") == "1":
        kw["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": int(os.environ.get("BENCH_MTP_TOKENS", "1")),
        }
    if os.environ.get("BENCH_ENFORCE_EAGER") == "1":
        kw["enforce_eager"] = True
    init_start = time.monotonic()
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=8192,
              max_num_seqs=32, tensor_parallel_size=TP, seed=0,
              gpu_memory_utilization=0.80, enable_prefix_caching=False,
              disable_log_stats=False, **kw)
    report = {"model": MODEL, "track": TRACK, "tp": TP,
              "engine_init_s": round(time.monotonic() - init_start, 2),
              "verify_repeats": VERIFY_REPEATS, "test_scope": "correctness_only",
              "engine_kwargs": dict(dtype="bfloat16", max_model_len=8192,
                  max_num_seqs=32, tensor_parallel_size=TP, seed=0,
                  gpu_memory_utilization=0.80, enable_prefix_caching=False,
                  disable_log_stats=False, **kw)}
    save_json(Path(OUT) / "run_config.json", report)
    progress("engine_ready", engine_init_s=report["engine_init_s"])
    if not accuracy(llm, SamplingParams):
        return
    save_json(Path(OUT) / "correctness_complete.json",
              {"model": MODEL, "track": TRACK, "scope": "correctness_only",
               "cudagraph_mode": os.environ.get("BENCH_CUDAGRAPH_MODE", "default")})
    progress("track_done", scope="correctness_only")


if __name__ == "__main__":
    main()
