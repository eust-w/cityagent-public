#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
失分病理分析：79 份模拟考答卷 vs 官方标签
回答三个问题：
  1. 两个时态封闭字段（是否新增/上一次定检状态）具体怎么错？（混淆矩阵）
  2. 基本信息里哪些字段最容易错？
  3. 错误和「报告有没有对比年份」「输入是否被截断」有没有关系？
"""
import json, re, sys
from collections import Counter, defaultdict

OUT = "eval_results/outputs_run1.jsonl"
SFT = "data/sft/sft_train.jsonl"

def parse_json_loose(text):
    text = text.strip()
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    chunk = text[i:j + 1]
    try:
        return json.loads(chunk)
    except Exception:
        pass
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", chunk))
    except Exception:
        return None

def norm(v):
    if v is None:
        return ""
    s = str(v).strip().replace(" ", "").replace("　", "").replace("（", "(").replace("）", ")")
    try:
        numeric = re.sub(r'[^\d.\-]', '', s)
        return f"num:{float(numeric):.2f}"
    except Exception:
        return s

def rows_to_map(rows, keys):
    m = {}
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, dict):
                m[tuple(norm(r.get(f)) for f in keys)] = r
    return m

samples = [json.loads(l) for l in open(OUT, encoding="utf-8")]
meta_by_report = {}
for l in open(SFT, encoding="utf-8"):
    d = json.loads(l)
    if d["meta"].get("year") == "2018年":
        meta_by_report[d["meta"]["report"]] = d["meta"].get("compare_kind", "未知")

# ---------- 1. 时态字段混淆 ----------
conf = defaultdict(Counter)   # field -> Counter[(gold, pred)]
closed = ["是否新增", "上一次定检状态", "发展程度"]
basic_field_stat = defaultdict(lambda: [0, 0])   # field -> [hit, total]
row_recall = [0, 0]  # 病害明细行召回: pred 里找得到这行吗
by_kind = defaultdict(lambda: [0, 0])  # compare_kind -> [行命中, 行总数]
by_trunc = defaultdict(lambda: [0, 0])
examples = []

for s in samples:
    pred = parse_json_loose(s["raw_output"])
    gold = parse_json_loose(s["gold_output"])
    if pred is None or gold is None:
        print(f"!! JSON 解析失败: {s['report']}  (finish={s['finish_reason']}, out_tok={s['output_tokens']})")
        continue
    kind = meta_by_report.get(s["report"], "未知")
    trunc = s["truncated_input"]

    # 基本信息逐字段
    gb = gold.get("基本信息") or {}
    pb = pred.get("基本信息") or {}
    for k in gb:
        basic_field_stat[k][1] += 1
        if norm(pb.get(k)) == norm(gb[k]):
            basic_field_stat[k][0] += 1

    # 病害明细
    gm = rows_to_map(gold.get("病害明细"), ["病害部位", "病害类型"])
    pm = rows_to_map(pred.get("病害明细"), ["病害部位", "病害类型"])
    for k, grow in gm.items():
        prow = pm.get(k)
        row_recall[1] += 1
        by_kind[kind][1] += 1
        by_trunc[trunc][1] += 1
        if prow is not None:
            row_recall[0] += 1
            by_kind[kind][0] += 1
            by_trunc[trunc][0] += 1
            for f in closed:
                g, p = norm(grow.get(f)), norm(prow.get(f))
                if g:
                    conf[f][(g, p)] += 1
                    if g != p and f != "发展程度" and len(examples) < 30:
                        examples.append((s["report"], f, grow.get(f), prow.get(f)))

# ---------- 输出 ----------
print("\n===== 1. 时态字段混淆矩阵（gold → pred  Top8）=====")
for f in closed:
    total = sum(conf[f].values())
    correct = sum(c for (g, p), c in conf[f].items() if g == p)
    print(f"\n【{f}】 正确率 {correct}/{total} = {correct/total*100:.1f}%")
    for (g, p), c in conf[f].most_common(8):
        mark = "  ✅" if g == p else "  ❌"
        print(f"  {c:>4} 次 | 标签「{g[:28]}」 → 模型「{p[:28]}」{mark}")

print("\n===== 2. 基本信息字段准确率排行 =====")
for k, (h, t) in sorted(basic_field_stat.items(), key=lambda x: x[1][0]/x[1][1]):
    print(f"  {h/t*100:5.1f}%  {k}  ({h}/{t})")

print(f"\n===== 3. 病害明细【行召回】（标签里的行，模型答案里有没有对应行）=====")
print(f"  总体: {row_recall[0]}/{row_recall[1]} = {row_recall[0]/row_recall[1]*100:.1f}%")
print("  按报告类型:")
for k, (h, t) in by_kind.items():
    print(f"    {k}: {h}/{t} = {h/t*100:.1f}%")
print("  按输入是否被截断:")
for k, (h, t) in by_trunc.items():
    print(f"    截断={k}: {h}/{t} = {h/t*100:.1f}%")

print("\n===== 4. 时态字段错误实例（前 10 条）=====")
for r, f, g, p in examples[:10]:
    print(f"  [{r}] {f}: 标签「{g}」 模型「{p}」")
