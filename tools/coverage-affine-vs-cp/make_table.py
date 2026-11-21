#!/usr/bin/env python3
"""results.json (run.py) -> coverage_table.md: the four classes per (case, tool), the reasons,
and the CP record's tiers.

  python3 make_table.py OUTDIR
"""

import json
import os
import sys

SHORT = {"ACCEPT_PARALLEL": "接受·平行", "ACCEPT_SERIAL": "接受·序列", "REJECT": "拒絕", "ERROR": "錯誤"}
HOW = {"guarded": "（執行期單射守門）", "csr": "（CSR 歸約）", "": ""}
TOOLS = [("cp_auto", "CP 全自動 (--auto)"), ("cp", "CP 手動旗標"), ("cp_main", "CP 不讀表"), ("mlir_affine", "MLIR affine (raise)"),
         ("mlir_affine_manual", "MLIR affine (手寫)"), ("polly", "Polly 預設"), ("polly_permissive", "Polly 寬鬆")]


def cell(r):
    return "—" if not r else SHORT[r["cls"]] + HOW.get(r.get("how", ""), "")


def main():
    d = sys.argv[1]
    data = json.load(open(os.path.join(d, "results.json")))
    out = ["# 覆蓋率對照表（A 組）", "",
           "四分類：接受·平行 / 接受·序列 / 拒絕 / 錯誤。定義見 tools/coverage-affine-vs-cp/run.py。", "",
           "| 類別 | 案例 | " + " | ".join(t[1] for t in TOOLS) + " |",
           "|---|---|" + "---|" * len(TOOLS)]
    for r in data["results"]:
        out.append(f"| {r['category']} | {r['name']} | " + " | ".join(cell(r[k]) for k, _ in TOOLS) + " |")
    out += ["", "## 每格的理由", ""]
    for r in data["results"]:
        out.append(f"### {r['name']} ({r['category']})")
        if r.get("note"):
            out.append(f"_{r['note']}_")
        out.append("")
        for k, label in TOOLS:
            x = r[k]
            if not x:
                continue
            out.append(f"- **{label}**: {cell(x)} — {x.get('reason', '')}")
            if k == "cp_auto" and x.get("premises"):
                out.append("  - 推得的前提（執行期守門）: " + "; ".join(
                    f"{kind} {arg}" + (f" ∈ [{lo}, {hi}]" if hi is not None else "") for kind, arg, lo, hi in x["premises"]))
                out.append("  - passes: " + json.dumps(x.get("passes", {}), ensure_ascii=False))
        c = r["cp"]
        if c.get("status"):
            out.append(f"  - CP record: status {c['status']}, ordered fold {c['fold']}, computation symbolic "
                       f"{c['computation_symbolic']}, domain symbolic {c['domain_symbolic']}, coordinate "
                       f"{c['coordinate']}, logical tier {c['logical']}, output injective {c['output_injective']}, "
                       f"{c['seconds']} s")
            if c["refusals"]:
                out.append("  - CP refusals: " + "; ".join(x[:160] for x in c["refusals"]))
        out.append("")
    rt = [r for r in data["results"] if r.get("runtime")]
    if rt:
        out += ["## 執行期情境（GPU，--auto record 自動降階為 Triton，對照逐純量 float64 直譯）", "",
                "| 案例 | 計畫 | 情境 | 索引表 | 走的路徑 | 正確 |", "|---|---|---|---|---|---|"]
        for r in rt:
            plan = r["cp_auto"].get("plan", {})
            ptxt = plan.get("fast_path", "?") + (f" / 重複→{plan['on_repeated_index']}" if plan.get("on_repeated_index") not in (None, "n/a") else "")
            if "scenarios" not in r["runtime"]:
                out.append(f"| {r['name']} | {ptxt} | — | — | {r['runtime']} | — |")
                continue
            for s in r["runtime"]["scenarios"]:
                if s.get("skipped"):
                    out.append(f"| {r['name']} | {ptxt} | {s['scenario']} | — | 略過：{s['skipped']} | — |")
                    continue
                path = s["path"] + ("" if s["path"] != "refused" else "：" + s.get("why", "")[:60])
                err = f"（誤差 {s['max_abs_err']:.1e}）" if "max_abs_err" in s else ""
                out.append(f"| {r['name']} | {ptxt} | {s['scenario']} | {s['tables']} | {path} | {s['ok']}{err} |")
        out.append("")
    path = os.path.join(d, "coverage_table.md")
    open(path, "w").write("\n".join(out) + "\n")
    print(path)


if __name__ == "__main__":
    main()
