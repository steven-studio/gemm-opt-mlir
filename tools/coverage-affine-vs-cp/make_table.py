#!/usr/bin/env python3
"""results.json (run.py) -> coverage_table.md: the four classes per (case, tool), the reasons,
and the CP record's tiers.

  python3 make_table.py OUTDIR
"""

import json
import os
import sys

SHORT = {"ACCEPT_PARALLEL": "接受·平行", "ACCEPT_SERIAL": "接受·序列", "REJECT": "拒絕", "ERROR": "錯誤"}
TOOLS = [("cp", "CP (本分支)"), ("cp_main", "CP (main)"), ("mlir_affine", "MLIR affine (raise)"),
         ("mlir_affine_manual", "MLIR affine (手寫)"), ("polly", "Polly 預設"), ("polly_permissive", "Polly 寬鬆")]


def cell(r):
    return "—" if not r else SHORT[r["cls"]]


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
            out.append(f"- **{label}**: {SHORT[x['cls']]} — {x.get('reason', '')}")
        c = r["cp"]
        if c.get("status"):
            out.append(f"  - CP record: status {c['status']}, ordered fold {c['fold']}, computation symbolic "
                       f"{c['computation_symbolic']}, domain symbolic {c['domain_symbolic']}, coordinate "
                       f"{c['coordinate']}, logical tier {c['logical']}, output injective {c['output_injective']}, "
                       f"{c['seconds']} s")
            if c["refusals"]:
                out.append("  - CP refusals: " + "; ".join(x[:160] for x in c["refusals"]))
        out.append("")
    path = os.path.join(d, "coverage_table.md")
    open(path, "w").write("\n".join(out) + "\n")
    print(path)


if __name__ == "__main__":
    main()
