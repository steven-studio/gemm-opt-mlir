"""Probes of what the generated dispatcher does and does not check (mixed-Mp conditions)."""
import importlib.util, sys, json, re, glob, os
import torch
sys.path.insert(0, "/home/steven-studio/work/gemm-opt-mlir/tools/grouped-gemm-triton")
import grouped_gemm_triton as gen
rec = "/home/steven-studio/work/gemm-opt-mlir/tools/grouped-gemm-triton/test/moe_grouped.record.json"
assert gen.main([rec, "-o", "kernels.py"]) == 0
spec = importlib.util.spec_from_file_location("k", "kernels.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
E, K, N = mod.E, mod.K, mod.N
dev = "cuda"; t32 = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)
out = {}
def attempt(name, fn):
    try:
        fn(); torch.cuda.synchronize(); out[name] = "accepted (no error)"
    except Exception as e:
        out[name] = f"raised {type(e).__name__}: {str(e)[:160]}"
    print(name, "->", out[name])

# 1. X aliases C (separation of roots 0 and 2 is an authorization of the record).
buf = torch.randn(64, max(K, N), device=dev)
x_alias = buf[:, :K]; c_alias = buf[:, :N]
w = torch.randn(E, K, N, device=dev)
attempt("X aliases C", lambda: mod.grouped_gemm(x_alias, w, c_alias, torch.arange(64, dtype=torch.int32, device=dev), t32([0, 8, 16, 24]), t32([8, 8, 8, 8])))
# 2. W aliases C storage.
cw = torch.randn(E * K * N, device=dev)
attempt("W aliases C", lambda: mod.grouped_gemm(torch.randn(32, K, device=dev), cw.view(E, K, N), cw.view(-1, N), torch.arange(32, dtype=torch.int32, device=dev), t32([0, 1, 2, 3]), t32([1, 1, 1, 1])))
# 3. sorted_token_ids out of range of X (index 10**6 into a 16-row X).
ids = torch.zeros(16, dtype=torch.int32, device=dev); ids[3] = 10**6
attempt("ids out of range of X", lambda: mod.grouped_gemm(torch.randn(16, K, device=dev), w, torch.zeros(16, N, device=dev), ids, t32([0, 4, 8, 12]), t32([4, 4, 4, 4])))
# 4. negative id
ids2 = torch.zeros(16, dtype=torch.int32, device=dev); ids2[0] = -5
attempt("negative id", lambda: mod.grouped_gemm(torch.randn(16, K, device=dev), w, torch.zeros(16, N, device=dev), ids2, t32([0, 4, 8, 12]), t32([4, 4, 4, 4])))
json.dump(out, open("probe_results.json", "w"), indent=1)
