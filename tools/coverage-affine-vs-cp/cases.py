"""Coverage corpus: one GEMM-like loop nest per case, in five categories.

Each case gives
  mlir  - the scf source read by CP (generalized-gemm-nest-report) and by MLIR's affine
          raising (mlir-opt --raise-scf-to-affine ...). Small static extents, so that CP's
          finite tier can enumerate its scope.
  c     - the same nest in C for Polly (clang 24 + Polly in opt). Extents of 64 (the
          structure is the same; small loops would be fully unrolled before Polly runs).
  cp    - extra reporter flags: --scope for scalar bounds, --table for index tables,
          --authorize for the root separations the record may assume.

Function arguments are numbered from 0; CP roots are the memref arguments in order.
"""

HDR = "  %c0 = arith.constant 0 : index\n  %c1 = arith.constant 1 : index\n  %c2 = arith.constant 2 : index\n  %c3 = arith.constant 3 : index\n  %c4 = arith.constant 4 : index\n  %c8 = arith.constant 8 : index\n"


def fn(args, body):
    return f"func.func @f({args}) {{\n{HDR}{body}  return\n}}\n"


def mac(a, b, c, ta, tb, tc):
    """%x = A[a]; %y = B[b]; C[c] += %x * %y (reassoc)."""
    return (f"%x = memref.load {a} : {ta}\n%y = memref.load {b} : {tb}\n%z = memref.load {c} : {tc}\n"
            f"%mm = arith.mulf %x, %y : f32\n%s = arith.addf %z, %mm fastmath<reassoc> : f32\n"
            f"memref.store %s, {c} : {tc}\n")


def nest(loops, inner):
    """loops: list of (iv, lo, hi, step, prologue)."""
    out, close = "", ""
    ind = "  "
    for iv, lo, hi, step, pro in loops:
        out += f"{ind}scf.for {iv} = {lo} to {hi} step {step} {{\n"
        for line in pro.strip().splitlines():
            out += f"{ind}  {line}\n"
        close = f"{ind}}}\n" + close
        ind += "  "
    for line in inner.strip().splitlines():
        out += f"{ind}{line}\n"
    return out + close


AUTH2 = "roots 0 and 2 are separate,roots 1 and 2 are separate"
CASES = []


def case(name, category, mlir, c, cp="", note="", affine=None):
    CASES.append(dict(name=name, category=category, mlir=mlir, c=c, cp=cp, note=note, affine=affine))


C_GEMM = "for (long i = 0; i < 64; i++) for (long j = 0; j < 64; j++) for (long k = 0; k < 64; k++) "

# ---- 1. Affine home ground -------------------------------------------------------------
case("dense_gemm", "1 affine",
     fn("%a: memref<4x2xf32>, %b: memref<2x2xf32>, %c: memref<4x2xf32>",
        nest([("%i", "%c0", "%c4", "%c1", ""), ("%j", "%c0", "%c2", "%c1", ""), ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%i, %k]", "%b[%k, %j]", "%c[%i, %j]", "memref<4x2xf32>", "memref<2x2xf32>", "memref<4x2xf32>"))),
     "void f(float A[restrict 64][64], float B[restrict 64][64], float C[restrict 64][64]) {\n  "
     + C_GEMM + "C[i][j] += A[i][k] * B[k][j];\n}\n",
     f"--authorize={AUTH2}")

case("batched_gemm", "1 affine",
     fn("%a: memref<2x4x2xf32>, %b: memref<2x2x2xf32>, %c: memref<2x4x2xf32>",
        nest([("%p", "%c0", "%c2", "%c1", ""), ("%i", "%c0", "%c4", "%c1", ""), ("%j", "%c0", "%c2", "%c1", ""),
              ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%p, %i, %k]", "%b[%p, %k, %j]", "%c[%p, %i, %j]", "memref<2x4x2xf32>", "memref<2x2x2xf32>",
                 "memref<2x4x2xf32>"))),
     "void f(float A[restrict 8][64][64], float B[restrict 8][64][64], float C[restrict 8][64][64]) {\n"
     "  for (long p = 0; p < 8; p++) " + C_GEMM + "C[p][i][j] += A[p][i][k] * B[p][k][j];\n}\n",
     f"--authorize={AUTH2}")

case("transposed_gemm", "1 affine",
     fn("%a: memref<2x4xf32>, %b: memref<2x2xf32>, %c: memref<4x2xf32>",
        nest([("%i", "%c0", "%c4", "%c1", ""), ("%j", "%c0", "%c2", "%c1", ""), ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%k, %i]", "%b[%k, %j]", "%c[%i, %j]", "memref<2x4xf32>", "memref<2x2xf32>", "memref<4x2xf32>"))),
     "void f(float A[restrict 64][64], float B[restrict 64][64], float C[restrict 64][64]) {\n  "
     + C_GEMM + "C[i][j] += A[k][i] * B[k][j];\n}\n",
     f"--authorize={AUTH2}")

# SYRK (PolyBench form, lower triangle): C[i][j] += A[i][k] * A[j][k], j <= i. A is both operands.
case("syrk_triangular", "1 affine",
     fn("%a: memref<4x2xf32>, %c: memref<4x4xf32>",
        nest([("%i", "%c0", "%c4", "%c1", "%ip1 = arith.addi %i, %c1 : index"), ("%j", "%c0", "%ip1", "%c1", ""),
              ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%i, %k]", "%a[%j, %k]", "%c[%i, %j]", "memref<4x2xf32>", "memref<4x2xf32>", "memref<4x4xf32>"))),
     "void f(float A[restrict 64][64], float C[restrict 64][64]) {\n"
     "  for (long i = 0; i < 64; i++) for (long j = 0; j <= i; j++) for (long k = 0; k < 64; k++) C[i][j] += A[i][k] * A[j][k];\n}\n",
     "--authorize=roots 0 and 1 are separate")

# TRMM (PolyBench form, in place): B[i][j] += A[k][i] * B[k][j], k in (i, M). B read and written.
case("trmm_inplace", "1 affine",
     fn("%a: memref<4x4xf32>, %b: memref<4x2xf32>",
        nest([("%i", "%c0", "%c4", "%c1", "%ip1 = arith.addi %i, %c1 : index"), ("%j", "%c0", "%c2", "%c1", ""),
              ("%k", "%ip1", "%c4", "%c1", "")],
             mac("%a[%k, %i]", "%b[%k, %j]", "%b[%i, %j]", "memref<4x4xf32>", "memref<4x2xf32>", "memref<4x2xf32>"))),
     "void f(float A[restrict 64][64], float B[restrict 64][64]) {\n"
     "  for (long i = 0; i < 64; i++) for (long j = 0; j < 64; j++) for (long k = i + 1; k < 64; k++) B[i][j] += A[k][i] * B[k][j];\n}\n",
     "--authorize=roots 0 and 1 are separate")

# Triangular product into a separate C: C[i][j] += A[i][k] * B[k][j], k <= i.
case("trmm_out_of_place", "1 affine",
     fn("%a: memref<4x4xf32>, %b: memref<4x2xf32>, %c: memref<4x2xf32>",
        nest([("%i", "%c0", "%c4", "%c1", "%ip1 = arith.addi %i, %c1 : index"), ("%j", "%c0", "%c2", "%c1", ""),
              ("%k", "%c0", "%ip1", "%c1", "")],
             mac("%a[%i, %k]", "%b[%k, %j]", "%c[%i, %j]", "memref<4x4xf32>", "memref<4x2xf32>", "memref<4x2xf32>"))),
     "void f(float A[restrict 64][64], float B[restrict 64][64], float C[restrict 64][64]) {\n"
     "  for (long i = 0; i < 64; i++) for (long j = 0; j < 64; j++) for (long k = 0; k <= i; k++) C[i][j] += A[i][k] * B[k][j];\n}\n",
     f"--authorize={AUTH2}")

case("step2", "1 affine",
     fn("%a: memref<8x2xf32>, %b: memref<2x2xf32>, %c: memref<8x2xf32>",
        nest([("%i", "%c0", "%c8", "%c2", ""), ("%j", "%c0", "%c2", "%c1", ""), ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%i, %k]", "%b[%k, %j]", "%c[%i, %j]", "memref<8x2xf32>", "memref<2x2xf32>", "memref<8x2xf32>"))),
     "void f(float A[restrict 128][64], float B[restrict 64][64], float C[restrict 128][64]) {\n"
     "  for (long i = 0; i < 128; i += 2) for (long j = 0; j < 64; j++) for (long k = 0; k < 64; k++) C[i][j] += A[i][k] * B[k][j];\n}\n",
     f"--authorize={AUTH2}")

case("stride_I_eq_2i", "1 affine",
     fn("%a: memref<8x2xf32>, %b: memref<2x2xf32>, %c: memref<8x2xf32>",
        nest([("%i", "%c0", "%c4", "%c1", "%ii = arith.muli %i, %c2 : index"), ("%j", "%c0", "%c2", "%c1", ""),
              ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%ii, %k]", "%b[%k, %j]", "%c[%ii, %j]", "memref<8x2xf32>", "memref<2x2xf32>", "memref<8x2xf32>"))),
     "void f(float A[restrict 128][64], float B[restrict 64][64], float C[restrict 128][64]) {\n  "
     + C_GEMM + "C[2 * i][j] += A[2 * i][k] * B[k][j];\n}\n",
     f"--authorize={AUTH2}")

# ---- 2. Parametric, shared %m ----------------------------------------------------------
case("param_shared_m", "2 parametric",
     fn("%a: memref<2x4x2xf32>, %b: memref<2x2x2xf32>, %c: memref<2x4x2xf32>, %m: index",
        nest([("%p", "%c0", "%c2", "%c1", ""), ("%i", "%c0", "%m", "%c1", ""), ("%j", "%c0", "%c2", "%c1", ""),
              ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%p, %i, %k]", "%b[%p, %k, %j]", "%c[%p, %i, %j]", "memref<2x4x2xf32>", "memref<2x2x2xf32>",
                 "memref<2x4x2xf32>"))),
     "void f(float A[restrict 8][128][64], float B[restrict 8][64][64], float C[restrict 8][128][64], long m) {\n"
     "  for (long p = 0; p < 8; p++) for (long i = 0; i < m; i++) for (long j = 0; j < 64; j++) for (long k = 0; k < 64; k++)\n"
     "    C[p][i][j] += A[p][i][k] * B[p][k][j];\n}\n",
     f"--scope=3:0:4 --authorize={AUTH2}")

# ---- 3. Data-dependent bounds ----------------------------------------------------------
case("rows_per_group", "3 data-dependent bound",
     fn("%a: memref<2x4x2xf32>, %b: memref<2x2x2xf32>, %c: memref<2x4x2xf32>, %rows: memref<2xindex>",
        nest([("%p", "%c0", "%c2", "%c1", "%m = memref.load %rows[%p] : memref<2xindex>"), ("%i", "%c0", "%m", "%c1", ""),
              ("%j", "%c0", "%c2", "%c1", ""), ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%p, %i, %k]", "%b[%p, %k, %j]", "%c[%p, %i, %j]", "memref<2x4x2xf32>", "memref<2x2x2xf32>",
                 "memref<2x4x2xf32>"))),
     "void f(float A[restrict 8][128][64], float B[restrict 8][64][64], float C[restrict 8][128][64], const long *restrict rows) {\n"
     "  for (long p = 0; p < 8; p++) for (long i = 0; i < rows[p]; i++) for (long j = 0; j < 64; j++) for (long k = 0; k < 64; k++)\n"
     "    C[p][i][j] += A[p][i][k] * B[p][k][j];\n}\n",
     f"--table=3:0:4 --authorize={AUTH2},roots 2 and 3 are separate")

# CSR-style grouped rows: group p owns rows [rowptr[p], rowptr[p+1]) of A and C.
case("csr_rowptr", "3 data-dependent bound",
     fn("%a: memref<4x2xf32>, %w: memref<2x2x2xf32>, %c: memref<4x2xf32>, %rowptr: memref<3xindex>",
        nest([("%p", "%c0", "%c2", "%c1",
               "%lo = memref.load %rowptr[%p] : memref<3xindex>\n%p1 = arith.addi %p, %c1 : index\n"
               "%hi = memref.load %rowptr[%p1] : memref<3xindex>"),
              ("%r", "%lo", "%hi", "%c1", ""), ("%j", "%c0", "%c2", "%c1", ""), ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%r, %k]", "%w[%p, %k, %j]", "%c[%r, %j]", "memref<4x2xf32>", "memref<2x2x2xf32>", "memref<4x2xf32>"))),
     "void f(float A[restrict 512][64], float W[restrict 8][64][64], float C[restrict 512][64], const long *restrict rowptr) {\n"
     "  for (long p = 0; p < 8; p++) for (long r = rowptr[p]; r < rowptr[p + 1]; r++) for (long j = 0; j < 64; j++) for (long k = 0; k < 64; k++)\n"
     "    C[r][j] += A[r][k] * W[p][k][j];\n}\n",
     f"--table=3:0:4 --index-invariants --authorize={AUTH2},roots 2 and 3 are separate",
     note="rowptr cells range independently over [0,4] (non-monotone tables give empty groups; groups never overlap). "
          "--index-invariants: the proof that no two groups share a row needs the loop invariant rowptr[p] <= r")

# ---- 4. Indirect access ----------------------------------------------------------------
case("gather_padded", "4 indirect",
     fn("%a: memref<3x2xf32>, %b: memref<2x2x2xf32>, %c: memref<2x2x2xf32>, %ids: memref<4xindex>, %m: index",
        nest([("%p", "%c0", "%c2", "%c1", ""), ("%i", "%c0", "%m", "%c1", ""), ("%j", "%c0", "%c2", "%c1", ""),
              ("%k", "%c0", "%c2", "%c1", "%r0 = arith.muli %p, %c2 : index\n%r = arith.addi %r0, %i : index\n"
                                          "%t = memref.load %ids[%r] : memref<4xindex>")],
             mac("%a[%t, %k]", "%b[%p, %k, %j]", "%c[%p, %i, %j]", "memref<3x2xf32>", "memref<2x2x2xf32>",
                 "memref<2x2x2xf32>"))),
     "void f(float X[restrict 4096][64], float W[restrict 8][64][64], float C[restrict 8][128][64], const long *restrict ids, long m) {\n"
     "  for (long p = 0; p < 8; p++) for (long i = 0; i < m; i++) for (long j = 0; j < 64; j++) for (long k = 0; k < 64; k++)\n"
     "    C[p][i][j] += X[ids[p * 128 + i]][k] * W[p][k][j];\n}\n",
     f"--table=3:0:2 --scope=4:0:2 --authorize={AUTH2},roots 2 and 3 are separate")

# Gather with compact output rows (the Triton generator's layout): C[p*CAP + i] += X[ids[p*CAP + i]] W[p].
case("gather_compact", "4 indirect",
     fn("%a: memref<3x2xf32>, %b: memref<2x2x2xf32>, %c: memref<4x2xf32>, %ids: memref<4xindex>, %m: index",
        nest([("%p", "%c0", "%c2", "%c1", ""), ("%i", "%c0", "%m", "%c1", ""), ("%j", "%c0", "%c2", "%c1", ""),
              ("%k", "%c0", "%c2", "%c1", "%r0 = arith.muli %p, %c2 : index\n%r = arith.addi %r0, %i : index\n"
                                          "%t = memref.load %ids[%r] : memref<4xindex>")],
             mac("%a[%t, %k]", "%b[%p, %k, %j]", "%c[%r, %j]", "memref<3x2xf32>", "memref<2x2x2xf32>", "memref<4x2xf32>"))),
     "void f(float X[restrict 4096][64], float W[restrict 8][64][64], float C[restrict 1024][64], const long *restrict ids, long m) {\n"
     "  for (long p = 0; p < 8; p++) for (long i = 0; i < m; i++) for (long j = 0; j < 64; j++) for (long k = 0; k < 64; k++)\n"
     "    C[p * 128 + i][j] += X[ids[p * 128 + i]][k] * W[p][k][j];\n}\n",
     f"--table=3:0:2 --scope=4:0:2 --authorize={AUTH2},roots 2 and 3 are separate")

# Scatter-add (MoE combine): C[ids[r]] += X[r] W[p]; ids may repeat, so cells take several contributions.
case("scatter_add", "4 indirect",
     fn("%a: memref<4x2xf32>, %b: memref<2x2x2xf32>, %c: memref<3x2xf32>, %ids: memref<4xindex>",
        nest([("%p", "%c0", "%c2", "%c1", ""), ("%i", "%c0", "%c2", "%c1", ""), ("%j", "%c0", "%c2", "%c1", ""),
              ("%k", "%c0", "%c2", "%c1", "%r0 = arith.muli %p, %c2 : index\n%r = arith.addi %r0, %i : index\n"
                                          "%t = memref.load %ids[%r] : memref<4xindex>")],
             mac("%a[%r, %k]", "%b[%p, %k, %j]", "%c[%t, %j]", "memref<4x2xf32>", "memref<2x2x2xf32>", "memref<3x2xf32>"))),
     "void f(float X[restrict 1024][64], float W[restrict 8][64][64], float C[restrict 4096][64], const long *restrict ids) {\n"
     "  for (long p = 0; p < 8; p++) for (long i = 0; i < 128; i++) for (long j = 0; j < 64; j++) for (long k = 0; k < 64; k++)\n"
     "    C[ids[p * 128 + i]][j] += X[p * 128 + i][k] * W[p][k][j];\n}\n",
     f"--table=3:0:2 --authorize={AUTH2},roots 2 and 3 are separate")

# ---- 5. Non-linear index ---------------------------------------------------------------
case("square_index", "5 non-linear",
     fn("%a: memref<3x2xf32>, %b: memref<2x2xf32>, %c: memref<9x2xf32>",
        nest([("%i", "%c0", "%c3", "%c1", "%ii = arith.muli %i, %i : index"), ("%j", "%c0", "%c2", "%c1", ""),
              ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%i, %k]", "%b[%k, %j]", "%c[%ii, %j]", "memref<3x2xf32>", "memref<2x2xf32>", "memref<9x2xf32>"))),
     "void f(float A[restrict 64][64], float B[restrict 64][64], float C[restrict 4096][64]) {\n  "
     + C_GEMM + "C[i * i][j] += A[i][k] * B[k][j];\n}\n",
     f"--authorize={AUTH2}")

case("div_index", "5 non-linear",
     fn("%a: memref<2x2xf32>, %b: memref<2x2xf32>, %c: memref<4x2xf32>",
        nest([("%i", "%c0", "%c4", "%c1", "%q = arith.divui %i, %c2 : index"), ("%j", "%c0", "%c2", "%c1", ""),
              ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%q, %k]", "%b[%k, %j]", "%c[%i, %j]", "memref<2x2xf32>", "memref<2x2xf32>", "memref<4x2xf32>"))),
     "void f(float A[restrict 32][64], float B[restrict 64][64], float C[restrict 64][64]) {\n  "
     + C_GEMM + "C[i][j] += A[i / 2][k] * B[k][j];\n}\n",
     f"--authorize={AUTH2}", note="constant divisor: quasi-affine (floordiv), which affine maps and Polly model")

case("mod_index", "5 non-linear",
     fn("%a: memref<2x2xf32>, %b: memref<2x2xf32>, %c: memref<4x2xf32>",
        nest([("%i", "%c0", "%c4", "%c1", "%q = arith.remui %i, %c2 : index"), ("%j", "%c0", "%c2", "%c1", ""),
              ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%q, %k]", "%b[%k, %j]", "%c[%i, %j]", "memref<2x2xf32>", "memref<2x2xf32>", "memref<4x2xf32>"))),
     "void f(float A[restrict 32][64], float B[restrict 64][64], float C[restrict 64][64]) {\n  "
     + C_GEMM + "C[i][j] += A[i % 2][k] * B[k][j];\n}\n",
     f"--authorize={AUTH2}", note="constant modulus: quasi-affine")

# Block-sparse rows: block b of A lands at block blk[b] of C (block size 2).
case("block_sparse", "5 non-linear",
     fn("%a: memref<4x2xf32>, %w: memref<2x2xf32>, %c: memref<4x2xf32>, %blk: memref<2xindex>",
        nest([("%b", "%c0", "%c2", "%c1", "%bb = memref.load %blk[%b] : memref<2xindex>\n%base = arith.muli %bb, %c2 : index\n"
                                          "%abase = arith.muli %b, %c2 : index"),
              ("%i", "%c0", "%c2", "%c1", "%row = arith.addi %base, %i : index\n%arow = arith.addi %abase, %i : index"),
              ("%j", "%c0", "%c2", "%c1", ""), ("%k", "%c0", "%c2", "%c1", "")],
             mac("%a[%arow, %k]", "%w[%k, %j]", "%c[%row, %j]", "memref<4x2xf32>", "memref<2x2xf32>", "memref<4x2xf32>"))),
     "void f(float A[restrict 4096][64], float W[restrict 64][64], float C[restrict 4096][64], const long *restrict blk) {\n"
     "  for (long b = 0; b < 32; b++) for (long i = 0; i < 128; i++) for (long j = 0; j < 64; j++) for (long k = 0; k < 64; k++)\n"
     "    C[blk[b] * 128 + i][j] += A[b * 128 + i][k] * W[k][j];\n}\n",
     f"--table=3:0:1 --authorize={AUTH2},roots 2 and 3 are separate")


# ---- Hand-written affine dialect, where the scf -> affine raising left a loop or an access
# that the affine dialect can express (triangular bounds, floordiv/mod by a constant). Fed to
# --affine-parallelize alone: the affine model's answer without the raising pass.
def _mac_affine(a, b, c, ta, tb, tc):
    return (f"%x = affine.load {a} : {ta}\n%y = affine.load {b} : {tb}\n%z = affine.load {c} : {tc}\n"
            f"%mm = arith.mulf %x, %y : f32\n%s = arith.addf %z, %mm fastmath<reassoc> : f32\n"
            f"affine.store %s, {c} : {tc}\n")


def _affine_fn(args, loops, body):
    out, close, ind = "", "", "  "
    for header in loops:
        out += f"{ind}{header} {{\n"
        close = f"{ind}}}\n" + close
        ind += "  "
    for line in body.strip().splitlines():
        out += f"{ind}{line}\n"
    return f"func.func @f({args}) {{\n{out}{close}  return\n}}\n"


MANUAL_AFFINE = {
    "syrk_triangular": _affine_fn("%a: memref<4x2xf32>, %c: memref<4x4xf32>",
        ["affine.for %i = 0 to 4", "affine.for %j = 0 to affine_map<(d0) -> (d0 + 1)>(%i)", "affine.for %k = 0 to 2"],
        _mac_affine("%a[%i, %k]", "%a[%j, %k]", "%c[%i, %j]", "memref<4x2xf32>", "memref<4x2xf32>", "memref<4x4xf32>")),
    "trmm_inplace": _affine_fn("%a: memref<4x4xf32>, %b: memref<4x2xf32>",
        ["affine.for %i = 0 to 4", "affine.for %j = 0 to 2", "affine.for %k = affine_map<(d0) -> (d0 + 1)>(%i) to 4"],
        _mac_affine("%a[%k, %i]", "%b[%k, %j]", "%b[%i, %j]", "memref<4x4xf32>", "memref<4x2xf32>", "memref<4x2xf32>")),
    "trmm_out_of_place": _affine_fn("%a: memref<4x4xf32>, %b: memref<4x2xf32>, %c: memref<4x2xf32>",
        ["affine.for %i = 0 to 4", "affine.for %j = 0 to 2", "affine.for %k = 0 to affine_map<(d0) -> (d0 + 1)>(%i)"],
        _mac_affine("%a[%i, %k]", "%b[%k, %j]", "%c[%i, %j]", "memref<4x4xf32>", "memref<4x2xf32>", "memref<4x2xf32>")),
    "div_index": _affine_fn("%a: memref<2x2xf32>, %b: memref<2x2xf32>, %c: memref<4x2xf32>",
        ["affine.for %i = 0 to 4", "affine.for %j = 0 to 2", "affine.for %k = 0 to 2"],
        _mac_affine("%a[%i floordiv 2, %k]", "%b[%k, %j]", "%c[%i, %j]", "memref<2x2xf32>", "memref<2x2xf32>", "memref<4x2xf32>")),
    "mod_index": _affine_fn("%a: memref<2x2xf32>, %b: memref<2x2xf32>, %c: memref<4x2xf32>",
        ["affine.for %i = 0 to 4", "affine.for %j = 0 to 2", "affine.for %k = 0 to 2"],
        _mac_affine("%a[%i mod 2, %k]", "%b[%k, %j]", "%c[%i, %j]", "memref<2x2xf32>", "memref<2x2xf32>", "memref<4x2xf32>")),
}
for _c in CASES:
    _c["affine"] = MANUAL_AFFINE.get(_c["name"])
