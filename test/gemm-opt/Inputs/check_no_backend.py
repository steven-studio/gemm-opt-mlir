#!/usr/bin/env python3
"""Fails when a source, CMake file or link line of gemm-opt-mlir depends on the Systolic backend."""
import pathlib
import re
import sys

src, obj = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
RULES = [
    (re.compile(r'#\s*include\s*[<"]Systolic/'), "includes a Systolic header"),
    (re.compile(r'\bMLIRSystolic\w*'), "names a Systolic library"),
    (re.compile(r'\b(mlir::)?systolic::'), "names the systolic namespace"),
    (re.compile(r'\bSystolic(Dialect|Ops)\b'), "names the Systolic dialect"),
    (re.compile(r'"systolic\.[a-z_]+'), "names a systolic operation"),
]
SUFFIXES = {".cpp", ".h", ".td", ".inc"}
files, bad = 0, []
for top in ("include", "lib", "tools", "research"):
    for path in sorted((src / top).rglob("*")):
        if not path.is_file() or not (path.suffix in SUFFIXES or path.name == "CMakeLists.txt"):
            continue
        files += 1
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            code = line.split("//", 1)[0] if path.suffix != ".txt" else line.split("#", 1)[0]
            for rule, what in RULES:
                if rule.search(code):
                    bad.append(f"{path.relative_to(src)}:{number}: {what}: {line.strip()}")
files += 1
for path in [src / "CMakeLists.txt"]:
    if re.search(r"Systolic", path.read_text().split("project(", 1)[1]):
        bad.append(f"{path.name}: names Systolic")
ninja = obj / "build.ninja"
if ninja.exists():
    for number, line in enumerate(ninja.read_text(errors="replace").splitlines(), 1):
        if "LINK_LIBRARIES" in line and re.search(r"[Ss]ystolic", line):
            bad.append(f"build.ninja:{number}: a link line names Systolic")
if bad:
    print("\n".join(bad))
    sys.exit(1)
print(f"no backend dependency: {files} files and the build's link lines scanned")
