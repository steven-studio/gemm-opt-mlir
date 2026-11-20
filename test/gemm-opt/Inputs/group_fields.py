"""Print gemm.group_axis and gemm.parametric_extents of each nest of a report, one line each."""
import json
import sys

for entry in json.load(open(sys.argv[1])):
    for nest in entry["nests"]:
        gemm = nest.get("gemm")
        if gemm is None:
            print(f"{entry['function']} nest {nest['nest']}: {nest['status']} no gemm")
            continue
        print(f"{entry['function']} nest {nest['nest']}: {nest['status']}")
        print("group_axis " + json.dumps(gemm["group_axis"], sort_keys=True))
        print("parametric_extents " + json.dumps(gemm["parametric_extents"], sort_keys=True))
