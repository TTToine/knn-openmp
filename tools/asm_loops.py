#!/usr/bin/env python3
"""Lists the hot loops of a function in the knn binary: for every backward jump
whose body contains vfmadd231ps, prints the number of instructions, FMAs,
distinct accumulator registers, vector loads, and vector instructions that touch
the stack (spills). Used to check the micro-kernels (accumulators in registers,
no spill).

  python3 tools/asm_loops.py build/knn mrnrILi4ELi2          # substring of the mangled symbol
  python3 tools/asm_loops.py build/knn mrnrILi6ELi2 --show   # also prints the largest loop
"""
import re
import subprocess
import sys


def main():
    binary, pattern = sys.argv[1], sys.argv[2]
    show = "--show" in sys.argv
    asm = subprocess.run(["objdump", "-d", "--no-show-raw-insn", binary], capture_output=True, text=True).stdout
    for chunk in asm.split("\n\n"):
        head = chunk.strip().split("\n", 1)[0]
        if pattern not in head or ".cold" in head or "_omp_fn" not in head:
            continue
        lines = []
        for l in chunk.split("\n")[1:]:
            m = re.match(r"\s*([0-9a-f]+):\s+(.*)", l)
            if m:
                lines.append((int(m.group(1), 16), m.group(2).strip()))
        addr = {a: i for i, (a, _) in enumerate(lines)}
        print(head.split("<")[1][:60])
        loops = []
        for i, (a, ins) in enumerate(lines):
            m = re.match(r"j\w+\s+([0-9a-f]+)", ins)
            if m and int(m.group(1), 16) in addr and int(m.group(1), 16) <= a:
                body = [x for _, x in lines[addr[int(m.group(1), 16)]:i + 1]]
                fma = [x for x in body if x.startswith("vfmadd") and "ymm" in x]
                if fma and not any(x.startswith("call") for x in body):
                    loops.append(body)
        seen = set()
        for body in sorted(loops, key=len):
            fma = [x for x in body if x.startswith("vfmadd") and "ymm" in x]
            if any(len(body) > len(s) and set(s) <= set(body) for s in seen):
                continue            # an outer loop containing an already listed inner loop
            seen.add(tuple(body))
            accs = {x.split(",")[-1].strip() for x in fma}
            loads = [x for x in body if re.search(r"\(%r", x) and "ymm" in x and "rsp" not in x]
            stack = [x for x in body if "rsp" in x and ("ymm" in x or "xmm" in x)]
            subs = [x for x in body if x.startswith("vsubps")]
            print(f"  loop: {len(body):3d} insns, {len(fma):2d} vfmadd231ps on {len(accs):2d} accumulators, "
                  f"{len(subs):2d} vsubps, {len(loads):2d} vector memory operands, {len(stack)} stack accesses")
        if show and seen:
            print("\n".join("      " + x for x in max(seen, key=lambda b: sum(x.startswith("vfmadd") for x in b))))


if __name__ == "__main__":
    main()
