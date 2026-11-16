#!/usr/bin/env bash
# Hardware/software description required in the report ("Experimental setup").
echo "== date";      date -Iseconds
echo "== kernel";    uname -a
echo "== cpu";       lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|MHz|L1d|L2|L3|Flags' | sed 's/Flags:.*\(avx2\).*/Flags: ... avx2 .../'
echo "== memory";    free -h | head -2
echo "== compiler";  g++ --version | head -1
echo "== cmake";     cmake --version | head -1
echo "== governor";  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "n/a (WSL2: set the Windows power plan to max performance)"
echo "== env";       env | grep -E '^OMP_' || echo "no OMP_* variables set"
# The two things that invalidated half of the first campaign: power source and
# Windows power plan. Both are visible from WSL2.
echo "== power"
for ps in /sys/class/power_supply/*; do
  [ -d "$ps" ] || continue
  echo "$(basename "$ps"): type=$(cat "$ps/type" 2>/dev/null) online=$(cat "$ps/online" 2>/dev/null) status=$(cat "$ps/status" 2>/dev/null) capacity=$(cat "$ps/capacity" 2>/dev/null)"
done
echo "== windows power plan"
(powercfg.exe /getactivescheme 2>/dev/null | tr -d '\r') || echo "n/a (powercfg.exe not reachable)"
echo "== physical memory (not visible from the VM: 'free' shows the WSL2 limit)"
echo "1 x 16 GB DDR4-2933, single channel (checked with Win32_PhysicalMemory): peak 23.5 GB/s"
echo "== thp";       cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null
echo "== build flags"
grep -m1 -o '\-O3.*-fopenmp' "$(dirname "$0")/../build/compile_commands.json" 2>/dev/null || echo "n/a"
echo "== commit";    git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null
