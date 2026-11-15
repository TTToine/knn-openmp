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
