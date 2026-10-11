#!/usr/bin/env python3
"""Pre-declared backend choice (written 2026-10-05, before any H200 Qwen3.8 measurement).

Candidates fa3, flashinfer, each run on the smoke points (SMOKE_POINTS in run_grid.pbs) with --smoke iterations.
Rule: a backend that lacks any smoke result is ineligible. Among eligible ones pick the lowest geometric mean of
(decoder step-ms medians + prefill forward-ms medians), all smoke latencies weighted equally. Tie (<1%) -> fa3.
Writes FROZEN_BACKEND (one word) and backend_choice.json; refuses to overwrite.
"""
import json, math, sys
from pathlib import Path

smoke, need = Path(sys.argv[1]), sys.argv[2:]
res = {}
for b in ("fa3", "flashinfer"):
    d, lat = smoke / b, []
    for pid in need:
        fs = sorted(d.glob(pid + "*.json"))
        if not fs:
            lat = None; break
        j = json.loads(fs[0].read_text())
        if pid.startswith("dec_"):
            lat.append(float(j["summaries"][0]["total_ms_per_step_median"]))
        else:
            ok = [float(p["forward_ms_median"]) for p in j["points"] if p.get("status") == "ok"]
            if len(ok) != len(j["points"]) or not ok:
                lat = None; break
            lat += ok
    res[b] = None if lat is None else dict(latencies=lat, geomean=math.exp(sum(map(math.log, lat)) / len(lat)))
ok = {b: r for b, r in res.items() if r}
if not ok:
    sys.exit("no backend passed the smoke")
best = min(ok, key=lambda b: ok[b]["geomean"])
if "fa3" in ok and ok["fa3"]["geomean"] <= ok[best]["geomean"] * 1.01:
    best = "fa3"
out = smoke.parent / "FROZEN_BACKEND"
if out.exists():
    sys.exit(f"{out} exists")
(smoke.parent / "backend_choice.json").write_text(json.dumps(dict(results=res, chosen=best), indent=1) + "\n")
out.write_text(best + "\n")
print(best, json.dumps({b: (r and round(r["geomean"], 3)) for b, r in res.items()}))
